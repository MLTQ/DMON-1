"""Reproducible Tiny Shakespeare benchmark for SOL and matched controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .baselines import (
    CausalCharacterTransformer,
    CharacterGRU,
    evaluate_gru,
    evaluate_transformer,
    match_gru_hidden_size,
    match_transformer_hidden_size,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .convergence import summarize_convergence
from .evaluate import evaluate_state_ablations
from .model import SolConfig, SparseAxonField
from .routing import RoutingTrafficConfig, routing_traffic_summary
from .schedule import (
    cosine_decay_learning_rate,
    set_optimizer_learning_rate,
)
from .stability import (
    load_sol_evaluation_history,
    summarize_exploratory_survival,
    summarize_stability,
)
from .structure import StructuralConfig, structural_summary
from .stream import CharacterVocabulary, ContinuousCharStream
from .topology import analyze_topology
from .train import ContinuousTrainer, generate

DEFAULT_CORPUS = Path("data/tinyshakespeare/input.txt")


def _device_memory(device: torch.device) -> dict[str, int | str]:
    """Return comparable current and peak memory telemetry for a device."""

    memory: dict[str, int | str] = {"device_type": device.type}
    if device.type != "cuda":
        return memory
    properties = torch.cuda.get_device_properties(device)
    memory.update(
        {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "total_bytes": properties.total_memory,
        }
    )
    return memory


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


def _split_corpus(path: Path, fraction: float = 0.9) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("benchmark corpus must contain at least 100 characters")
    split = int(len(text) * fraction)
    return text[:split], text[split:]


def _optimization_config(
    args: argparse.Namespace,
    base_learning_rate: float | None = None,
) -> dict[str, float | int]:
    """Return the optimizer policy recorded with checkpoints and summaries."""

    return {
        "base_learning_rate": (
            args.learning_rate
            if base_learning_rate is None
            else base_learning_rate
        ),
        "learning_rate_decay_start": args.learning_rate_decay_start,
        "learning_rate_decay_end": args.learning_rate_decay_end,
        "minimum_learning_rate_ratio": args.minimum_learning_rate_ratio,
    }


def _set_learning_rate_for_update(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    update: int,
    base_learning_rate: float | None = None,
) -> float:
    """Apply the deterministic learning rate for the upcoming update."""

    learning_rate = cosine_decay_learning_rate(
        (
            args.learning_rate
            if base_learning_rate is None
            else base_learning_rate
        ),
        update,
        args.learning_rate_decay_start,
        args.learning_rate_decay_end,
        args.minimum_learning_rate_ratio,
    )
    set_optimizer_learning_rate(optimizer, learning_rate)
    return learning_rate


def _sol_config(args: argparse.Namespace, vocab_size: int) -> SolConfig:
    metabolic = {
        "stimulation_gain": args.external_energy_gain,
        "energy_transport_rate": args.energy_transport_rate,
        "energy_maintenance_flow": args.energy_maintenance_flow,
        "quiescence_energy": args.quiescence_energy,
        "full_activity_energy": args.full_activity_energy,
    }
    if args.no_metabolism:
        metabolic.update(
            {
                "energy_start": 1.0,
                "basal_cost": 0.0,
                "activity_cost": 0.0,
                "stimulation_gain": 0.0,
                "energy_transport_rate": 0.0,
                "energy_maintenance_flow": 0.0,
            }
        )
    reward: dict[str, float] = {
        "reward_gain": args.cell_reward_gain,
        "fast_plasticity_gain": args.fast_plasticity_gain,
        "reward_baseline_decay": args.reward_baseline_decay,
        "backward_credit_gain": args.backward_credit_gain,
        "backward_credit_decay": args.backward_credit_decay,
        "output_error_credit_gain": args.output_error_credit_gain,
        "output_error_credit_decay": args.output_error_credit_decay,
        "eligibility_routed_output_credit": (
            args.eligibility_routed_output_credit
        ),
        "eligibility_routing_gain": args.eligibility_routing_gain,
        "reward_plastic_output_credit_routing": (
            args.reward_plastic_output_credit_routing
        ),
        "exploratory_output_credit_routing": (
            args.exploratory_output_credit_routing
        ),
        "credit_routing_preference_decay": (
            args.credit_routing_preference_decay
        ),
        "credit_routing_plasticity_gain": (
            args.credit_routing_plasticity_gain
        ),
        "credit_routing_preference_limit": (
            args.credit_routing_preference_limit
        ),
        "structural_probe_gain": (
            args.structural_probe_gain
            if args.structural_plasticity or args.structural_probes_only
            else 0.0
        ),
    }
    if args.no_reward:
        reward.update(
            {
                "reward_gain": 0.0,
                "backward_credit_gain": 0.0,
                "output_error_credit_gain": 0.0,
                "fast_plasticity_gain": 0.0,
            }
        )
    elif args.no_fast_plasticity:
        reward["fast_plasticity_gain"] = 0.0
    return SolConfig(
        vocab_size=vocab_size,
        cells=args.cells,
        channels=args.channels,
        dendrites=args.dendrites,
        initial_active_dendrites=args.initial_active_dendrites,
        sensory_cells=args.sensory_cells,
        output_cells=args.output_cells,
        message_steps=args.message_steps,
        topology_seed=args.seed,
        **metabolic,
        **reward,
    )


def _run_sol(
    args: argparse.Namespace,
    train_text: str,
    validation_text: str,
    vocabulary: CharacterVocabulary,
) -> dict[str, Any]:
    device = torch.device(args.device)
    latest = args.out_dir / "latest.pt"
    if args.resume and latest.exists():
        trainer, checkpoint_metadata = load_checkpoint(latest, train_text, device)
        print(f"[sol] resumed update {trainer.updates} from {latest}")
    else:
        checkpoint_metadata = {}
        model = SparseAxonField(_sol_config(args, len(vocabulary)))
        trainer = ContinuousTrainer(
            model,
            train_text,
            vocabulary,
            batch_size=args.batch,
            chunk_length=args.chunk,
            learning_rate=args.learning_rate,
            frozen_parameters=(
                ("edge_weight", "edge_bias") if args.freeze_edges else ()
            ),
            structural_config=StructuralConfig(
                enabled=(
                    args.structural_plasticity
                    or args.structural_probes_only
                ),
                allow_rewiring=args.structural_plasticity,
                interval=args.structural_interval,
                warmup_updates=args.structural_warmup,
                replacements_per_phase=args.structural_replacements,
                confirmation_phases=args.structural_confirmation_phases,
                require_global_fitness=args.structural_global_fitness,
                global_fitness_margin=(
                    args.structural_global_fitness_margin
                ),
                credit_decay=args.structural_credit_decay,
                credit_margin=args.structural_credit_margin,
                min_edge_age=args.structural_min_edge_age,
                growth_cost=args.structural_growth_cost,
                min_endpoint_energy=args.structural_min_energy,
                probation_updates=args.structural_probation_updates,
                probation_margin=args.structural_probation_margin,
                probation_baseline_decay=(
                    args.structural_probation_baseline_decay
                ),
                probation_exploratory_traffic=(
                    args.structural_probation_exploratory_traffic
                ),
                variable_fan_in=args.structural_variable_fan_in,
                min_active_dendrites=(
                    args.structural_min_active_dendrites
                ),
                prune_usage_threshold=(
                    args.structural_prune_usage_threshold
                ),
                prune_credit_threshold=(
                    args.structural_prune_credit_threshold
                ),
                usage_gain=args.structural_usage_gain,
                vector_credit_gain=(
                    args.structural_vector_credit_gain
                ),
                locality_gain=args.structural_locality_gain,
            ),
            routing_traffic_config=RoutingTrafficConfig(
                enabled=args.exploratory_output_credit_routing,
                interval=args.routing_traffic_interval,
                warmup_updates=args.routing_traffic_warmup,
                trial_updates=args.routing_traffic_updates,
                boundary_interval=args.eval_every,
                randomization_alpha=(
                    args.routing_traffic_randomization_alpha
                ),
                margin=args.routing_traffic_margin,
                proposal_step=args.routing_traffic_step,
                minimum_eligibility=(
                    args.routing_traffic_minimum_eligibility
                ),
            ),
            device=device,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    chance_bpc = math.log(len(vocabulary)) / math.log(2.0)
    print(
        f"[sol] params={parameter_count:,} device={device} "
        f"updates={trainer.updates}->{args.updates} chance_bpc={chance_bpc:.3f}"
    )
    best_bpc = float(checkpoint_metadata.get("best_bpc", math.inf))
    started = time.monotonic()
    last_eval: dict[str, Any] = {}
    evaluation_history = load_sol_evaluation_history(
        args.out_dir / "metrics.jsonl"
    )

    while trainer.updates < args.updates:
        learning_rate = _set_learning_rate_for_update(
            trainer.optimizer,
            args,
            trainer.updates + 1,
            trainer.learning_rate,
        )
        metrics = trainer.step()
        update = trainer.updates
        if update == 1 or update % args.log_every == 0:
            elapsed = time.monotonic() - started
            row = {
                "kind": "train",
                "model": "sol",
                "update": update,
                "tokens": update * args.batch * args.chunk,
                "tokens_per_second": (
                    (update * args.batch * args.chunk)
                    / max(elapsed, 1e-9)
                ),
                "learning_rate": learning_rate,
                "device_memory": _device_memory(device),
                **asdict(metrics),
            }
            _append_jsonl(args.out_dir / "metrics.jsonl", row)
            print(
                f"[sol:{update:6d}] loss={metrics.loss:.4f} "
                f"bpc={metrics.loss / math.log(2.0):.3f} "
                f"energy={metrics.mean_energy:.3f} "
                f"viable={metrics.mean_viability:.3f} "
                f"cell={metrics.cell_credit:.2e} edge={metrics.edge_credit:.2e}"
            )

        if update % args.eval_every == 0 or update == args.updates:
            last_eval = evaluate_state_ablations(
                trainer.model,
                vocabulary,
                validation_text,
                device=device,
                tokens=args.eval_tokens,
                warmup=args.eval_warmup,
            )
            persistent_bpc = float(
                last_eval["persistent"]["bits_per_character"]
            )
            sample, _ = generate(
                trainer.model,
                vocabulary,
                args.prompt,
                args.generate,
                device=device,
            )
            eval_row = {
                "kind": "evaluation",
                "model": "sol",
                "update": update,
                "ablations": last_eval,
                "sample": sample,
                "device_memory": _device_memory(device),
            }
            _append_jsonl(args.out_dir / "metrics.jsonl", eval_row)
            evaluation_history.append((update, persistent_bpc))
            print(
                f"[sol:{update:6d}] heldout_bpc={persistent_bpc:.3f} "
                f"reset={last_eval['reset_each_token']['bits_per_character']:.3f} "
                f"shuffle={last_eval['shuffled_cells']['bits_per_character']:.3f}"
            )
            if persistent_bpc < best_bpc:
                best_bpc = persistent_bpc
                save_checkpoint(
                    args.out_dir / "best.pt",
                    trainer,
                    {
                        "model": "sol",
                        "best_bpc": best_bpc,
                        "evaluation": last_eval,
                        "sample": sample,
                        "optimization": _optimization_config(
                            args, trainer.learning_rate
                        ),
                    },
                )

        if update % args.checkpoint_every == 0 or update == args.updates:
            save_checkpoint(
                latest,
                trainer,
                {
                    "model": "sol",
                    "best_bpc": best_bpc,
                    "evaluation": last_eval,
                    "optimization": _optimization_config(
                        args, trainer.learning_rate
                    ),
                },
            )

    summary = {
        "model": "sol",
        "parameters": parameter_count,
        "chance_bpc": chance_bpc,
        "updates": trainer.updates,
        "best_bpc": best_bpc,
        "evaluation": last_eval,
        "config": asdict(trainer.model.cfg),
        "optimization": _optimization_config(args, trainer.learning_rate),
        "device_memory": _device_memory(device),
        "frozen_parameters": list(trainer.frozen_parameters),
        "topology": analyze_topology(
            trainer.model.sources,
            trainer.model.sensory_indices,
            trainer.model.output_indices,
            trainer.model.active_edges,
        ).to_dict(),
        "structure": structural_summary(
            trainer.model,
            trainer.structural_config,
            trainer.structural_probation,
        ),
        "routing_traffic": routing_traffic_summary(
            trainer.routing_traffic_config,
            trainer.routing_traffic_trial,
        ),
        "stability": summarize_stability(
            evaluation_history, args.max_final_regression_bpc
        ),
        "convergence": summarize_convergence(
            evaluation_history,
            window=args.convergence_window,
            max_terminal_slope_bpc_per_100_updates=(
                args.max_terminal_slope_bpc_per_100
            ),
        ),
        "exploratory_survival": summarize_exploratory_survival(
            evaluation_history,
            trainer.structural_probation.trial_history,
            args.max_final_regression_bpc,
        ),
    }
    _write_json(args.out_dir / "summary.json", summary)
    return summary


def _run_gru(
    args: argparse.Namespace,
    train_text: str,
    validation_text: str,
    vocabulary: CharacterVocabulary,
) -> dict[str, Any]:
    device = torch.device(args.device)
    sol = SparseAxonField(_sol_config(args, len(vocabulary)))
    target_parameters = sum(parameter.numel() for parameter in sol.parameters())
    hidden_size = match_gru_hidden_size(len(vocabulary), target_parameters)
    del sol
    model = CharacterGRU(len(vocabulary), hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    stream = ContinuousCharStream(train_text, vocabulary, args.batch, device)
    state = model.initial_state(args.batch, device)
    parameter_count = model.parameter_count()
    started = time.monotonic()
    best_bpc = math.inf
    last_eval: dict[str, Any] = {}

    print(
        f"[gru] hidden={hidden_size} params={parameter_count:,} "
        f"target={target_parameters:,} device={device}"
    )
    for update in range(1, args.updates + 1):
        learning_rate = _set_learning_rate_for_update(
            optimizer,
            args,
            update,
        )
        inputs, targets = stream.next(args.chunk)
        optimizer.zero_grad(set_to_none=True)
        logits, state = model(inputs, state.detach())
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        state = state.detach()

        if update == 1 or update % args.log_every == 0:
            elapsed = time.monotonic() - started
            row = {
                "kind": "train",
                "model": "gru",
                "update": update,
                "tokens": update * args.batch * args.chunk,
                "tokens_per_second": (
                    update * args.batch * args.chunk / max(elapsed, 1e-9)
                ),
                "loss": float(loss.item()),
                "bits_per_character": float(loss.item()) / math.log(2.0),
                "learning_rate": learning_rate,
            }
            _append_jsonl(args.out_dir / "metrics.jsonl", row)
            print(
                f"[gru:{update:6d}] loss={loss.item():.4f} "
                f"bpc={loss.item() / math.log(2.0):.3f}"
            )

        if update % args.eval_every == 0 or update == args.updates:
            last_eval = evaluate_gru(
                model,
                vocabulary.encode(validation_text),
                tokens=args.eval_tokens,
                warmup=args.eval_warmup,
            )
            best_bpc = min(
                best_bpc, float(last_eval["bits_per_character"])
            )
            _append_jsonl(
                args.out_dir / "metrics.jsonl",
                {
                    "kind": "evaluation",
                    "model": "gru",
                    "update": update,
                    "metrics": last_eval,
                },
            )
            print(
                f"[gru:{update:6d}] "
                f"heldout_bpc={last_eval['bits_per_character']:.3f}"
            )

    summary = {
        "model": "gru",
        "parameters": parameter_count,
        "target_parameters": target_parameters,
        "hidden_size": hidden_size,
        "updates": args.updates,
        "best_bpc": best_bpc,
        "evaluation": last_eval,
        "optimization": _optimization_config(args),
    }
    _write_json(args.out_dir / "summary.json", summary)
    temporary = args.out_dir / "latest.pt.tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state.cpu(),
            "stream": stream.state_dict(),
            "vocabulary": list(vocabulary.characters),
            "summary": summary,
        },
        temporary,
    )
    os.replace(temporary, args.out_dir / "latest.pt")
    return summary


def _run_transformer(
    args: argparse.Namespace,
    train_text: str,
    validation_text: str,
    vocabulary: CharacterVocabulary,
) -> dict[str, Any]:
    device = torch.device(args.device)
    sol = SparseAxonField(_sol_config(args, len(vocabulary)))
    target_parameters = sum(parameter.numel() for parameter in sol.parameters())
    del sol
    hidden_size = match_transformer_hidden_size(
        len(vocabulary),
        target_parameters,
        layers=args.transformer_layers,
        heads=args.transformer_heads,
        context=args.transformer_context,
    )
    model = CausalCharacterTransformer(
        len(vocabulary),
        hidden_size,
        layers=args.transformer_layers,
        heads=args.transformer_heads,
        context=args.transformer_context,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    stream = ContinuousCharStream(train_text, vocabulary, args.batch, device)
    state = model.initial_state(args.batch, device)
    parameter_count = model.parameter_count()
    started = time.monotonic()
    best_bpc = math.inf
    last_eval: dict[str, Any] = {}

    print(
        f"[transformer] hidden={hidden_size} params={parameter_count:,} "
        f"target={target_parameters:,} context={args.transformer_context} "
        f"device={device}"
    )
    for update in range(1, args.updates + 1):
        learning_rate = _set_learning_rate_for_update(
            optimizer,
            args,
            update,
        )
        inputs, targets = stream.next(args.chunk)
        optimizer.zero_grad(set_to_none=True)
        logits, state = model(inputs, state)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update == 1 or update % args.log_every == 0:
            elapsed = time.monotonic() - started
            row = {
                "kind": "train",
                "model": "transformer",
                "update": update,
                "tokens": update * args.batch * args.chunk,
                "tokens_per_second": (
                    update * args.batch * args.chunk / max(elapsed, 1e-9)
                ),
                "loss": float(loss.item()),
                "bits_per_character": float(loss.item()) / math.log(2.0),
                "learning_rate": learning_rate,
            }
            _append_jsonl(args.out_dir / "metrics.jsonl", row)
            print(
                f"[transformer:{update:6d}] loss={loss.item():.4f} "
                f"bpc={loss.item() / math.log(2.0):.3f}"
            )

        if update % args.eval_every == 0 or update == args.updates:
            last_eval = evaluate_transformer(
                model,
                vocabulary.encode(validation_text),
                tokens=args.eval_tokens,
                warmup=args.eval_warmup,
            )
            best_bpc = min(
                best_bpc, float(last_eval["bits_per_character"])
            )
            _append_jsonl(
                args.out_dir / "metrics.jsonl",
                {
                    "kind": "evaluation",
                    "model": "transformer",
                    "update": update,
                    "metrics": last_eval,
                },
            )
            print(
                f"[transformer:{update:6d}] "
                f"heldout_bpc={last_eval['bits_per_character']:.3f}"
            )

    summary = {
        "model": "transformer",
        "parameters": parameter_count,
        "target_parameters": target_parameters,
        "hidden_size": hidden_size,
        "layers": args.transformer_layers,
        "heads": args.transformer_heads,
        "context": args.transformer_context,
        "updates": args.updates,
        "best_bpc": best_bpc,
        "evaluation": last_eval,
        "optimization": _optimization_config(args),
    }
    _write_json(args.out_dir / "summary.json", summary)
    temporary = args.out_dir / "latest.pt.tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state.cpu(),
            "stream": stream.state_dict(),
            "vocabulary": list(vocabulary.characters),
            "summary": summary,
        },
        temporary,
    )
    os.replace(temporary, args.out_dir / "latest.pt")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", choices=("sol", "gru", "transformer"), default="sol"
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--learning-rate-decay-start",
        type=int,
        default=0,
        help="hold the base rate through this update; zero end disables decay",
    )
    parser.add_argument(
        "--learning-rate-decay-end",
        type=int,
        default=0,
        help="reach the minimum learning rate at this update",
    )
    parser.add_argument(
        "--minimum-learning-rate-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dendrites", type=int, default=8)
    parser.add_argument("--initial-active-dendrites", type=int, default=None)
    parser.add_argument("--sensory-cells", type=int, default=8)
    parser.add_argument("--output-cells", type=int, default=8)
    parser.add_argument("--message-steps", type=int, default=3)
    parser.add_argument("--cell-reward-gain", type=float, default=0.25)
    parser.add_argument("--backward-credit-gain", type=float, default=0.0)
    parser.add_argument(
        "--backward-credit-decay",
        type=float,
        default=0.80,
    )
    parser.add_argument("--output-error-credit-gain", type=float, default=0.0)
    parser.add_argument(
        "--output-error-credit-decay",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--eligibility-routed-output-credit",
        action="store_true",
        help=(
            "route decoder-shaped reverse credit toward installed sources "
            "whose event eligibility aligns with that correction"
        ),
    )
    parser.add_argument(
        "--eligibility-routing-gain",
        type=float,
        default=1.0,
        help=(
            "scale event-memory alignment before normalizing decoder-credit "
            "routing; 1 reproduces the original routed mechanism"
        ),
    )
    parser.add_argument(
        "--reward-plastic-output-credit-routing",
        action="store_true",
        help=(
            "let delayed reward reinforce remembered branch-specific "
            "decoder-credit routing events"
        ),
    )
    parser.add_argument(
        "--exploratory-output-credit-routing",
        action="store_true",
        help=(
            "causally test bounded reverse-credit preference proposals "
            "with live candidate/incumbent traffic"
        ),
    )
    parser.add_argument(
        "--credit-routing-preference-decay",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--credit-routing-plasticity-gain",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--credit-routing-preference-limit",
        type=float,
        default=0.25,
    )
    parser.add_argument("--routing-traffic-interval", type=int, default=25)
    parser.add_argument("--routing-traffic-warmup", type=int, default=75)
    parser.add_argument("--routing-traffic-updates", type=int, default=20)
    parser.add_argument(
        "--routing-traffic-randomization-alpha",
        type=float,
        default=0.10,
    )
    parser.add_argument("--routing-traffic-margin", type=float, default=0.0)
    parser.add_argument("--routing-traffic-step", type=float, default=0.05)
    parser.add_argument(
        "--routing-traffic-minimum-eligibility",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--fast-plasticity-gain", type=float, default=0.04)
    parser.add_argument("--reward-baseline-decay", type=float, default=0.99)
    parser.add_argument("--energy-transport-rate", type=float, default=0.50)
    parser.add_argument("--energy-maintenance-flow", type=float, default=0.0)
    parser.add_argument("--external-energy-gain", type=float, default=0.05)
    parser.add_argument("--quiescence-energy", type=float, default=0.01)
    parser.add_argument("--full-activity-energy", type=float, default=0.05)
    parser.add_argument("--structural-probe-gain", type=float, default=0.03)
    parser.add_argument("--structural-interval", type=int, default=100)
    parser.add_argument("--structural-warmup", type=int, default=500)
    parser.add_argument("--structural-replacements", type=int, default=1)
    parser.add_argument(
        "--structural-confirmation-phases", type=int, default=1
    )
    parser.add_argument(
        "--structural-global-fitness", action="store_true"
    )
    parser.add_argument(
        "--structural-global-fitness-margin", type=float, default=0.0
    )
    parser.add_argument("--structural-credit-decay", type=float, default=0.99)
    parser.add_argument("--structural-credit-margin", type=float, default=1e-3)
    parser.add_argument("--structural-min-edge-age", type=int, default=250)
    parser.add_argument("--structural-growth-cost", type=float, default=0.01)
    parser.add_argument("--structural-min-energy", type=float, default=0.05)
    parser.add_argument(
        "--structural-probation-updates", type=int, default=0
    )
    parser.add_argument(
        "--structural-probation-margin", type=float, default=0.0
    )
    parser.add_argument(
        "--structural-probation-baseline-decay",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--structural-probation-exploratory-traffic",
        action="store_true",
        help=(
            "alternate one candidate probe on/off inside the same "
            "continuously learning organism before grafting"
        ),
    )
    parser.add_argument(
        "--structural-variable-fan-in",
        action="store_true",
        help=(
            "allow active dendrite count to change within fixed slot capacity"
        ),
    )
    parser.add_argument(
        "--structural-min-active-dendrites",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--structural-prune-usage-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--structural-prune-credit-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument("--structural-usage-gain", type=float, default=1.0)
    parser.add_argument(
        "--structural-vector-credit-gain",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--structural-locality-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-final-regression-bpc", type=float, default=0.5)
    parser.add_argument("--convergence-window", type=int, default=5)
    parser.add_argument(
        "--max-terminal-slope-bpc-per-100",
        type=float,
        default=0.01,
        help=(
            "practical-equivalence bound for the terminal validation slope"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-tokens", type=int, default=2048)
    parser.add_argument("--eval-warmup", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-context", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--freeze-edges",
        action="store_true",
        help="Train the cell rule while keeping edge weights and biases fixed",
    )
    parser.add_argument(
        "--no-metabolism",
        action="store_true",
        help="Hold energy at one to isolate prediction capability from economy",
    )
    parser.add_argument(
        "--no-reward",
        action="store_true",
        help="Disable cell and synapse reward-modulated eligibility feedback",
    )
    parser.add_argument(
        "--no-fast-plasticity",
        action="store_true",
        help="Disable fast synaptic efficacy while keeping cell reward feedback",
    )
    parser.add_argument(
        "--structural-plasticity",
        action="store_true",
        help="Probe, grow, and prune directed dendrites between BPTT windows",
    )
    parser.add_argument(
        "--structural-probes-only",
        action="store_true",
        help="Run causal candidate probes without changing fixed topology",
    )
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--generate", type=int, default=240)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if (
        args.learning_rate_decay_start < 0
        or args.learning_rate_decay_end < 0
    ):
        parser.error("learning-rate decay boundaries must be non-negative")
    if (
        args.learning_rate_decay_end != 0
        and args.learning_rate_decay_end
        <= args.learning_rate_decay_start
    ):
        parser.error(
            "--learning-rate-decay-end must exceed "
            "--learning-rate-decay-start"
        )
    if not 0 <= args.minimum_learning_rate_ratio <= 1:
        parser.error("--minimum-learning-rate-ratio must be in [0, 1]")
    if args.fast_plasticity_gain < 0:
        parser.error("--fast-plasticity-gain must be non-negative")
    if args.cell_reward_gain < 0:
        parser.error("--cell-reward-gain must be non-negative")
    if args.backward_credit_gain < 0:
        parser.error("--backward-credit-gain must be non-negative")
    if not 0 <= args.backward_credit_decay < 1:
        parser.error("--backward-credit-decay must be in [0, 1)")
    if args.output_error_credit_gain < 0:
        parser.error("--output-error-credit-gain must be non-negative")
    if not 0 <= args.output_error_credit_decay < 1:
        parser.error("--output-error-credit-decay must be in [0, 1)")
    if args.eligibility_routing_gain < 0:
        parser.error("--eligibility-routing-gain must be non-negative")
    if not 0 <= args.credit_routing_preference_decay < 1:
        parser.error(
            "--credit-routing-preference-decay must be in [0, 1)"
        )
    if args.credit_routing_plasticity_gain < 0:
        parser.error(
            "--credit-routing-plasticity-gain must be non-negative"
        )
    if args.credit_routing_preference_limit <= 0:
        parser.error(
            "--credit-routing-preference-limit must be positive"
        )
    if sum(
        (
            args.eligibility_routed_output_credit,
            args.reward_plastic_output_credit_routing,
            args.exploratory_output_credit_routing,
        )
    ) > 1:
        parser.error(
            "output-credit routing policies are mutually exclusive"
        )
    if args.exploratory_output_credit_routing:
        if args.output_error_credit_gain <= 0 or args.no_reward:
            parser.error(
                "--exploratory-output-credit-routing requires "
                "--output-error-credit-gain > 0 and reward"
            )
        if args.routing_traffic_interval < 1:
            parser.error("--routing-traffic-interval must be positive")
        if args.routing_traffic_warmup < 0:
            parser.error(
                "--routing-traffic-warmup must be non-negative"
            )
        if (
            args.routing_traffic_updates < 4
            or args.routing_traffic_updates % 4 != 0
            or args.routing_traffic_updates > 64
        ):
            parser.error(
                "--routing-traffic-updates must be 4 to 64 "
                "and divisible by 4"
            )
        if not 0 < args.routing_traffic_randomization_alpha <= 1:
            parser.error(
                "--routing-traffic-randomization-alpha must be in (0, 1]"
            )
        if args.routing_traffic_margin < 0:
            parser.error(
                "--routing-traffic-margin must be non-negative"
            )
        if (
            args.routing_traffic_step <= 0
            or args.routing_traffic_step
            > args.credit_routing_preference_limit
        ):
            parser.error(
                "--routing-traffic-step must be positive and no greater "
                "than --credit-routing-preference-limit"
            )
        if args.routing_traffic_minimum_eligibility < 0:
            parser.error(
                "--routing-traffic-minimum-eligibility must be non-negative"
            )
    if args.structural_probe_gain <= 0 and (
        args.structural_plasticity or args.structural_probes_only
    ):
        parser.error("structural probes require --structural-probe-gain > 0")
    if args.structural_plasticity and args.structural_probes_only:
        parser.error(
            "--structural-plasticity and --structural-probes-only are exclusive"
        )
    if args.freeze_edges and args.structural_plasticity:
        parser.error("--freeze-edges cannot be combined with rewiring")
    if not 0 <= args.reward_baseline_decay < 1:
        parser.error("--reward-baseline-decay must be in [0, 1)")
    if not 0 <= args.energy_transport_rate <= 1:
        parser.error("--energy-transport-rate must be in [0, 1]")
    if not 0 <= args.energy_maintenance_flow <= 1:
        parser.error("--energy-maintenance-flow must be in [0, 1]")
    if args.external_energy_gain < 0:
        parser.error("--external-energy-gain must be non-negative")
    if not (
        0 <= args.quiescence_energy < args.full_activity_energy <= 1
    ):
        parser.error(
            "energy thresholds must satisfy 0 <= quiescence < full activity <= 1"
        )
    if args.max_final_regression_bpc < 0:
        parser.error("--max-final-regression-bpc must be non-negative")
    if args.convergence_window < 3:
        parser.error("--convergence-window must be at least 3")
    if args.max_terminal_slope_bpc_per_100 < 0:
        parser.error(
            "--max-terminal-slope-bpc-per-100 must be non-negative"
        )
    if args.structural_min_active_dendrites < 1:
        parser.error("--structural-min-active-dendrites must be positive")
    if (
        args.structural_min_active_dendrites
        > args.dendrites
    ):
        parser.error(
            "--structural-min-active-dendrites cannot exceed --dendrites"
        )
    if args.structural_prune_usage_threshold < 0:
        parser.error(
            "--structural-prune-usage-threshold must be non-negative"
        )
    if (
        args.structural_usage_gain < 0
        or args.structural_vector_credit_gain < 0
        or args.structural_locality_gain < 0
    ):
        parser.error("structural evidence gains must be non-negative")
    if not 0 <= args.structural_credit_decay < 1:
        parser.error("--structural-credit-decay must be in [0, 1)")
    if args.structural_credit_margin < 0:
        parser.error("--structural-credit-margin must be non-negative")
    if args.structural_global_fitness_margin < 0:
        parser.error(
            "--structural-global-fitness-margin must be non-negative"
        )
    if args.structural_growth_cost < 0:
        parser.error("--structural-growth-cost must be non-negative")
    if not 0 <= args.structural_min_energy <= 1:
        parser.error("--structural-min-energy must be in [0, 1]")
    for name in ("log_every", "eval_every", "checkpoint_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "structural_interval",
        "structural_replacements",
        "structural_confirmation_phases",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("structural_warmup", "structural_min_edge_age"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.structural_probation_updates < 0:
        parser.error("--structural-probation-updates must be non-negative")
    if (
        args.structural_probation_exploratory_traffic
        and (
            args.structural_probation_updates < 2
            or args.structural_probation_updates % 2 != 0
        )
    ):
        parser.error(
            "--structural-probation-exploratory-traffic requires "
            "an even --structural-probation-updates >= 2"
        )
    if not 0 <= args.structural_probation_baseline_decay < 1:
        parser.error(
            "--structural-probation-baseline-decay must be in [0, 1)"
        )
    if args.chunk > args.transformer_context:
        parser.error("--chunk must not exceed --transformer-context")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_text, validation_text = _split_corpus(args.file)
    vocabulary = CharacterVocabulary.from_text(train_text + validation_text)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_characters": len(train_text),
        "validation_characters": len(validation_text),
        "vocabulary": list(vocabulary.characters),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    if args.model == "sol":
        _run_sol(args, train_text, validation_text, vocabulary)
    elif args.model == "gru":
        _run_gru(args, train_text, validation_text, vocabulary)
    else:
        _run_transformer(args, train_text, validation_text, vocabulary)


if __name__ == "__main__":
    main()
