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
from .evaluate import evaluate_state_ablations
from .model import SolConfig, SparseAxonField
from .stream import CharacterVocabulary, ContinuousCharStream
from .topology import analyze_topology
from .train import ContinuousTrainer, generate

DEFAULT_CORPUS = Path("data/tinyshakespeare/input.txt")


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


def _sol_config(args: argparse.Namespace, vocab_size: int) -> SolConfig:
    metabolic = {} if not args.no_metabolism else {
        "energy_start": 1.0,
        "basal_cost": 0.0,
        "activity_cost": 0.0,
        "stimulation_gain": 0.0,
    }
    reward: dict[str, float] = {
        "fast_plasticity_gain": args.fast_plasticity_gain
    }
    if args.no_reward:
        reward = {"reward_gain": 0.0, "fast_plasticity_gain": 0.0}
    elif args.no_fast_plasticity:
        reward = {"fast_plasticity_gain": 0.0}
    return SolConfig(
        vocab_size=vocab_size,
        cells=args.cells,
        channels=args.channels,
        dendrites=args.dendrites,
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
            device=device,
        )

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

    while trainer.updates < args.updates:
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
                **asdict(metrics),
            }
            _append_jsonl(args.out_dir / "metrics.jsonl", row)
            print(
                f"[sol:{update:6d}] loss={metrics.loss:.4f} "
                f"bpc={metrics.loss / math.log(2.0):.3f} "
                f"energy={metrics.mean_energy:.3f} "
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
            }
            _append_jsonl(args.out_dir / "metrics.jsonl", eval_row)
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
        "frozen_parameters": list(trainer.frozen_parameters),
        "topology": analyze_topology(
            trainer.model.sources,
            trainer.model.sensory_indices,
            trainer.model.output_indices,
        ).to_dict(),
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
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dendrites", type=int, default=8)
    parser.add_argument("--sensory-cells", type=int, default=8)
    parser.add_argument("--output-cells", type=int, default=8)
    parser.add_argument("--message-steps", type=int, default=3)
    parser.add_argument("--fast-plasticity-gain", type=float, default=0.02)
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
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--generate", type=int, default=240)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    if args.fast_plasticity_gain < 0:
        parser.error("--fast-plasticity-gain must be non-negative")
    for name in ("log_every", "eval_every", "checkpoint_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
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
