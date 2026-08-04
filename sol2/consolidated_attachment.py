"""Resumable S1-P3 utility-consolidation branch runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoint import pack_state, unpack_state
from .consolidation import (
    ConsolidationPolicy,
    UtilityProfile,
    calibrate_causal_utility,
    make_utility_profile,
    profile_summary,
)
from .growth import activate_reserve_dendrites
from .interventions import disable_graft_edges, zero_private_adapters
from .model import Sol2, count_parameters
from .optim import add_optimizer_parameters, guarded_step, set_learning_rate
from .procedural_acquisition import neuron_telemetry
from .procedural_benchmark import (
    PhaseResult,
    _detach_state,
    _step_episode,
    evaluate_regime,
    evaluate_with_ablations,
)
from .procedural_task import ProceduralTask
from .organ_attachment import _length_curve, _load_acquisition

SCHEMA = 1
BRANCHES = ("plastic", "uniform", "consolidated", "shuffled")
LN2 = math.log(2.0)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _parameter_digest(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _a_organ_named_parameters(model: Sol2):
    for name, parameter in model.named_parameters():
        if name.startswith(("embedding.", "sensory_gain", "sensory_bias", "output_organ.")):
            yield name, parameter


def _substrate_named_parameters(model: Sol2):
    for name, parameter in model.named_parameters():
        if not name.startswith(
            ("embedding.", "sensory_gain", "sensory_bias", "output_organ.", "attached_organs.")
        ):
            yield name, parameter


def _train_phase(
    model: Sol2,
    optimizer,
    state,
    task,
    regime,
    cfg,
    policy: ConsolidationPolicy,
    *,
    updates: int,
    global_update_start: int,
    max_steps: int,
    generator_seed: int,
) -> PhaseResult:
    generator = torch.Generator().manual_seed(generator_seed)
    records = []
    accepted = 0
    rejected = 0
    model.train()
    for local_update in range(1, updates + 1):
        global_update = global_update_start + local_update
        set_learning_rate(optimizer, cfg, global_update)
        steps = int(torch.randint(1, max_steps + 1, (1,), generator=generator).item())
        batch = task.sample_batch(regime, cfg.batch_size, steps, generator, cfg.device)
        logits, next_state = _step_episode(
            model, batch.tokens, state, organ_name="B"
        )
        loss = F.cross_entropy(logits, batch.answer_tokens)
        next_state = _detach_state(next_state)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = policy.capture_before_step()
        health = guarded_step(model, optimizer, cfg)
        if health.accepted:
            policy.apply_after_step(before)
            accepted += 1
        else:
            rejected += 1
        state = next_state
        records.append(
            {
                "update": local_update,
                "global_update": global_update,
                "steps": steps,
                "answer_bits": float(loss.detach()) / LN2,
                "answer_accuracy": float(
                    logits.detach().argmax(-1).eq(batch.answer_tokens).float().mean()
                ),
                "gradient_norm": health.total_norm,
                "accepted": health.accepted,
            }
        )
        if local_update % cfg.log_every == 0 or local_update == updates:
            tail = records[-min(cfg.log_every, len(records)) :]
            accuracy = sum(row["answer_accuracy"] for row in tail) / len(tail)
            bits = sum(row["answer_bits"] for row in tail) / len(tail)
            print(
                f"[{regime.name}] u{local_update} bits={bits:.3f} "
                f"acc={accuracy:.3f} rejects={rejected}",
                flush=True,
            )
    return PhaseResult(records, state, accepted, rejected)


def _quartile(values: torch.Tensor, utility: torch.Tensor) -> dict:
    if values.numel() != utility.numel() or values.numel() < 1:
        raise ValueError("drift values and utility must be non-empty and aligned")
    count = max(1, values.numel() // 4)
    order = torch.argsort(utility, stable=True)
    low_values = values[order[:count]]
    high_values = values[order[-count:]]
    return {
        "low_count": count,
        "high_count": count,
        "low_mean": float(low_values.mean()),
        "high_mean": float(high_values.mean()),
    }


@torch.no_grad()
def _drift_summary(
    model: Sol2, anchor: dict[str, torch.Tensor], profile: UtilityProfile
) -> dict:
    gain_delta = model.cell_gain.detach() - anchor["cell_gain"]
    bias_delta = model.cell_bias.detach() - anchor["cell_bias"]
    expression_delta = (gain_delta.square() + bias_delta.square()).mean(-1).sqrt()
    expression_utility = profile.measured_cell[model.expression_cells]
    edge_delta = (model.graph.edge_logit.detach() - anchor["graph.edge_logit"]).abs()
    active = model.graph.active
    genome_squared = []
    for name, parameter in model.named_parameters():
        if name.startswith(("graph.query.", "graph.key.", "graph.value.", "tissues.")):
            genome_squared.append((parameter.detach() - anchor[name]).square().flatten())
    result = {
        "expression_rms": float(expression_delta.square().mean().sqrt()),
        "expression_by_measured_utility": _quartile(expression_delta, expression_utility),
        "active_edge_abs_mean": float(edge_delta[active].mean()),
        "edge_by_measured_utility": _quartile(
            edge_delta[active], profile.measured_edge[active]
        ),
        "genome_rms": float(torch.cat(genome_squared).mean().sqrt()),
    }
    if model.cell_adapter_down is not None:
        down_delta = (
            model.cell_adapter_down.detach() - anchor["cell_adapter_down"]
        ).square().mean(dim=(1, 2))
        up_delta = (
            model.cell_adapter_up.detach() - anchor["cell_adapter_up"]
        ).square().mean(dim=(1, 2))
        adapter_delta = (0.5 * (down_delta + up_delta)).sqrt()
        result["adapter_rms"] = float(adapter_delta.square().mean().sqrt())
        result["adapter_by_measured_utility"] = _quartile(
            adapter_delta, expression_utility
        )
    return result


def _evaluate(
    model: Sol2,
    state,
    task,
    regime_a,
    regime_b,
    cfg,
    profile: UtilityProfile,
    anchor,
    *,
    update: int,
    max_steps: int,
    batches: int,
) -> dict:
    a = evaluate_regime(
        model,
        state,
        task,
        regime_a,
        cfg,
        steps=max_steps,
        batches=batches,
        generator_seed=cfg.seed + 40_000 + update,
        organ_name="A",
    )
    b = _length_curve(
        model,
        state,
        task,
        regime_b,
        cfg,
        organ_name="B",
        max_steps=max_steps,
        batches=batches,
        generator_seed=cfg.seed + 50_000 + update,
    )
    return {
        "update": update,
        "a_fixed": a,
        "b_by_length": b,
        "min_ab_accuracy": min(a["answer_accuracy"], b[str(max_steps)]["answer_accuracy"]),
        "drift": _drift_summary(model, anchor, profile),
        "telemetry": neuron_telemetry(model, state),
    }


def _utility_lesions(
    model: Sol2,
    state,
    task,
    regime_a,
    regime_b,
    cfg,
    profile: UtilityProfile,
    *,
    max_steps: int,
    batches: int,
) -> dict:
    internal = model.internal_idx
    order = torch.argsort(profile.measured_cell[internal])
    count = max(1, len(internal) // 4)
    low = internal[order[:count]]
    high = internal[order[-count:]]
    result = {"cells_per_lesion": count}
    for organ, regime, seed in (("A", regime_a, 80_000), ("B", regime_b, 90_000)):
        common = dict(
            model=model,
            state=state,
            task=task,
            regime=regime,
            cfg=cfg,
            steps=max_steps,
            batches=batches,
            generator_seed=cfg.seed + seed,
            organ_name=organ,
        )
        result[organ] = {
            "normal": evaluate_regime(**common),
            "freeze_high_utility": evaluate_regime(**common, frozen_idx=high),
            "freeze_low_utility": evaluate_regime(**common, frozen_idx=low),
        }
    return result


def _reserve_lesions(
    model: Sol2,
    state,
    task,
    regime_a,
    regime_b,
    cfg,
    profile: UtilityProfile,
    grafts: list[dict],
    *,
    max_steps: int,
    batches: int,
) -> dict:
    """Measure whether new capability resides in private reserve computation."""

    internal = model.internal_idx
    order = torch.argsort(profile.measured_cell[internal], stable=True)
    count = max(1, len(internal) // 4)
    low = internal[order[:count]]
    result = {"low_utility_cells": count, "graft_edges": len(grafts)}
    for organ, regime, seed in (("A", regime_a, 110_000), ("B", regime_b, 120_000)):
        common = dict(
            model=model,
            state=state,
            task=task,
            regime=regime,
            cfg=cfg,
            steps=max_steps,
            batches=batches,
            generator_seed=cfg.seed + seed,
            organ_name=organ,
        )
        organ_result = {"normal": evaluate_regime(**common)}
        if model.cell_adapter_up is not None:
            with zero_private_adapters(model, low):
                organ_result["zero_low_utility_adapters"] = evaluate_regime(**common)
            with zero_private_adapters(model, internal):
                organ_result["zero_all_internal_adapters"] = evaluate_regime(**common)
        if grafts:
            with disable_graft_edges(model, grafts):
                organ_result["disable_graft_edges"] = evaluate_regime(**common)
        result[organ] = organ_result
    return result


def _save_checkpoint(
    path: Path,
    *,
    branch: str,
    model: Sol2,
    optimizer,
    state,
    cfg,
    update: int,
    accepted_updates: int,
    rejected_updates: int,
    records: list[dict],
    evaluations: list[dict],
    profile: UtilityProfile,
    grafts: list[dict],
) -> None:
    payload = {
        "schema": SCHEMA,
        "branch": branch,
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": pack_state(state),
        "update": update,
        "accepted_updates": accepted_updates,
        "rejected_updates": rejected_updates,
        "records": records,
        "evaluations": evaluations,
        "utility_profile": profile.cpu_payload(),
        "grafts": grafts,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_consolidated_attachment_branch(
    acquisition_path: Path,
    out_dir: Path,
    branch: str,
    *,
    device: str,
    adaptation_updates: int,
    eval_every: int,
    eval_batches: int,
    final_eval_batches: int,
    utility_batches: int,
    max_steps: int,
    threshold: float = 0.65,
    temperature: float = 0.10,
    minimum_plasticity: float = 0.02,
    genome_plasticity: float = 0.05,
    directional_edges: bool = False,
    reserve_growth: bool = False,
    reserve_target_fraction: float = 0.5,
    reserve_slots_per_target: int = 2,
    reserve_edge_raw: float = -1.5,
    max_graft_a_drop: float | None = None,
    resume: bool = False,
) -> dict:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    if min(adaptation_updates, eval_every, eval_batches, final_eval_batches, utility_batches, max_steps) < 1:
        raise ValueError("all experiment budgets must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "adaptation.pt"
    metrics_path = out_dir / "metrics.json"
    acquisition, cfg, model, optimizer, state = _load_acquisition(acquisition_path, device)
    task = ProceduralTask()
    regime_a = task.base_regime("A-acquisition", cfg.seed + 101)
    regime_b = task.remapped_interface(regime_a, "B-attached-organ", cfg.seed + 202)
    measured_cell, measured_edge, calibration = calibrate_causal_utility(
        model,
        state,
        task,
        regime_a,
        cfg,
        batches=utility_batches,
        steps=max_steps,
        generator_seed=cfg.seed + 30_000,
        directional_edges=directional_edges,
    )
    profile = make_utility_profile(
        model,
        measured_cell,
        measured_edge,
        branch=branch,
        threshold=threshold,
        temperature=temperature,
        minimum_plasticity=minimum_plasticity,
        genome_plasticity=genome_plasticity,
        shuffle_seed=cfg.seed + 31_000,
    )
    pre_graft_a = evaluate_regime(
        model,
        state,
        task,
        regime_a,
        cfg,
        steps=max_steps,
        batches=eval_batches,
        generator_seed=cfg.seed + 32_000,
        organ_name="A",
    )
    grafts = (
        activate_reserve_dendrites(
            model,
            profile.applied_cell,
            target_fraction=reserve_target_fraction,
            slots_per_target=reserve_slots_per_target,
            weak_raw=reserve_edge_raw,
        )
        if reserve_growth
        else []
    )
    post_graft_a = evaluate_regime(
        model,
        state,
        task,
        regime_a,
        cfg,
        steps=max_steps,
        batches=eval_batches,
        generator_seed=cfg.seed + 32_000,
        organ_name="A",
    )
    graft_a_drop = pre_graft_a["answer_accuracy"] - post_graft_a["answer_accuracy"]
    if max_graft_a_drop is not None and graft_a_drop > max_graft_a_drop:
        raise RuntimeError(
            f"reserve graft A drop {graft_a_drop:.4f} exceeds {max_graft_a_drop:.4f}"
        )
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in _substrate_named_parameters(model)
    }
    a_digest_before = _parameter_digest(_a_organ_named_parameters(model))
    organ_seed = 100_000 + cfg.seed
    organ = model.attach_organ("B", seed=organ_seed)
    add_optimizer_parameters(
        optimizer, organ.named_parameters(prefix="attached_organs.B"), cfg
    )
    cfg = cfg.scaled(updates=int(acquisition["update"]) + adaptation_updates)
    policy = ConsolidationPolicy(model, profile)
    update = 0
    accepted_updates = 0
    rejected_updates = 0
    records: list[dict] = []
    evaluations: list[dict] = []
    baseline = _evaluate(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        profile,
        anchor,
        update=0,
        max_steps=max_steps,
        batches=eval_batches,
    )

    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if payload.get("schema") != SCHEMA or payload.get("branch") != branch:
            raise ValueError("incompatible consolidated-attachment checkpoint")
        if not torch.equal(
            payload["utility_profile"]["measured_cell"], measured_cell.detach().cpu()
        ):
            raise ValueError("utility calibration changed across resume")
        if payload.get("grafts", []) != grafts:
            raise ValueError("reserve graft ledger changed across resume")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        state = unpack_state(payload["state"], device)
        update = int(payload["update"])
        accepted_updates = int(payload["accepted_updates"])
        rejected_updates = int(payload["rejected_updates"])
        records = list(payload["records"])
        evaluations = list(payload["evaluations"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_state = payload.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    while update < adaptation_updates:
        interval = min(eval_every, adaptation_updates - update)
        phase = _train_phase(
            model,
            optimizer,
            state,
            task,
            regime_b,
            cfg,
            policy,
            updates=interval,
            global_update_start=int(acquisition["update"]) + update,
            max_steps=max_steps,
            generator_seed=cfg.seed + 20_000 + update,
        )
        state = phase.state
        update += interval
        accepted_updates += phase.accepted_updates
        rejected_updates += phase.rejected_updates
        records.extend(phase.records)
        evaluation = _evaluate(
            model,
            state,
            task,
            regime_a,
            regime_b,
            cfg,
            profile,
            anchor,
            update=update,
            max_steps=max_steps,
            batches=eval_batches,
        )
        evaluations.append(evaluation)
        print(
            f"[{branch}] u{update} A4={evaluation['a_fixed']['answer_accuracy']:.3f} "
            f"B4={evaluation['b_by_length'][str(max_steps)]['answer_accuracy']:.3f} "
            f"min={evaluation['min_ab_accuracy']:.3f}",
            flush=True,
        )
        _save_checkpoint(
            checkpoint_path,
            branch=branch,
            model=model,
            optimizer=optimizer,
            state=state,
            cfg=cfg,
            update=update,
            accepted_updates=accepted_updates,
            rejected_updates=rejected_updates,
            records=records,
            evaluations=evaluations,
            profile=profile,
            grafts=grafts,
        )
        _atomic_json(
            metrics_path,
            {
                "branch": branch,
                "update": update,
                "calibration": calibration,
                "profile": profile_summary(model, profile),
                "growth": {
                    "grafts": grafts,
                    "pre_graft_a": pre_graft_a,
                    "post_graft_a": post_graft_a,
                    "a_accuracy_drop": graft_a_drop,
                },
                "baseline": baseline,
                "evaluations": evaluations,
                "records": records,
            },
        )

    final = _evaluate(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        profile,
        anchor,
        update=update,
        max_steps=max_steps,
        batches=final_eval_batches,
    )
    b_lesions = evaluate_with_ablations(
        model,
        state,
        task,
        regime_b,
        cfg,
        steps=max_steps,
        batches=final_eval_batches,
        generator_seed=cfg.seed + 60_000,
        organ_name="B",
    )
    utility_lesions = _utility_lesions(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        profile,
        max_steps=max_steps,
        batches=final_eval_batches,
    )
    reserve_lesions = _reserve_lesions(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        profile,
        grafts,
        max_steps=max_steps,
        batches=final_eval_batches,
    )
    final_grafts = []
    for graft in grafts:
        raw = float(model.graph.edge_logit[graft["target"], graft["slot"]].detach())
        final_grafts.append(
            {
                **graft,
                "final_edge_logit": raw,
                "edge_logit_delta": raw - float(graft["initial_edge_logit"]),
                "final_edge_bias": model.cfg.edge_logit_limit * math.tanh(raw),
            }
        )
    result = {
        "branch": branch,
        "source_update": int(acquisition["update"]),
        "update": update,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": count_parameters(model),
        "accepted_updates": accepted_updates,
        "rejected_updates": rejected_updates,
        "config": cfg.to_dict(),
        "protocol": {
            "adaptation_updates": adaptation_updates,
            "eval_every": eval_every,
            "eval_batches": eval_batches,
            "final_eval_batches": final_eval_batches,
            "utility_batches": utility_batches,
            "max_steps": max_steps,
            "threshold": threshold,
            "temperature": temperature,
            "minimum_plasticity": minimum_plasticity,
            "genome_plasticity": genome_plasticity,
            "directional_edges": directional_edges,
            "reserve_growth": reserve_growth,
            "reserve_target_fraction": reserve_target_fraction,
            "reserve_slots_per_target": reserve_slots_per_target,
            "reserve_edge_raw": reserve_edge_raw,
            "max_graft_a_drop": max_graft_a_drop,
            "organ_seed": organ_seed,
        },
        "calibration": calibration,
        "profile": profile_summary(model, profile),
        "utility": {
            "measured_cell": measured_cell.detach().cpu().tolist(),
            "applied_cell": profile.applied_cell.detach().cpu().tolist(),
        },
        "integrity": {
            "a_organ_digest_before": a_digest_before,
            "a_organ_digest_after": _parameter_digest(_a_organ_named_parameters(model)),
            "a_organ_unchanged": a_digest_before == _parameter_digest(_a_organ_named_parameters(model)),
        },
        "growth": {
            "grafts": final_grafts,
            "pre_graft_a": pre_graft_a,
            "post_graft_a": post_graft_a,
            "a_accuracy_drop": graft_a_drop,
            "within_limit": (
                max_graft_a_drop is None or graft_a_drop <= max_graft_a_drop
            ),
        },
        "summary": {
            "baseline": baseline,
            "final": final,
            "b_lesions": b_lesions,
            "utility_lesions": utility_lesions,
            "reserve_lesions": reserve_lesions,
        },
        "records": records,
        "evaluations": evaluations,
    }
    _atomic_json(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SOL2 consolidated organ attachment")
    parser.add_argument("--acquisition-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adaptation-updates", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--final-eval-batches", type=int, default=64)
    parser.add_argument("--utility-batches", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--minimum-plasticity", type=float, default=0.02)
    parser.add_argument("--genome-plasticity", type=float, default=0.05)
    parser.add_argument("--directional-edges", action="store_true")
    parser.add_argument("--reserve-growth", action="store_true")
    parser.add_argument("--reserve-target-fraction", type=float, default=0.5)
    parser.add_argument("--reserve-slots-per-target", type=int, default=2)
    parser.add_argument("--reserve-edge-raw", type=float, default=-1.5)
    parser.add_argument("--max-graft-a-drop", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_consolidated_attachment_branch(
        args.acquisition_checkpoint,
        args.out_dir,
        args.branch,
        device=args.device,
        adaptation_updates=args.adaptation_updates,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        final_eval_batches=args.final_eval_batches,
        utility_batches=args.utility_batches,
        max_steps=args.max_steps,
        threshold=args.threshold,
        temperature=args.temperature,
        minimum_plasticity=args.minimum_plasticity,
        genome_plasticity=args.genome_plasticity,
        directional_edges=args.directional_edges,
        reserve_growth=args.reserve_growth,
        reserve_target_fraction=args.reserve_target_fraction,
        reserve_slots_per_target=args.reserve_slots_per_target,
        reserve_edge_raw=args.reserve_edge_raw,
        max_graft_a_drop=args.max_graft_a_drop,
        resume=args.resume,
    )
    final = result["summary"]["final"]
    print(
        json.dumps(
            {
                "branch": result["branch"],
                "a_fixed": final["a_fixed"],
                "b_by_length": final["b_by_length"],
                "min_ab_accuracy": final["min_ab_accuracy"],
                "drift": final["drift"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
