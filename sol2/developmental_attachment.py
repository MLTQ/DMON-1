"""Resumable S1-P4 anchored attachment and endogenous development runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .anchored_consolidation import (
    AnchorProfile,
    ProximalAnchorPolicy,
    anchor_profile_summary,
    make_anchor_profile,
)
from .checkpoint import (
    pack_state,
    restore_rng_state,
    same_tensor_values,
    unpack_state,
)
from .consolidation import calibrate_causal_utility
from .development import DevelopmentController, decision_payload
from .growth import grow_relay_tissue
from .interventions import zero_private_adapters
from .model import Sol2, count_parameters
from .optim import add_optimizer_parameters, guarded_step, set_learning_rate
from .organ_attachment import _length_curve, _load_acquisition
from .procedural_acquisition import neuron_telemetry
from .procedural_benchmark import (
    _detach_state,
    _step_episode,
    evaluate_regime,
    evaluate_with_ablations,
)
from .procedural_task import ProceduralTask

SCHEMA = 1
BRANCHES = ("plastic", "uniform_anchor", "measured_anchor", "developmental")
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


def _quartile(values: torch.Tensor, utility: torch.Tensor) -> dict:
    count = max(1, values.numel() // 4)
    order = torch.argsort(utility, stable=True)
    return {
        "low_count": count,
        "high_count": count,
        "low_mean": float(values[order[:count]].mean()),
        "high_mean": float(values[order[-count:]].mean()),
    }


@torch.no_grad()
def _drift_summary(
    model: Sol2, policy: ProximalAnchorPolicy, profile: AnchorProfile
) -> dict:
    anchors = policy.anchors
    rows = policy.base_expression_rows
    gain_delta = model.cell_gain[:rows].detach() - anchors["cell_gain"]
    bias_delta = model.cell_bias[:rows].detach() - anchors["cell_bias"]
    expression_delta = (gain_delta.square() + bias_delta.square()).mean(-1).sqrt()
    expression_utility = profile.measured_cell[
        model.expression_cells[:rows]
    ]
    edge_delta = (
        model.graph.edge_logit[: policy.base_cells].detach()
        - anchors["graph.edge_logit"]
    ).abs()
    active = model.graph.active[: policy.base_cells]
    genome_squared = []
    for name, anchor in anchors.items():
        if name.startswith(("graph.query.", "graph.key.", "graph.value.", "tissues.")):
            genome_squared.append((dict(model.named_parameters())[name] - anchor).square().flatten())
    result = {
        "expression_rms": float(expression_delta.square().mean().sqrt()),
        "expression_by_measured_utility": _quartile(
            expression_delta, expression_utility
        ),
        "installed_edge_abs_mean": float(edge_delta[active].mean()),
        "edge_by_measured_utility": _quartile(
            edge_delta[active], profile.measured_edge[active]
        ),
        "genome_rms": float(torch.cat(genome_squared).mean().sqrt()),
        "anchor_energy": policy.anchor_energy(),
    }
    if model.cell_adapter_down is not None:
        down = (
            model.cell_adapter_down[:rows].detach() - anchors["cell_adapter_down"]
        ).square().mean(dim=(1, 2))
        up = (
            model.cell_adapter_up[:rows].detach() - anchors["cell_adapter_up"]
        ).square().mean(dim=(1, 2))
        adapter_delta = (0.5 * (down + up)).sqrt()
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
    policy: ProximalAnchorPolicy,
    profile: AnchorProfile,
    *,
    update: int,
    max_steps: int,
    batches: int,
    pressure: dict,
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
        "pressure": pressure,
        "drift": _drift_summary(model, policy, profile),
        "telemetry": neuron_telemetry(model, state),
    }


def _train_interval(
    model: Sol2,
    optimizer,
    state,
    task,
    regime_b,
    cfg,
    policy: ProximalAnchorPolicy,
    *,
    updates: int,
    global_update_start: int,
    max_steps: int,
    generator_seed: int,
) -> tuple[object, list[dict], int, int, dict]:
    generator = torch.Generator().manual_seed(generator_seed)
    records = []
    accepted = 0
    rejected = 0
    pressure_sums = {
        "pressure": 0.0,
        "anchored_gradient_squared": 0.0,
        "substrate_gradient_squared": 0.0,
        "accessible_gradient_squared": 0.0,
    }
    model.train()
    for local_update in range(1, updates + 1):
        global_update = global_update_start + local_update
        set_learning_rate(optimizer, cfg, global_update)
        steps = int(torch.randint(1, max_steps + 1, (1,), generator=generator).item())
        batch = task.sample_batch(regime_b, cfg.batch_size, steps, generator, cfg.device)
        logits, next_state = _step_episode(model, batch.tokens, state, organ_name="B")
        loss = F.cross_entropy(logits, batch.answer_tokens)
        next_state = _detach_state(next_state)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pressure = policy.gradient_pressure()
        health = guarded_step(model, optimizer, cfg)
        if health.accepted:
            policy.apply_proximal()
            accepted += 1
        else:
            rejected += 1
        for key in pressure_sums:
            pressure_sums[key] += pressure[key]
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
                "pressure": pressure["pressure"],
                "accepted": health.accepted,
            }
        )
        if local_update % cfg.log_every == 0 or local_update == updates:
            tail = records[-min(cfg.log_every, len(records)) :]
            print(
                f"[{regime_b.name}] u{local_update} "
                f"acc={sum(row['answer_accuracy'] for row in tail) / len(tail):.3f} "
                f"pressure={sum(row['pressure'] for row in tail) / len(tail):.3f} "
                f"rejects={rejected}",
                flush=True,
            )
    pressure_mean = {key: value / updates for key, value in pressure_sums.items()}
    return state, records, accepted, rejected, pressure_mean


def _growth_lesions(
    model: Sol2,
    state,
    task,
    regime_a,
    regime_b,
    cfg,
    grown_cells: list[int],
    *,
    max_steps: int,
    batches: int,
) -> dict | None:
    if not grown_cells:
        return None
    cells = torch.tensor(grown_cells, device=model.mutable_idx.device)
    result = {"cells": grown_cells}
    for organ, regime, seed in (("A", regime_a, 130_000), ("B", regime_b, 140_000)):
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
        normal = evaluate_regime(**common)
        frozen = evaluate_regime(**common, frozen_idx=cells)
        with zero_private_adapters(model, cells):
            adapters_zero = evaluate_regime(**common)
        result[organ] = {
            "normal": normal,
            "freeze_grown_cells": frozen,
            "zero_grown_adapters": adapters_zero,
        }
    return result


def _mature_utility_lesions(
    model: Sol2,
    state,
    task,
    regime_a,
    regime_b,
    cfg,
    profile: AnchorProfile,
    *,
    max_steps: int,
    batches: int,
) -> dict:
    internal = model.internal_idx[model.internal_idx < len(profile.measured_cell)]
    order = torch.argsort(profile.measured_cell[internal], stable=True)
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
    decisions: list[dict],
    growth_events: list[dict],
    controller: DevelopmentController,
    profile: AnchorProfile,
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
        "decisions": decisions,
        "growth_events": growth_events,
        "controller": controller.state_dict(),
        "controller_config": controller.configuration(),
        "anchor_profile": profile.cpu_payload(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_developmental_attachment_branch(
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
    anchor_rate: float = 0.01,
    growth_cells: int = 16,
    high_pressure: float = 0.75,
    plateau_pressure: float = 0.60,
    plateau_gain: float = 0.03,
    patience_checks: int = 2,
    growth_min_update: int = 600,
    growth_refractory: int = 600,
    max_growth_events: int = 2,
    max_growth_a_drop: float = 0.05,
    resume: bool = False,
) -> dict:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    if min(
        adaptation_updates,
        eval_every,
        eval_batches,
        final_eval_batches,
        utility_batches,
        max_steps,
        growth_cells,
    ) < 1:
        raise ValueError("all experiment budgets must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "adaptation.pt"
    metrics_path = out_dir / "metrics.json"
    acquisition, cfg, model, optimizer, state = _load_acquisition(
        acquisition_path, device
    )
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
        directional_edges=True,
    )
    profile = make_anchor_profile(
        model,
        measured_cell,
        measured_edge,
        branch=branch,
        threshold=threshold,
        temperature=temperature,
        anchor_rate=anchor_rate,
    )
    policy = ProximalAnchorPolicy(model, profile)
    a_digest_before = _parameter_digest(_a_organ_named_parameters(model))
    organ_seed = 100_000 + cfg.seed
    organ = model.attach_organ("B", seed=organ_seed)
    add_optimizer_parameters(
        optimizer, organ.named_parameters(prefix="attached_organs.B"), cfg
    )
    cfg = cfg.scaled(updates=int(acquisition["update"]) + adaptation_updates)
    model.cfg = cfg
    controller = DevelopmentController(
        high_pressure=high_pressure,
        plateau_pressure=plateau_pressure,
        plateau_gain=plateau_gain,
        patience_checks=patience_checks,
        min_update=growth_min_update,
        refractory_updates=growth_refractory,
        max_events=max_growth_events,
    )
    update = 0
    accepted_updates = 0
    rejected_updates = 0
    records: list[dict] = []
    evaluations: list[dict] = []
    decisions: list[dict] = []
    growth_events: list[dict] = []

    zero_pressure = {
        "pressure": 0.0,
        "anchored_gradient_squared": 0.0,
        "substrate_gradient_squared": 0.0,
        "accessible_gradient_squared": 0.0,
    }
    baseline = _evaluate(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        policy,
        profile,
        update=0,
        max_steps=max_steps,
        batches=eval_batches,
        pressure=zero_pressure,
    )

    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if payload.get("schema") != SCHEMA or payload.get("branch") != branch:
            raise ValueError("incompatible developmental checkpoint")
        if not same_tensor_values(
            payload["anchor_profile"]["measured_cell"], measured_cell
        ):
            raise ValueError("utility calibration changed across resume")
        if payload["controller_config"] != controller.configuration():
            raise ValueError("developmental thresholds changed across resume")
        for event in payload["growth_events"]:
            added, migrate, grafts = grow_relay_tissue(
                model, optimizer, int(event["n_new"])
            )
            if added != event["cells"] or grafts != event["grafts"]:
                raise ValueError("growth replay changed across resume")
            state = migrate(state)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        state = unpack_state(payload["state"], device)
        cfg = type(cfg).from_dict(payload["config"])
        model.cfg = cfg
        update = int(payload["update"])
        accepted_updates = int(payload["accepted_updates"])
        rejected_updates = int(payload["rejected_updates"])
        records = list(payload["records"])
        evaluations = list(payload["evaluations"])
        decisions = list(payload["decisions"])
        growth_events = list(payload["growth_events"])
        controller.load_state_dict(payload["controller"])
        restore_rng_state(payload)

    while update < adaptation_updates:
        interval = min(eval_every, adaptation_updates - update)
        state, interval_records, accepted, rejected, pressure = _train_interval(
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
        update += interval
        accepted_updates += accepted
        rejected_updates += rejected
        records.extend(interval_records)
        evaluation = _evaluate(
            model,
            state,
            task,
            regime_a,
            regime_b,
            cfg,
            policy,
            profile,
            update=update,
            max_steps=max_steps,
            batches=eval_batches,
            pressure=pressure,
        )
        evaluations.append(evaluation)
        if branch == "developmental":
            decision = controller.observe(
                update=update,
                b_accuracy=evaluation["b_by_length"][str(max_steps)]["answer_accuracy"],
                pressure=pressure["pressure"],
            )
            decision_row = decision_payload(decision)
            if decision.trigger:
                pre_growth = evaluate_regime(
                    model,
                    state,
                    task,
                    regime_a,
                    cfg,
                    steps=max_steps,
                    batches=eval_batches,
                    generator_seed=cfg.seed + 150_000 + update,
                    organ_name="A",
                )
                added, migrate, grafts = grow_relay_tissue(
                    model, optimizer, growth_cells
                )
                state = migrate(state)
                cfg = model.cfg.scaled(
                    updates=int(acquisition["update"]) + adaptation_updates,
                    device=device,
                )
                model.cfg = cfg
                post_growth = evaluate_regime(
                    model,
                    state,
                    task,
                    regime_a,
                    cfg,
                    steps=max_steps,
                    batches=eval_batches,
                    generator_seed=cfg.seed + 150_000 + update,
                    organ_name="A",
                )
                drop = (
                    pre_growth["answer_accuracy"]
                    - post_growth["answer_accuracy"]
                )
                if drop > max_growth_a_drop:
                    raise RuntimeError(
                        f"growth A drop {drop:.4f} exceeds {max_growth_a_drop:.4f}"
                    )
                event = {
                    "update": update,
                    "reason": decision.reason,
                    "pressure": pressure["pressure"],
                    "n_new": growth_cells,
                    "cells": added,
                    "grafts": grafts,
                    "pre_growth_a": pre_growth,
                    "post_growth_a": post_growth,
                    "a_accuracy_drop": drop,
                    "parameters_after": count_parameters(model),
                }
                growth_events.append(event)
                decision_row["growth_event"] = len(growth_events) - 1
            decisions.append(decision_row)
        print(
            f"[{branch}] u{update} A4={evaluation['a_fixed']['answer_accuracy']:.3f} "
            f"B4={evaluation['b_by_length'][str(max_steps)]['answer_accuracy']:.3f} "
            f"pressure={pressure['pressure']:.3f} cells={model.n_cells}",
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
            decisions=decisions,
            growth_events=growth_events,
            controller=controller,
            profile=profile,
        )
        _atomic_json(
            metrics_path,
            {
                "branch": branch,
                "update": update,
                "calibration": calibration,
                "profile": anchor_profile_summary(model, profile),
                "baseline": baseline,
                "evaluations": evaluations,
                "decisions": decisions,
                "growth_events": growth_events,
                "records": records,
            },
        )

    final_pressure = evaluations[-1]["pressure"] if evaluations else zero_pressure
    final = _evaluate(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        policy,
        profile,
        update=update,
        max_steps=max_steps,
        batches=final_eval_batches,
        pressure=final_pressure,
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
    utility_lesions = _mature_utility_lesions(
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
    grown_cells = [cell for event in growth_events for cell in event["cells"]]
    growth_lesions = _growth_lesions(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        grown_cells,
        max_steps=max_steps,
        batches=final_eval_batches,
    )
    digest_after = _parameter_digest(_a_organ_named_parameters(model))
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
            "anchor_rate": anchor_rate,
            "growth_cells": growth_cells,
            "high_pressure": high_pressure,
            "plateau_pressure": plateau_pressure,
            "plateau_gain": plateau_gain,
            "patience_checks": patience_checks,
            "growth_min_update": growth_min_update,
            "growth_refractory": growth_refractory,
            "max_growth_events": max_growth_events,
            "max_growth_a_drop": max_growth_a_drop,
            "organ_seed": organ_seed,
        },
        "calibration": calibration,
        "profile": anchor_profile_summary(model, profile),
        "integrity": {
            "a_organ_digest_before": a_digest_before,
            "a_organ_digest_after": digest_after,
            "a_organ_unchanged": a_digest_before == digest_after,
        },
        "growth_events": growth_events,
        "decisions": decisions,
        "summary": {
            "baseline": baseline,
            "final": final,
            "b_lesions": b_lesions,
            "utility_lesions": utility_lesions,
            "growth_lesions": growth_lesions,
        },
        "records": records,
        "evaluations": evaluations,
    }
    _atomic_json(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SOL2 anchored developmental attachment")
    parser.add_argument("--acquisition-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adaptation-updates", type=int, default=3_000)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--final-eval-batches", type=int, default=64)
    parser.add_argument("--utility-batches", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--anchor-rate", type=float, default=0.01)
    parser.add_argument("--growth-cells", type=int, default=16)
    parser.add_argument("--high-pressure", type=float, default=0.75)
    parser.add_argument("--plateau-pressure", type=float, default=0.60)
    parser.add_argument("--plateau-gain", type=float, default=0.03)
    parser.add_argument("--patience-checks", type=int, default=2)
    parser.add_argument("--growth-min-update", type=int, default=600)
    parser.add_argument("--growth-refractory", type=int, default=600)
    parser.add_argument("--max-growth-events", type=int, default=2)
    parser.add_argument("--max-growth-a-drop", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_developmental_attachment_branch(
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
        anchor_rate=args.anchor_rate,
        growth_cells=args.growth_cells,
        high_pressure=args.high_pressure,
        plateau_pressure=args.plateau_pressure,
        plateau_gain=args.plateau_gain,
        patience_checks=args.patience_checks,
        growth_min_update=args.growth_min_update,
        growth_refractory=args.growth_refractory,
        max_growth_events=args.max_growth_events,
        max_growth_a_drop=args.max_growth_a_drop,
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
                "growth_events": result["growth_events"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
