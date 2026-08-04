"""Resumable true-organ attachment branches for the SOL2 procedural organism."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint import pack_state, restore_rng_state, unpack_state
from .config import Sol2Config
from .model import Sol2, count_parameters
from .optim import add_optimizer_parameters, build_optimizer
from .procedural_acquisition import neuron_telemetry
from .procedural_benchmark import (
    evaluate_regime,
    evaluate_with_ablations,
    train_phase,
)
from .procedural_task import ProceduralTask, ProcedureRegime
from .train import build_model

SCHEMA = 1
BRANCHES = ("control", "full", "organ_only", "scratch")


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
        if name.startswith(
            ("embedding.", "sensory_gain", "sensory_bias", "output_organ.")
        ):
            yield name, parameter


def _substrate_named_parameters(model: Sol2):
    for name, parameter in model.named_parameters():
        if not name.startswith(
            (
                "embedding.",
                "sensory_gain",
                "sensory_bias",
                "output_organ.",
                "attached_organs.",
            )
        ):
            yield name, parameter


def _length_curve(
    model: Sol2,
    state,
    task: ProceduralTask,
    regime: ProcedureRegime,
    cfg: Sol2Config,
    *,
    organ_name: str,
    max_steps: int,
    batches: int,
    generator_seed: int,
) -> dict[str, dict]:
    return {
        str(steps): evaluate_regime(
            model,
            state,
            task,
            regime,
            cfg,
            steps=steps,
            batches=batches,
            generator_seed=generator_seed + steps,
            organ_name=organ_name,
        )
        for steps in range(1, max_steps + 1)
    }


def _load_acquisition(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != 1:
        raise ValueError("unsupported acquisition checkpoint schema")
    cfg = Sol2Config.from_dict(payload["config"])
    if cfg.device != device:
        cfg = cfg.scaled(device=device)
    model = build_model("creature", cfg, device)
    assert isinstance(model, Sol2)
    model.load_state_dict(payload["model"])
    optimizer = build_optimizer(model, cfg)
    optimizer.load_state_dict(payload["optimizer"])
    state = unpack_state(payload["state"], device)
    return payload, cfg, model, optimizer, state


def _build_branch(
    acquisition_path: Path,
    device: str,
    branch: str,
):
    acquisition, cfg, acquired, acquired_optimizer, acquired_state = _load_acquisition(
        acquisition_path, device
    )
    organ_seed = 100_000 + cfg.seed
    if branch == "control":
        return acquisition, cfg, acquired, acquired_optimizer, acquired_state, "A"

    if branch == "scratch":
        model = build_model("creature", cfg, device)
        assert isinstance(model, Sol2)
        model.attach_organ("B", seed=organ_seed)
        optimizer = build_optimizer(model, cfg)
        state = model.initial_state(cfg.batch_size, device)
        return acquisition, cfg, model, optimizer, state, "B"

    acquired.attach_organ("B", seed=organ_seed)
    if branch == "organ_only":
        for parameter in acquired.parameters():
            parameter.requires_grad_(False)
        for parameter in acquired.organ_parameters("B"):
            parameter.requires_grad_(True)
        optimizer = build_optimizer(acquired, cfg)
    else:
        optimizer = acquired_optimizer
        add_optimizer_parameters(
            optimizer,
            acquired.attached_organs["B"].named_parameters(
                prefix="attached_organs.B"
            ),
            cfg,
        )
    return acquisition, cfg, acquired, optimizer, acquired_state, "B"


def _save_branch_checkpoint(
    path: Path,
    *,
    branch: str,
    model: Sol2,
    optimizer,
    state,
    cfg: Sol2Config,
    update: int,
    accepted_updates: int,
    rejected_updates: int,
    records: list[dict],
    evaluations: list[dict],
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
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _evaluation(
    model: Sol2,
    state,
    task: ProceduralTask,
    regime_a: ProcedureRegime,
    regime_b: ProcedureRegime,
    cfg: Sol2Config,
    *,
    update: int,
    selected_organ: str,
    selected_regime: ProcedureRegime,
    max_steps: int,
    eval_batches: int,
) -> dict:
    a_seed = cfg.seed + 40_000 + update
    b_seed = cfg.seed + 50_000 + update
    selected_seed = a_seed if selected_organ == "A" else b_seed
    selected = _length_curve(
        model,
        state,
        task,
        selected_regime,
        cfg,
        organ_name=selected_organ,
        max_steps=max_steps,
        batches=eval_batches,
        generator_seed=selected_seed,
    )
    result = {
        "update": update,
        "selected_organ": selected_organ,
        "selected_by_length": selected,
        "telemetry": neuron_telemetry(model, state),
    }
    if selected_organ == "B":
        result["a_immediate"] = evaluate_regime(
            model,
            state,
            task,
            regime_a,
            cfg,
            steps=max_steps,
            batches=1,
            generator_seed=a_seed + max_steps,
            organ_name="A",
        )
        result["a_sustained"] = evaluate_regime(
            model,
            state,
            task,
            regime_a,
            cfg,
            steps=max_steps,
            batches=eval_batches,
            generator_seed=a_seed + max_steps,
            organ_name="A",
        )
        # A and B evaluations use separate cloned states and therefore cannot prime
        # or erase one another.
        result["b_fixed4"] = selected[str(max_steps)]
    else:
        result["a_immediate"] = selected[str(max_steps)]
        result["a_sustained"] = selected[str(max_steps)]
    return result


def _window(records: list[dict], start: int, stop: int) -> dict:
    rows = records[start:stop]
    return {
        "updates": len(rows),
        "answer_accuracy": sum(row["answer_accuracy"] for row in rows) / len(rows),
        "answer_bits": sum(row["answer_bits"] for row in rows) / len(rows),
    }


def _run_removal_and_reattachment(
    model: Sol2,
    optimizer,
    state,
    task: ProceduralTask,
    regime_a: ProcedureRegime,
    regime_b: ProcedureRegime,
    cfg: Sol2Config,
    *,
    adaptation_updates: int,
    source_update: int,
    a_updates: int,
    recovery_updates: int,
    max_steps: int,
    eval_batches: int,
) -> tuple[dict, object]:
    seed = cfg.seed + 70_000
    before = evaluate_regime(
        model,
        state,
        task,
        regime_b,
        cfg,
        steps=max_steps,
        batches=eval_batches,
        generator_seed=seed,
        organ_name="B",
    )
    detached = model.detach_organ("B")
    a_phase = train_phase(
        model,
        optimizer,
        state,
        task,
        regime_a,
        cfg,
        updates=a_updates,
        global_update_start=source_update + adaptation_updates,
        min_steps=1,
        max_steps=max_steps,
        generator_seed=seed + 1_000,
        log_every=cfg.log_every,
        organ_name="A",
    )
    state = a_phase.state
    a_after = evaluate_regime(
        model,
        state,
        task,
        regime_a,
        cfg,
        steps=max_steps,
        batches=eval_batches,
        generator_seed=seed + 2_000,
        organ_name="A",
    )
    model.reattach_organ("B", detached)
    recovery = {
        "0": evaluate_regime(
            model,
            state,
            task,
            regime_b,
            cfg,
            steps=max_steps,
            batches=eval_batches,
            generator_seed=seed + 3_000,
            organ_name="B",
        )
    }
    recovery_records = []
    completed = 0
    for target in (25, 100, recovery_updates):
        if target <= completed or target > recovery_updates:
            continue
        phase = train_phase(
            model,
            optimizer,
            state,
            task,
            regime_b,
            cfg,
            updates=target - completed,
            global_update_start=(
                source_update + adaptation_updates + a_updates + completed
            ),
            min_steps=1,
            max_steps=max_steps,
            generator_seed=seed + 4_000 + completed,
            log_every=cfg.log_every,
            organ_name="B",
        )
        state = phase.state
        recovery_records.extend(phase.records)
        completed = target
        recovery[str(target)] = evaluate_regime(
            model,
            state,
            task,
            regime_b,
            cfg,
            steps=max_steps,
            batches=eval_batches,
            generator_seed=seed + 5_000 + target,
            organ_name="B",
        )
    return (
        {
            "b_before_removal": before,
            "a_exercise_updates": a_updates,
            "a_after_removal": a_after,
            "b_recovery": recovery,
            "a_records": a_phase.records,
            "b_recovery_records": recovery_records,
        },
        state,
    )


def run_organ_attachment_branch(
    acquisition_path: Path,
    out_dir: Path,
    branch: str,
    *,
    device: str,
    adaptation_updates: int,
    eval_every: int,
    eval_batches: int,
    final_eval_batches: int,
    max_steps: int,
    a_detached_updates: int,
    recovery_updates: int,
    resume: bool = False,
) -> dict:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    if (
        adaptation_updates < 1
        or eval_every < 1
        or eval_batches < 1
        or final_eval_batches < 1
        or max_steps < 1
    ):
        raise ValueError("adaptation and evaluation budgets must be positive")
    if a_detached_updates < 0 or recovery_updates < 0:
        raise ValueError("removal and recovery budgets cannot be negative")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "adaptation.pt"
    metrics_path = out_dir / "metrics.json"
    acquisition, cfg, model, optimizer, state, selected_organ = _build_branch(
        acquisition_path, device, branch
    )
    organ_seed = 100_000 + cfg.seed
    cfg = cfg.scaled(updates=int(acquisition["update"]) + adaptation_updates)
    task = ProceduralTask()
    regime_a = task.base_regime("A-acquisition", cfg.seed + 101)
    regime_b = task.remapped_interface(regime_a, "B-attached-organ", cfg.seed + 202)
    selected_regime = regime_a if branch == "control" else regime_b
    update = 0
    accepted_updates = 0
    rejected_updates = 0
    records: list[dict] = []
    evaluations: list[dict] = []

    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if payload.get("schema") != SCHEMA or payload.get("branch") != branch:
            raise ValueError("incompatible organ-attachment checkpoint")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        state = unpack_state(payload["state"], device)
        update = int(payload["update"])
        accepted_updates = int(payload["accepted_updates"])
        rejected_updates = int(payload["rejected_updates"])
        records = list(payload["records"])
        evaluations = list(payload["evaluations"])
        restore_rng_state(payload)

    a_digest_before = _parameter_digest(_a_organ_named_parameters(model))
    substrate_digest_before = _parameter_digest(_substrate_named_parameters(model))
    while update < adaptation_updates:
        interval = min(eval_every, adaptation_updates - update)
        sample_seed = cfg.seed + (20_000 if selected_organ == "B" else 10_000) + update
        phase = train_phase(
            model,
            optimizer,
            state,
            task,
            selected_regime,
            cfg,
            updates=interval,
            global_update_start=int(acquisition["update"]) + update,
            min_steps=1,
            max_steps=max_steps,
            generator_seed=sample_seed,
            log_every=cfg.log_every,
            organ_name=selected_organ,
        )
        state = phase.state
        update += interval
        accepted_updates += phase.accepted_updates
        rejected_updates += phase.rejected_updates
        records.extend(phase.records)
        evaluation = _evaluation(
            model,
            state,
            task,
            regime_a,
            regime_b,
            cfg,
            update=update,
            selected_organ=selected_organ,
            selected_regime=selected_regime,
            max_steps=max_steps,
            eval_batches=eval_batches,
        )
        evaluations.append(evaluation)
        print(
            f"[{branch}] u{update} {selected_organ} length4="
            f"{evaluation['selected_by_length'][str(max_steps)]['answer_accuracy']:.3f} "
            f"rejects={rejected_updates}",
            flush=True,
        )
        _save_branch_checkpoint(
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
        )
        _atomic_json(
            metrics_path,
            {
                "branch": branch,
                "source_update": int(acquisition["update"]),
                "update": update,
                "total_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "trainable_parameters": count_parameters(model),
                "accepted_updates": accepted_updates,
                "rejected_updates": rejected_updates,
                "config": cfg.to_dict(),
                "records": records,
                "evaluations": evaluations,
            },
        )

    final_evaluation = _evaluation(
        model,
        state,
        task,
        regime_a,
        regime_b,
        cfg,
        update=update,
        selected_organ=selected_organ,
        selected_regime=selected_regime,
        max_steps=max_steps,
        eval_batches=final_eval_batches,
    )
    a_digest_after_adaptation = _parameter_digest(_a_organ_named_parameters(model))
    substrate_digest_after_adaptation = _parameter_digest(
        _substrate_named_parameters(model)
    )
    lesions = evaluate_with_ablations(
        model,
        state,
        task,
        selected_regime,
        cfg,
        steps=max_steps,
        batches=final_eval_batches,
        generator_seed=cfg.seed + 60_000,
        organ_name=selected_organ,
    )
    removal = None
    if branch == "full":
        removal, state = _run_removal_and_reattachment(
            model,
            optimizer,
            state,
            task,
            regime_a,
            regime_b,
            cfg,
            adaptation_updates=adaptation_updates,
            source_update=int(acquisition["update"]),
            a_updates=a_detached_updates,
            recovery_updates=recovery_updates,
            max_steps=max_steps,
            eval_batches=final_eval_batches,
        )

    early_width = max(1, round(len(records) * 0.10))
    late_width = max(1, round(len(records) * 0.25))
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
            "max_steps": max_steps,
            "a_detached_updates": a_detached_updates,
            "recovery_updates": recovery_updates,
            "organ_seed": organ_seed,
        },
        "integrity": {
            "a_organ_digest_before": a_digest_before,
            "a_organ_digest_after_adaptation": a_digest_after_adaptation,
            "a_organ_unchanged": a_digest_before == a_digest_after_adaptation,
            "substrate_digest_before": substrate_digest_before,
            "substrate_digest_after_adaptation": substrate_digest_after_adaptation,
            "substrate_unchanged": (
                substrate_digest_before == substrate_digest_after_adaptation
            ),
        },
        "summary": {
            "early": _window(records, 0, early_width),
            "late": _window(records, len(records) - late_width, len(records)),
            "final_selected_by_length": final_evaluation["selected_by_length"],
            "final_a_immediate": final_evaluation["a_immediate"],
            "final_a_sustained": final_evaluation["a_sustained"],
            "lesions": lesions,
            "removal_and_reattachment": removal,
        },
        "records": records,
        "evaluations": evaluations,
    }
    _atomic_json(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SOL2 true organ attachment branch")
    parser.add_argument("--acquisition-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adaptation-updates", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--final-eval-batches", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--a-detached-updates", type=int, default=500)
    parser.add_argument("--recovery-updates", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_organ_attachment_branch(
        args.acquisition_checkpoint,
        args.out_dir,
        args.branch,
        device=args.device,
        adaptation_updates=args.adaptation_updates,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        final_eval_batches=args.final_eval_batches,
        max_steps=args.max_steps,
        a_detached_updates=args.a_detached_updates,
        recovery_updates=args.recovery_updates,
        resume=args.resume,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
