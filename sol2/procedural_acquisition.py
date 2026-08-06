"""Resumable mastery-gated procedural acquisition and neuron telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import pack_state, restore_rng_state, unpack_state
from .config import Sol2Config
from .model import Sol2, count_parameters
from .optim import build_optimizer
from .procedural_benchmark import evaluate_regime, train_phase
from .procedural_task import ProceduralTask
from .train import build_model

SCHEMA = 1


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    model: Sol2,
    optimizer: torch.optim.Optimizer,
    state,
    cfg: Sol2Config,
    update: int,
    accepted_updates: int,
    rejected_updates: int,
    records: list[dict],
    evaluations: list[dict],
    mastery_streak: int,
) -> None:
    payload = {
        "schema": SCHEMA,
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": pack_state(state),
        "update": update,
        "accepted_updates": accepted_updates,
        "rejected_updates": rejected_updates,
        "records": records,
        "evaluations": evaluations,
        "mastery_streak": mastery_streak,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, device: str) -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported procedural acquisition checkpoint schema")
    return payload


@torch.no_grad()
def neuron_telemetry(model: Sol2, state) -> dict:
    """Reduce a living state and learned anatomy to visualization-sized JSON."""

    hidden = state.hidden.detach()
    activation_mean_abs = hidden.abs().mean(dim=(0, 2)).cpu()
    activation_rms = hidden.square().mean(dim=(0, 2)).sqrt().cpu()
    activation_std = hidden.std(dim=(0, 2), unbiased=False).cpu()
    identity_rms = torch.zeros(model.n_cells)
    adapter_up_rms = torch.zeros(model.n_cells)
    if model.cell_gain is not None:
        expression = (
            model.cell_gain.detach().square() + model.cell_bias.detach().square()
        ).mean(-1).sqrt().cpu()
        identity_rms[model.expression_cells.cpu()] = expression
    if model.cell_adapter_up is not None:
        adapter_expression = (
            model.cell_adapter_up.detach().square().mean(dim=(1, 2)).sqrt().cpu()
        )
        adapter_up_rms[model.expression_cells.cpu()] = adapter_expression

    tissue_by_cell = ["unknown"] * model.n_cells
    for tissue in ("input", "memory", "compute", "relay", "output"):
        for cell in model.tissue_indices(tissue).detach().cpu().tolist():
            tissue_by_cell[cell] = tissue

    active = model.graph.active.detach().cpu()
    sources = model.graph.sources.detach().cpu()
    edge_bias = (
        model.cfg.edge_logit_limit * torch.tanh(model.graph.edge_logit.detach())
    ).cpu()
    edges = []
    for target in range(model.n_cells):
        for slot in torch.nonzero(active[target], as_tuple=True)[0].tolist():
            edges.append(
                {
                    "source": int(sources[target, slot]),
                    "target": target,
                    "slot": slot,
                    "bias": float(edge_bias[target, slot]),
                }
            )
    indegree = active.sum(-1).tolist()
    cells = [
        {
            "id": cell,
            "tissue": tissue_by_cell[cell],
            "activation_mean_abs": float(activation_mean_abs[cell]),
            "activation_rms": float(activation_rms[cell]),
            "activation_std": float(activation_std[cell]),
            "identity_rms": float(identity_rms[cell]),
            "adapter_up_rms": float(adapter_up_rms[cell]),
            "active_indegree": int(indegree[cell]),
        }
        for cell in range(model.n_cells)
    ]
    return {"cells": cells, "edges": edges}


def _compatible_geometry(saved: Sol2Config, requested: Sol2Config) -> bool:
    ignored = {"device", "updates"}
    return all(
        key in ignored or value == requested.to_dict()[key]
        for key, value in saved.to_dict().items()
    )


def run_acquisition_calibration(
    cfg: Sol2Config,
    out_dir: Path,
    *,
    max_updates: int,
    evaluation_interval: int,
    stage_updates: int,
    eval_batches: int,
    mastery_accuracy: float,
    mastery_checks: int,
    resume: bool = False,
) -> dict:
    """Train staged composition until repeated fixed-length mastery or a hard cap."""

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "acquisition.pt"
    metrics_path = out_dir / "metrics.json"
    task = ProceduralTask()
    cfg = cfg.scaled(vocab_size=task.vocab_size, updates=max_updates)
    regime = task.base_regime("A-acquisition", cfg.seed + 101)
    model = build_model("creature", cfg, cfg.device)
    assert isinstance(model, Sol2)
    optimizer = build_optimizer(model, cfg)
    state = model.initial_state(cfg.batch_size, cfg.device)
    update = 0
    accepted_updates = 0
    rejected_updates = 0
    records: list[dict] = []
    evaluations: list[dict] = []
    mastery_streak = 0

    if resume and checkpoint_path.exists():
        payload = _load_checkpoint(checkpoint_path, cfg.device)
        saved_cfg = Sol2Config.from_dict(payload["config"])
        if not _compatible_geometry(saved_cfg, cfg):
            raise ValueError("resume configuration changes acquisition geometry")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        state = unpack_state(payload["state"], cfg.device)
        update = int(payload["update"])
        accepted_updates = int(payload["accepted_updates"])
        rejected_updates = int(payload["rejected_updates"])
        records = list(payload["records"])
        evaluations = list(payload["evaluations"])
        mastery_streak = int(payload["mastery_streak"])
        restore_rng_state(payload)

    while update < max_updates and mastery_streak < mastery_checks:
        interval_updates = min(evaluation_interval, max_updates - update)
        admitted_steps = min(4, 1 + update // stage_updates)
        phase = train_phase(
            model,
            optimizer,
            state,
            task,
            regime,
            cfg,
            updates=interval_updates,
            global_update_start=update,
            min_steps=1,
            max_steps=admitted_steps,
            generator_seed=cfg.seed + 10_000 + update,
            log_every=cfg.log_every,
        )
        state = phase.state
        update += interval_updates
        accepted_updates += phase.accepted_updates
        rejected_updates += phase.rejected_updates
        records.extend(phase.records)
        fixed = evaluate_regime(
            model,
            state,
            task,
            regime,
            cfg,
            steps=4,
            batches=eval_batches,
            generator_seed=cfg.seed + 20_000 + update,
        )
        mastery_streak = (
            mastery_streak + 1
            if fixed["answer_accuracy"] >= mastery_accuracy
            else 0
        )
        evaluation = {
            "update": update,
            "admitted_steps": admitted_steps,
            **fixed,
            "mastery_streak": mastery_streak,
            "telemetry": neuron_telemetry(model, state),
        }
        evaluations.append(evaluation)
        print(
            f"[mastery] u{update} admitted={admitted_steps} "
            f"length4={fixed['answer_accuracy']:.3f} "
            f"streak={mastery_streak}/{mastery_checks}",
            flush=True,
        )
        _save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            state=state,
            cfg=cfg,
            update=update,
            accepted_updates=accepted_updates,
            rejected_updates=rejected_updates,
            records=records,
            evaluations=evaluations,
            mastery_streak=mastery_streak,
        )
        _atomic_json(
            metrics_path,
            {
                "kind": "creature",
                "parameters": count_parameters(model),
                "config": cfg.to_dict(),
                "protocol": {
                    "max_updates": max_updates,
                    "evaluation_interval": evaluation_interval,
                    "stage_updates": stage_updates,
                    "eval_batches": eval_batches,
                    "mastery_accuracy": mastery_accuracy,
                    "mastery_checks": mastery_checks,
                },
                "update": update,
                "accepted_updates": accepted_updates,
                "rejected_updates": rejected_updates,
                "mastered": mastery_streak >= mastery_checks,
                "evaluations": evaluations,
                "records": records,
            },
        )

    return json.loads(metrics_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Mastery-gated SOL2 acquisition")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-updates", type=int, default=10_000)
    parser.add_argument("--evaluation-interval", type=int, default=1_000)
    parser.add_argument("--stage-updates", type=int, default=1_000)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--mastery-accuracy", type=float, default=0.80)
    parser.add_argument("--mastery-checks", type=int, default=2)
    parser.add_argument("--operator-bound", type=float, default=4.0)
    parser.add_argument("--n-memory", type=int, default=16)
    parser.add_argument("--n-compute", type=int, default=64)
    parser.add_argument("--n-relay", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--n-dendrites", type=int, default=12)
    parser.add_argument("--initial-active-dendrites", type=int, default=8)
    parser.add_argument("--steps-per-token", type=int, default=3)
    parser.add_argument("--organ-queries", type=int, default=4)
    parser.add_argument("--cell-adapter-rank", type=int, default=0)
    parser.add_argument("--cell-adapter-gain", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = Sol2Config(
        n_memory=args.n_memory,
        n_compute=args.n_compute,
        n_relay=args.n_relay,
        hidden=args.hidden,
        n_dendrites=args.n_dendrites,
        initial_active_dendrites=args.initial_active_dendrites,
        steps_per_token=args.steps_per_token,
        organ_queries=args.organ_queries,
        cell_adapter_rank=args.cell_adapter_rank,
        cell_adapter_gain=args.cell_adapter_gain,
        batch_size=args.batch_size,
        operator_bound=args.operator_bound,
        seed=args.seed,
        device=args.device,
    )
    result = run_acquisition_calibration(
        cfg,
        args.out_dir,
        max_updates=args.max_updates,
        evaluation_interval=args.evaluation_interval,
        stage_updates=args.stage_updates,
        eval_batches=args.eval_batches,
        mastery_accuracy=args.mastery_accuracy,
        mastery_checks=args.mastery_checks,
        resume=args.resume,
    )
    last = result["evaluations"][-1]
    print(
        json.dumps(
            {
                "parameters": result["parameters"],
                "update": result["update"],
                "mastered": result["mastered"],
                "last_evaluation": {
                    key: value for key, value in last.items() if key != "telemetry"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
