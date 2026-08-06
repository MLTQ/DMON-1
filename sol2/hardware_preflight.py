"""Portable capacity and throughput preflight for unfamiliar SOL2 CUDA hosts."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import Sol2Config
from .growth import grow_relay_tissue
from .model import Sol2, count_parameters
from .optim import build_optimizer, guarded_step
from .procedural_benchmark import _step_episode
from .procedural_task import ProceduralTask
from .train import build_model

SCHEMA = 1


def _git_revision() -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _device_fingerprint(device: torch.device) -> dict:
    result = {
        "requested": str(device),
        "type": device.type,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "git_revision": _git_revision(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "multiprocessors": properties.multi_processor_count,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    else:
        result.update(
            {
                "name": platform.processor() or "cpu",
                "compute_capability": None,
                "total_memory_bytes": None,
                "multiprocessors": None,
                "bf16_supported": False,
            }
        )
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_updates(
    model: Sol2,
    optimizer: torch.optim.Optimizer,
    state,
    task: ProceduralTask,
    cfg: Sol2Config,
    *,
    updates: int,
    program_steps: int,
    generator_seed: int,
) -> tuple[object, int, int]:
    generator = torch.Generator().manual_seed(generator_seed)
    regime = task.base_regime("hardware-preflight", cfg.seed + 101)
    accepted = 0
    rejected = 0
    model.train()
    for _ in range(updates):
        batch = task.sample_batch(
            regime, cfg.batch_size, program_steps, generator, cfg.device
        )
        logits, next_state = _step_episode(model, batch.tokens, state)
        loss = F.cross_entropy(logits, batch.answer_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        health = guarded_step(model, optimizer, cfg)
        if health.accepted:
            accepted += 1
        else:
            rejected += 1
        state = next_state.detach()
    return state, accepted, rejected


def _measure_phase(
    name: str,
    model: Sol2,
    optimizer: torch.optim.Optimizer,
    state,
    task: ProceduralTask,
    cfg: Sol2Config,
    *,
    updates: int,
    program_steps: int,
    generator_seed: int,
) -> tuple[object, dict]:
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    state, accepted, rejected = _run_updates(
        model,
        optimizer,
        state,
        task,
        cfg,
        updates=updates,
        program_steps=program_steps,
        generator_seed=generator_seed,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    return state, {
        "name": name,
        "cells": model.n_cells,
        "trainable_parameters": count_parameters(model),
        "updates": updates,
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "elapsed_seconds": elapsed,
        "updates_per_second": updates / elapsed,
        "examples_per_second": updates * cfg.batch_size / elapsed,
        "answer_examples": updates * cfg.batch_size,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def run_hardware_preflight(
    cfg: Sol2Config,
    *,
    warmup_updates: int,
    measure_updates: int,
    program_steps: int,
    growth_cells: int,
) -> dict:
    """Benchmark the exact eager procedural path before result-bearing work."""

    if min(warmup_updates, measure_updates, program_steps) < 1 or growth_cells < 0:
        raise ValueError("preflight budgets must be positive and growth nonnegative")
    cfg = cfg.scaled(vocab_size=ProceduralTask().vocab_size)
    base_config = cfg.to_dict()
    torch.manual_seed(cfg.seed)
    if torch.device(cfg.device).type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    task = ProceduralTask()
    model = build_model("creature", cfg, cfg.device)
    assert isinstance(model, Sol2)
    optimizer = build_optimizer(model, cfg)
    state = model.initial_state(cfg.batch_size, cfg.device)

    state, _, _ = _run_updates(
        model,
        optimizer,
        state,
        task,
        cfg,
        updates=warmup_updates,
        program_steps=program_steps,
        generator_seed=cfg.seed + 10_000,
    )
    phases = []
    state, base = _measure_phase(
        "base",
        model,
        optimizer,
        state,
        task,
        cfg,
        updates=measure_updates,
        program_steps=program_steps,
        generator_seed=cfg.seed + 20_000,
    )
    phases.append(base)

    growth = {"requested_cells": growth_cells, "cells": [], "grafts": []}
    if growth_cells:
        added, migrate, grafts = grow_relay_tissue(model, optimizer, growth_cells)
        state = migrate(state)
        cfg = model.cfg.scaled(device=cfg.device)
        growth.update({"cells": added, "grafts": grafts})
        state, _, _ = _run_updates(
            model,
            optimizer,
            state,
            task,
            cfg,
            updates=warmup_updates,
            program_steps=program_steps,
            generator_seed=cfg.seed + 30_000,
        )
        state, grown = _measure_phase(
            "grown",
            model,
            optimizer,
            state,
            task,
            cfg,
            updates=measure_updates,
            program_steps=program_steps,
            generator_seed=cfg.seed + 40_000,
        )
        phases.append(grown)

    fingerprint = _device_fingerprint(torch.device(cfg.device))
    capacity = {
        "estimated_concurrent_processes_at_80pct": None,
        "basis": "peak_reserved_bytes; estimate only, validate with separate processes",
    }
    peak_reserved = max(
        (row["peak_reserved_bytes"] or 0 for row in phases), default=0
    )
    total_memory = fingerprint["total_memory_bytes"]
    if peak_reserved > 0 and total_memory is not None:
        capacity["estimated_concurrent_processes_at_80pct"] = max(
            1, math.floor(0.80 * total_memory / peak_reserved)
        )
    return {
        "schema": SCHEMA,
        "hardware": fingerprint,
        "protocol": {
            "precision": "float32",
            "warmup_updates_per_phase": warmup_updates,
            "measure_updates_per_phase": measure_updates,
            "program_steps": program_steps,
            "growth_cells": growth_cells,
            "base_config": base_config,
        },
        "phases": phases,
        "growth": growth,
        "capacity": capacity,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a SOL2 CUDA host")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measure-updates", type=int, default=25)
    parser.add_argument("--program-steps", type=int, default=4)
    parser.add_argument("--growth-cells", type=int, default=32)
    parser.add_argument("--n-memory", type=int, default=64)
    parser.add_argument("--n-compute", type=int, default=256)
    parser.add_argument("--n-relay", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--n-dendrites", type=int, default=16)
    parser.add_argument("--initial-active-dendrites", type=int, default=12)
    parser.add_argument("--steps-per-token", type=int, default=5)
    parser.add_argument("--organ-queries", type=int, default=8)
    parser.add_argument("--cell-adapter-rank", type=int, default=4)
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
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        operator_bound=4.0,
    )
    result = run_hardware_preflight(
        cfg,
        warmup_updates=args.warmup_updates,
        measure_updates=args.measure_updates,
        program_steps=args.program_steps,
        growth_cells=args.growth_cells,
    )
    rendered = json.dumps(result, indent=2)
    if args.out is not None:
        _atomic_json(args.out, result)
    print(rendered)


if __name__ == "__main__":
    main()
