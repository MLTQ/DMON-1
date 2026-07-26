"""Validate candidate SOL runs and atomically promote the best local checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from .checkpoint import load_organism


def _candidate(run_directory: Path) -> dict[str, Any]:
    summary_path = run_directory / "summary.json"
    checkpoint_path = run_directory / "best.pt"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise ValueError(
            f"{run_directory}: completed summary.json and best.pt are required"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("model") != "sol":
        raise ValueError(f"{run_directory}: only SOL checkpoints can be promoted")
    best_bpc = float(summary.get("best_bpc", math.nan))
    if not math.isfinite(best_bpc):
        raise ValueError(f"{run_directory}: best_bpc is missing or non-finite")
    organism = load_organism(checkpoint_path)
    checkpoint_bpc = float(organism.metadata.get("best_bpc", math.nan))
    if not math.isfinite(checkpoint_bpc):
        raise ValueError(f"{checkpoint_path}: checkpoint best_bpc is missing")
    if not math.isclose(best_bpc, checkpoint_bpc, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{run_directory}: summary/checkpoint best_bpc mismatch "
            f"({best_bpc} != {checkpoint_bpc})"
        )
    return {
        "run": run_directory,
        "checkpoint": checkpoint_path,
        "best_bpc": best_bpc,
        "updates": organism.updates,
        "cells": organism.model.cfg.cells,
        "dendrites": organism.model.cfg.dendrites,
    }


def promote_best_checkpoint(
    run_directories: list[str | Path],
    destination: str | Path = "sol/runs/live.pt",
) -> dict[str, Any]:
    """Promote the lowest-held-out-BPC valid SOL checkpoint."""

    if not run_directories:
        raise ValueError("at least one run directory is required")
    candidates = [_candidate(Path(path)) for path in run_directories]
    winner = min(candidates, key=lambda value: value["best_bpc"])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(winner["checkpoint"], temporary)
    os.replace(temporary, destination)
    manifest = {
        "checkpoint": str(destination),
        "source_run": str(winner["run"]),
        "source_checkpoint": str(winner["checkpoint"]),
        "best_bpc": winner["best_bpc"],
        "updates": winner["updates"],
        "cells": winner["cells"],
        "dendrites": winner["dendrites"],
        "candidates": [
            {
                "run": str(candidate["run"]),
                "best_bpc": candidate["best_bpc"],
                "updates": candidate["updates"],
            }
            for candidate in sorted(
                candidates, key=lambda value: value["best_bpc"]
            )
        ],
    }
    manifest_path = destination.with_suffix(".json")
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("sol/runs/live.pt")
    )
    args = parser.parse_args()
    manifest = promote_best_checkpoint(args.run, args.destination)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
