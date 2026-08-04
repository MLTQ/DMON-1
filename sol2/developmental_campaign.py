"""Generate capacity-gated multi-seed and multi-size SOL2 campaign manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import write_manifest

BRANCHES = ("plastic", "uniform_anchor", "measured_anchor", "developmental")
DEFAULT_SIZES = (400, 800, 1600, 3200)


def geometry_for_cells(total_cells: int) -> dict:
    """Keep physical organs/memory fixed and scale compute:relay at four to one."""

    fixed = 8 + 64 + 8
    flexible = total_cells - fixed
    if total_cells < 400 or flexible % 5:
        raise ValueError("total cells must be >=400 and leave a 4:1 compute/relay split")
    relay = flexible // 5
    compute = 4 * relay
    return {
        "total_cells": total_cells,
        "n_input": 8,
        "n_memory": 64,
        "n_compute": compute,
        "n_relay": relay,
        "n_output": 8,
        "growth_cells_per_event": total_cells // 25,
    }


def build_jobs(
    *,
    seeds: list[int],
    sizes: list[int],
    acquisition_updates: int,
    adaptation_updates: int,
) -> tuple[list[dict], dict]:
    """Build capacity probes, mastery acquisitions, and matched attachment arms."""

    if not seeds or not sizes or min(seeds) < 0:
        raise ValueError("campaign needs nonnegative seeds and at least one size")
    if acquisition_updates < 1 or adaptation_updates < 1:
        raise ValueError("campaign update budgets must be positive")
    geometries = [geometry_for_cells(size) for size in sizes]
    jobs = []
    for geometry in geometries:
        size = geometry["total_cells"]
        size_name = f"n{size}"
        preflight_id = f"preflight-{size_name}"
        jobs.append(
            {
                "id": preflight_id,
                "kind": "capacity_preflight",
                "command": [
                    "{python}",
                    "-m",
                    "sol2.hardware_preflight",
                    "--device",
                    "cuda:0",
                    "--out",
                    f"{{root}}/preflight/{size_name}.json",
                    "--batch-size",
                    "24",
                    "--n-memory",
                    "64",
                    "--n-compute",
                    str(geometry["n_compute"]),
                    "--n-relay",
                    str(geometry["n_relay"]),
                    "--hidden",
                    "192",
                    "--n-dendrites",
                    "16",
                    "--initial-active-dendrites",
                    "12",
                    "--steps-per-token",
                    "5",
                    "--organ-queries",
                    "8",
                    "--cell-adapter-rank",
                    "4",
                    "--growth-cells",
                    str(2 * geometry["growth_cells_per_event"]),
                ],
                "depends_on": [],
                "expected_artifact": f"preflight/{size_name}.json",
                "completion_check": {"json_field": ["schema"], "equals": 1},
            }
        )
        for seed in seeds:
            run_root = f"{size_name}/s{seed}"
            acquisition_id = f"acquire-{size_name}-s{seed}"
            jobs.append(
                {
                    "id": acquisition_id,
                    "kind": "mastery_acquisition",
                    "command": [
                        "{python}",
                        "-m",
                        "sol2.procedural_acquisition",
                        "--device",
                        "cuda:0",
                        "--out-dir",
                        f"{{root}}/{run_root}/acquisition",
                        "--seed",
                        str(seed),
                        "--batch-size",
                        "24",
                        "--max-updates",
                        str(acquisition_updates),
                        "--evaluation-interval",
                        "1000",
                        "--stage-updates",
                        "1000",
                        "--eval-batches",
                        "64",
                        "--mastery-accuracy",
                        "0.80",
                        "--mastery-checks",
                        "2",
                        "--operator-bound",
                        "4.0",
                        "--n-memory",
                        "64",
                        "--n-compute",
                        str(geometry["n_compute"]),
                        "--n-relay",
                        str(geometry["n_relay"]),
                        "--hidden",
                        "192",
                        "--n-dendrites",
                        "16",
                        "--initial-active-dendrites",
                        "12",
                        "--steps-per-token",
                        "5",
                        "--organ-queries",
                        "8",
                        "--cell-adapter-rank",
                        "4",
                        "--cell-adapter-gain",
                        "0.5",
                        "--resume",
                    ],
                    "depends_on": [preflight_id],
                    "expected_artifact": f"{run_root}/acquisition/metrics.json",
                    "completion_check": {
                        "json_field": ["mastered"],
                        "equals": True,
                    },
                }
            )
            for branch in BRANCHES:
                job_id = f"adapt-{size_name}-s{seed}-{branch}"
                jobs.append(
                    {
                        "id": job_id,
                        "kind": "developmental_attachment",
                        "command": [
                            "{python}",
                            "-m",
                            "sol2.developmental_attachment",
                            "--device",
                            "cuda:0",
                            "--acquisition-checkpoint",
                            f"{{root}}/{run_root}/acquisition/acquisition.pt",
                            "--out-dir",
                            f"{{root}}/{run_root}/{branch}",
                            "--branch",
                            branch,
                            "--adaptation-updates",
                            str(adaptation_updates),
                            "--eval-every",
                            "300",
                            "--eval-batches",
                            "16",
                            "--final-eval-batches",
                            "64",
                            "--utility-batches",
                            "64",
                            "--max-steps",
                            "4",
                            "--threshold",
                            "0.65",
                            "--temperature",
                            "0.10",
                            "--anchor-rate",
                            "0.01",
                            "--growth-cells",
                            str(geometry["growth_cells_per_event"]),
                            "--high-pressure",
                            "0.75",
                            "--plateau-pressure",
                            "0.60",
                            "--plateau-gain",
                            "0.03",
                            "--patience-checks",
                            "2",
                            "--growth-min-update",
                            "600",
                            "--growth-refractory",
                            "600",
                            "--max-growth-events",
                            "2",
                            "--max-growth-a-drop",
                            "0.05",
                            "--resume",
                        ],
                        "depends_on": [acquisition_id],
                        "expected_artifact": f"{run_root}/{branch}/metrics.json",
                        "completion_check": {
                            "json_field": ["update"],
                            "equals": adaptation_updates,
                        },
                    }
                )
    protocol = {
        "name": "developmental-scaling-candidate",
        "status": "capacity-gated; scientific claims freeze after S1-P4 completes",
        "seeds": seeds,
        "geometries": geometries,
        "branches": list(BRANCHES),
        "acquisition_updates": acquisition_updates,
        "adaptation_updates": adaptation_updates,
        "effective_batch_size": 24,
        "fixed_hidden": 192,
        "fixed_installed_degree": 12,
    }
    return jobs, protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a SOL2 developmental campaign")
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--acquisition-updates", type=int, default=10_000)
    parser.add_argument("--adaptation-updates", type=int, default=3_000)
    args = parser.parse_args()
    jobs, protocol = build_jobs(
        seeds=args.seeds,
        sizes=args.sizes,
        acquisition_updates=args.acquisition_updates,
        adaptation_updates=args.adaptation_updates,
    )
    manifest = write_manifest(args.root, jobs, protocol)
    print(
        json.dumps(
            {
                "manifest": str(args.root / "MANIFEST.json"),
                "manifest_hash": manifest["manifest_hash"],
                "jobs": len(jobs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
