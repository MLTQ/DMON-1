"""Build a guarded S0 comparison report from completed SOL benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunResult:
    name: str
    path: Path
    model: str
    parameters: int
    updates: int
    best_bpc: float
    final_bpc: float
    summary: dict[str, Any]


def load_run(name: str, path: str | Path) -> RunResult:
    """Load a completed run and reject missing or invalid headline metrics."""

    root = Path(path)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"{name}: missing summary.json; run is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation = summary.get("evaluation", {})
    if summary.get("model") == "sol":
        persistent = evaluation.get("persistent", {})
        final_bpc = float(persistent.get("bits_per_character", math.nan))
    else:
        final_bpc = float(evaluation.get("bits_per_character", math.nan))
    best_bpc = float(summary.get("best_bpc", math.nan))
    parameters = int(summary.get("parameters", 0))
    updates = int(summary.get("updates", 0))
    if parameters < 1:
        raise ValueError(f"{name}: parameter count is missing or zero")
    if updates < 1:
        raise ValueError(f"{name}: update count is missing or zero")
    if not math.isfinite(best_bpc) or not math.isfinite(final_bpc):
        raise ValueError(f"{name}: BPC is missing or non-finite")
    return RunResult(
        name=name,
        path=root,
        model=str(summary["model"]),
        parameters=parameters,
        updates=updates,
        best_bpc=best_bpc,
        final_bpc=final_bpc,
        summary=summary,
    )


def compare_runs(
    runs: list[RunResult],
    *,
    parameter_tolerance: float = 0.10,
    update_tolerance: int = 0,
) -> dict[str, Any]:
    """Compare completed runs only when budgets are genuinely commensurate."""

    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    reference = next((run for run in runs if run.model == "sol"), runs[0])
    rows: list[dict[str, Any]] = []
    for run in runs:
        parameter_ratio = run.parameters / reference.parameters
        if abs(parameter_ratio - 1.0) > parameter_tolerance:
            raise ValueError(
                f"{run.name}: parameter ratio {parameter_ratio:.3f} exceeds "
                f"tolerance {parameter_tolerance:.3f}"
            )
        if abs(run.updates - reference.updates) > update_tolerance:
            raise ValueError(
                f"{run.name}: updates {run.updates} do not match "
                f"reference {reference.updates}"
            )
        row: dict[str, Any] = {
            "name": run.name,
            "model": run.model,
            "parameters": run.parameters,
            "parameter_ratio": parameter_ratio,
            "updates": run.updates,
            "best_bpc": run.best_bpc,
            "final_bpc": run.final_bpc,
            "delta_from_reference": run.final_bpc - reference.final_bpc,
        }
        if run.model == "sol":
            evaluation = run.summary["evaluation"]
            persistent = float(
                evaluation["persistent"]["bits_per_character"]
            )
            reset = float(
                evaluation["reset_each_token"]["bits_per_character"]
            )
            shuffled = float(
                evaluation["shuffled_cells"]["bits_per_character"]
            )
            row["reset_penalty_bpc"] = reset - persistent
            row["shuffle_penalty_bpc"] = shuffled - persistent
        rows.append(row)
    winner = min(rows, key=lambda row: row["final_bpc"])
    return {
        "reference": reference.name,
        "winner": winner["name"],
        "winner_bpc": winner["final_bpc"],
        "rows": rows,
    }


def markdown_report(comparison: dict[str, Any]) -> str:
    """Render the guarded comparison as a compact Markdown table."""

    lines = [
        "# SOL S0 comparison",
        "",
        "| Run | Model | Params | Ratio | Updates | Best BPC | Final BPC | Reset Δ | Shuffle Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        reset = row.get("reset_penalty_bpc")
        shuffle = row.get("shuffle_penalty_bpc")
        lines.append(
            "| {name} | {model} | {parameters:,} | {parameter_ratio:.3f} | "
            "{updates:,} | {best_bpc:.3f} | {final_bpc:.3f} | {reset} | "
            "{shuffle} |".format(
                **row,
                reset=f"{reset:+.3f}" if reset is not None else "—",
                shuffle=f"{shuffle:+.3f}" if shuffle is not None else "—",
            )
        )
    lines.extend(
        [
            "",
            f"Lowest final held-out BPC: **{comparison['winner']}** "
            f"at **{comparison['winner_bpc']:.3f}**.",
            "",
            "Parameter and update budgets were validated before this table was rendered.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Completed benchmark run; repeat for each comparison",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    runs: list[RunResult] = []
    for specification in args.run:
        if "=" not in specification:
            parser.error("--run must use NAME=PATH")
        name, path = specification.split("=", 1)
        runs.append(load_run(name, path))
    comparison = compare_runs(
        runs, parameter_tolerance=args.parameter_tolerance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown_report(comparison), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
