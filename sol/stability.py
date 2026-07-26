"""Completed-run stability measurements for continuous SOL experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_sol_evaluation_history(path: str | Path) -> list[tuple[int, float]]:
    """Read finite SOL validation points, keeping the last row per update."""

    source = Path(path)
    if not source.exists():
        return []
    by_update: dict[int, float] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "evaluation" or row.get("model") != "sol":
            continue
        bpc = float(
            row["ablations"]["persistent"]["bits_per_character"]
        )
        if math.isfinite(bpc):
            by_update[int(row["update"])] = bpc
    return sorted(by_update.items())


def summarize_stability(
    history: list[tuple[int, float]],
    max_regression_bpc: float,
) -> dict[str, Any]:
    """Measure post-best collapse without penalizing ordinary early learning."""

    if max_regression_bpc < 0:
        raise ValueError("max_regression_bpc must be non-negative")
    deduplicated = sorted(dict(history).items())
    if not deduplicated:
        raise ValueError("at least one finite SOL evaluation is required")
    best_update, best_bpc = min(
        deduplicated, key=lambda item: item[1]
    )
    final_update, final_bpc = deduplicated[-1]
    post_best = [
        bpc for update, bpc in deduplicated if update >= best_update
    ]
    worst_post_best = max(post_best)
    final_regression = final_bpc - best_bpc
    worst_regression = worst_post_best - best_bpc
    return {
        "stable": (
            final_regression <= max_regression_bpc
            and worst_regression <= max_regression_bpc
        ),
        "max_regression_bpc": max_regression_bpc,
        "best_update": best_update,
        "best_bpc": best_bpc,
        "final_update": final_update,
        "final_bpc": final_bpc,
        "final_regression_bpc": final_regression,
        "worst_post_best_bpc": worst_post_best,
        "worst_regression_bpc": worst_regression,
        "evaluations": len(deduplicated),
    }


def summarize_exploratory_survival(
    history: list[tuple[int, float]],
    trial_history: list[dict[str, Any]],
    max_regression_bpc: float,
) -> dict[str, Any]:
    """Align each live traffic trial with validation before and after it."""

    if max_regression_bpc < 0:
        raise ValueError("max_regression_bpc must be non-negative")
    evaluations = sorted(dict(history).items())
    trials = [
        dict(trial)
        for trial in trial_history
        if trial.get("mode") == "exploratory_traffic"
        and not trial.get("virtual", False)
    ]
    aligned: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        started = int(trial["started_update"])
        resolved = int(trial["resolved_update"])
        next_started = (
            int(trials[index + 1]["started_update"])
            if index + 1 < len(trials)
            else math.inf
        )
        before = [
            (update, bpc)
            for update, bpc in evaluations
            if update <= started
        ]
        after = [
            (update, bpc)
            for update, bpc in evaluations
            if resolved <= update < next_started
        ]
        row = {
            **trial,
            "baseline_update": None,
            "baseline_bpc": None,
            "post_evaluations": len(after),
            "first_post_update": None,
            "first_post_bpc": None,
            "final_post_update": None,
            "final_post_bpc": None,
            "worst_post_bpc": None,
            "worst_regression_bpc": None,
            "survived": None,
        }
        if before:
            baseline_update, baseline_bpc = before[-1]
            row.update(
                {
                    "baseline_update": baseline_update,
                    "baseline_bpc": baseline_bpc,
                }
            )
        if after:
            first_update, first_bpc = after[0]
            final_update, final_bpc = after[-1]
            worst_post_bpc = max(bpc for _, bpc in after)
            row.update(
                {
                    "first_post_update": first_update,
                    "first_post_bpc": first_bpc,
                    "final_post_update": final_update,
                    "final_post_bpc": final_bpc,
                    "worst_post_bpc": worst_post_bpc,
                }
            )
        if before and after:
            worst_regression = worst_post_bpc - baseline_bpc
            row.update(
                {
                    "worst_regression_bpc": worst_regression,
                    "survived": worst_regression <= max_regression_bpc,
                }
            )
        aligned.append(row)

    evaluated = [trial for trial in aligned if trial["survived"] is not None]
    stable = [trial for trial in evaluated if trial["survived"]]
    advantages = [
        float(trial["decision_advantage"]) for trial in trials
    ]
    return {
        "max_regression_bpc": max_regression_bpc,
        "trials": len(trials),
        "committed": sum(
            trial.get("outcome") == "committed" for trial in trials
        ),
        "rejected": sum(
            trial.get("outcome") == "rejected" for trial in trials
        ),
        "evaluated_trials": len(evaluated),
        "pending_trials": len(aligned) - len(evaluated),
        "survived_trials": len(stable),
        "unstable_trials": len(evaluated) - len(stable),
        "mean_decision_advantage": (
            sum(advantages) / len(advantages) if advantages else None
        ),
        "trial_history": aligned,
    }
