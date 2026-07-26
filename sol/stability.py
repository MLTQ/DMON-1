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
