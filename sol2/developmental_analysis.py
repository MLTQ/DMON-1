"""Aggregate frozen S1-P4 anchor and endogenous-development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCHES = ("plastic", "uniform_anchor", "measured_anchor", "developmental")
GATE_EPSILON = 1e-12


def _accuracy(payload: dict, organ: str) -> float:
    final = payload["summary"]["final"]
    if organ == "A":
        return float(final["a_fixed"]["answer_accuracy"])
    return float(
        final["b_by_length"][str(payload["protocol"]["max_steps"])][
            "answer_accuracy"
        ]
    )


def analyze(root: Path) -> dict:
    payloads = {
        branch: json.loads((root / branch / "metrics.json").read_text())
        for branch in BRANCHES
    }
    rows = {}
    for branch, payload in payloads.items():
        a = _accuracy(payload, "A")
        b = _accuracy(payload, "B")
        pressure = [float(row["pressure"]["pressure"]) for row in payload["evaluations"]]
        growth = payload["summary"]["growth_lesions"]
        row = {
            "a_accuracy": a,
            "b_accuracy": b,
            "min_ab_accuracy": min(a, b),
            "rejected_updates": int(payload["rejected_updates"]),
            "a_organ_unchanged": bool(payload["integrity"]["a_organ_unchanged"]),
            "pressure_mean": sum(pressure) / len(pressure),
            "pressure_max": max(pressure),
            "growth_events": len(payload["growth_events"]),
            "grown_cells": sum(len(event["cells"]) for event in payload["growth_events"]),
            "growth_b_freeze_penalty": None,
            "growth_b_adapter_penalty": None,
            "growth_a_freeze_improvement": None,
        }
        if growth is not None:
            row["growth_b_freeze_penalty"] = float(
                growth["B"]["normal"]["answer_accuracy"]
                - growth["B"]["freeze_grown_cells"]["answer_accuracy"]
            )
            row["growth_b_adapter_penalty"] = float(
                growth["B"]["normal"]["answer_accuracy"]
                - growth["B"]["zero_grown_adapters"]["answer_accuracy"]
            )
            row["growth_a_freeze_improvement"] = float(
                growth["A"]["freeze_grown_cells"]["answer_accuracy"]
                - growth["A"]["normal"]["answer_accuracy"]
            )
        rows[branch] = row

    measured_gain = (
        rows["measured_anchor"]["min_ab_accuracy"]
        - rows["uniform_anchor"]["min_ab_accuracy"]
    )
    development_gain = (
        rows["developmental"]["min_ab_accuracy"]
        - rows["measured_anchor"]["min_ab_accuracy"]
    )
    developmental = rows["developmental"]
    anchored = ("uniform_anchor", "measured_anchor", "developmental")
    gates = {
        "anchored_arm_simultaneous_ab_80": any(
            rows[branch]["a_accuracy"] >= 0.80
            and rows[branch]["b_accuracy"] >= 0.80
            for branch in anchored
        ),
        "measured_beats_uniform_by_10pp": measured_gain >= 0.10 - GATE_EPSILON,
        "development_beats_measured_by_10pp": development_gain
        >= 0.10 - GATE_EPSILON,
        "development_triggered": developmental["growth_events"] > 0,
        "grown_cells_causal_for_b": (
            developmental["growth_b_freeze_penalty"] is not None
            and developmental["growth_b_freeze_penalty"]
            >= 0.10 - GATE_EPSILON
        ),
        "grown_cells_do_not_merely_harm_a": (
            developmental["growth_a_freeze_improvement"] is not None
            and developmental["growth_a_freeze_improvement"] <= 0.05
        ),
        "stable_and_integrity_clean": all(
            row["rejected_updates"] == 0 and row["a_organ_unchanged"]
            for row in rows.values()
        ),
    }
    return {
        "branches": rows,
        "contrasts": {
            "measured_minus_uniform_min_ab": measured_gain,
            "developmental_minus_measured_min_ab": development_gain,
        },
        "gates": gates,
        "capability_success": gates["anchored_arm_simultaneous_ab_80"],
        "anchor_attribution_success": gates["measured_beats_uniform_by_10pp"],
        "development_success": all(
            gates[name]
            for name in (
                "development_beats_measured_by_10pp",
                "development_triggered",
                "grown_cells_causal_for_b",
                "grown_cells_do_not_merely_harm_a",
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate S1-P4 result arms")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    rendered = json.dumps(result, indent=2)
    if args.out is not None:
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
