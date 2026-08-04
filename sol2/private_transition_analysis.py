"""Aggregate the frozen S1-P3b private-transition reserve gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCHES = ("plastic", "uniform", "consolidated", "shuffled")


def _accuracy(payload: dict, organ: str) -> float:
    final = payload["summary"]["final"]
    if organ == "A":
        return float(final["a_fixed"]["answer_accuracy"])
    return float(
        final["b_by_length"][str(payload["protocol"]["max_steps"])][
            "answer_accuracy"
        ]
    )


def _reserve_penalty(payload: dict, organ: str, lesion: str) -> float | None:
    rows = payload["summary"]["reserve_lesions"][organ]
    if lesion not in rows:
        return None
    return float(rows["normal"]["answer_accuracy"] - rows[lesion]["answer_accuracy"])


def analyze(root: Path) -> dict:
    """Load four matched result arms and evaluate preregistered capability gates."""

    payloads = {
        branch: json.loads((root / branch / "metrics.json").read_text())
        for branch in BRANCHES
    }
    rows = {}
    for branch, payload in payloads.items():
        a = _accuracy(payload, "A")
        b = _accuracy(payload, "B")
        drift = payload["summary"]["final"]["drift"]
        adapter = drift.get("adapter_by_measured_utility")
        rows[branch] = {
            "a_accuracy": a,
            "b_accuracy": b,
            "min_ab_accuracy": min(a, b),
            "rejected_updates": int(payload["rejected_updates"]),
            "a_organ_unchanged": bool(payload["integrity"]["a_organ_unchanged"]),
            "graft_within_limit": bool(payload["growth"]["within_limit"]),
            "graft_a_drop": float(payload["growth"]["a_accuracy_drop"]),
            "adapter_low_high_drift_ratio": (
                adapter["low_mean"] / max(adapter["high_mean"], 1e-12)
                if adapter is not None
                else None
            ),
            "b_zero_low_adapter_penalty": _reserve_penalty(
                payload, "B", "zero_low_utility_adapters"
            ),
            "b_zero_all_adapter_penalty": _reserve_penalty(
                payload, "B", "zero_all_internal_adapters"
            ),
            "b_disable_graft_penalty": _reserve_penalty(
                payload, "B", "disable_graft_edges"
            ),
        }

    protected = ("uniform", "consolidated", "shuffled")
    measured_gain = (
        rows["consolidated"]["min_ab_accuracy"]
        - rows["uniform"]["min_ab_accuracy"]
    )
    shuffled_gain = (
        rows["shuffled"]["min_ab_accuracy"] - rows["uniform"]["min_ab_accuracy"]
    )
    consolidated = rows["consolidated"]
    gates = {
        "protected_arm_simultaneous_ab_80": any(
            rows[branch]["a_accuracy"] >= 0.80
            and rows[branch]["b_accuracy"] >= 0.80
            for branch in protected
        ),
        "measured_beats_uniform_by_10pp": measured_gain >= 0.10,
        "shuffle_removes_half_measured_gain": (
            measured_gain > 0.0 and shuffled_gain <= 0.5 * measured_gain
        ),
        "reserve_adapters_change_more": (
            consolidated["adapter_low_high_drift_ratio"] is not None
            and consolidated["adapter_low_high_drift_ratio"] > 1.0
        ),
        "reserve_adapters_are_causal_for_b": (
            consolidated["b_zero_low_adapter_penalty"] is not None
            and consolidated["b_zero_low_adapter_penalty"] > 0.0
        ),
        "graft_edges_are_causal_for_b": (
            consolidated["b_disable_graft_penalty"] is not None
            and consolidated["b_disable_graft_penalty"] > 0.0
        ),
        "stable_and_integrity_clean": all(
            row["rejected_updates"] == 0
            and row["a_organ_unchanged"]
            and row["graft_within_limit"]
            for row in rows.values()
        ),
    }
    return {
        "branches": rows,
        "contrasts": {
            "consolidated_minus_uniform_min_ab": measured_gain,
            "shuffled_minus_uniform_min_ab": shuffled_gain,
        },
        "gates": gates,
        "capability_success": gates["protected_arm_simultaneous_ab_80"],
        "measured_allocation_success": (
            gates["measured_beats_uniform_by_10pp"]
            and gates["shuffle_removes_half_measured_gain"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate S1-P3b result arms")
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
