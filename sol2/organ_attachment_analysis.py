"""Aggregate matched S1-P2 organ-attachment branches and apply frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .organ_attachment import BRANCHES


def _accuracy(payload: dict, *path: str) -> float:
    value = payload
    for key in path:
        value = value[key]
    return float(value)


def analyze_organ_attachment(root: Path) -> dict:
    branches = {
        branch: json.loads((root / branch / "metrics.json").read_text())
        for branch in BRANCHES
    }
    protocols = [payload["protocol"] for payload in branches.values()]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("organ-attachment branch protocols do not match")
    if any(payload["rejected_updates"] for payload in branches.values()):
        raise ValueError("at least one organ-attachment branch rejected updates")

    control = branches["control"]
    full = branches["full"]
    organ_only = branches["organ_only"]
    scratch = branches["scratch"]
    full_b = _accuracy(full, "summary", "final_selected_by_length", "4", "answer_accuracy")
    scratch_b = _accuracy(
        scratch, "summary", "final_selected_by_length", "4", "answer_accuracy"
    )
    organ_only_b = _accuracy(
        organ_only,
        "summary",
        "final_selected_by_length",
        "4",
        "answer_accuracy",
    )
    control_a = _accuracy(
        control, "summary", "final_selected_by_length", "4", "answer_accuracy"
    )
    returned_a = _accuracy(
        full, "summary", "final_a_sustained", "answer_accuracy"
    )
    early_transfer = _accuracy(full, "summary", "early", "answer_accuracy") - _accuracy(
        scratch, "summary", "early", "answer_accuracy"
    )
    removal = full["summary"]["removal_and_reattachment"]
    recovery_keys = sorted(
        (int(key) for key in removal["b_recovery"]), reverse=True
    )
    final_recovery = removal["b_recovery"][str(recovery_keys[0])]["answer_accuracy"]
    pre_removal = removal["b_before_removal"]["answer_accuracy"]

    lesion_penalties = {}
    normal = full["summary"]["lesions"]["normal"]["answer_accuracy"]
    for name in ("freeze_internal", "freeze_compute", "freeze_relay", "topology_rewired"):
        lesion_penalties[name] = normal - full["summary"]["lesions"][name][
            "answer_accuracy"
        ]

    gates = {
        "b_full_at_least_50pct": full_b >= 0.50,
        "early_transfer_at_least_10pp": early_transfer >= 0.10,
        "a_return_within_15pp_of_control": control_a - returned_a <= 0.15,
        "b_reattach_within_10pp_by_final_recovery": pre_removal - final_recovery <= 0.10,
    }
    return {
        "protocol": protocols[0],
        "results": {
            "control_a_length4": control_a,
            "full_b_length4": full_b,
            "organ_only_b_length4": organ_only_b,
            "scratch_b_length4": scratch_b,
            "full_minus_scratch_early_accuracy": early_transfer,
            "full_a_return_length4": returned_a,
            "a_return_gap_from_control": control_a - returned_a,
            "b_before_removal": pre_removal,
            "b_after_final_recovery": final_recovery,
            "b_recovery_gap": pre_removal - final_recovery,
            "full_b_lesion_penalties": lesion_penalties,
        },
        "integrity": {
            "all_b_branches_preserved_a_organ": all(
                branches[name]["integrity"]["a_organ_unchanged"]
                for name in ("full", "organ_only", "scratch")
            ),
            "organ_only_preserved_substrate": organ_only["integrity"][
                "substrate_unchanged"
            ],
            "zero_rejected_updates": True,
        },
        "gates": gates,
        "strong_success": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze S1-P2 organ attachment")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze_organ_attachment(args.root)
    rendered = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
