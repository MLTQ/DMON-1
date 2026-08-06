"""Aggregate the frozen S1-P3 consolidation gates across matched branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BRANCHES = ("plastic", "consolidated", "shuffled")


def _accuracy(payload: dict, organ: str) -> float:
    final = payload["summary"]["final"]
    if organ == "A":
        return float(final["a_fixed"]["answer_accuracy"])
    max_steps = str(payload["protocol"]["max_steps"])
    return float(final["b_by_length"][max_steps]["answer_accuracy"])


def _lesion_penalty(payload: dict, organ: str, lesion: str) -> float:
    rows = payload["summary"]["utility_lesions"][organ]
    return float(rows["normal"]["answer_accuracy"] - rows[lesion]["answer_accuracy"])


def _equal_quartile(values: torch.Tensor, utility: torch.Tensor) -> dict:
    count = max(1, values.numel() // 4)
    order = torch.argsort(utility, stable=True)
    return {
        "low_count": count,
        "high_count": count,
        "low_mean": float(values[order[:count]].mean()),
        "high_mean": float(values[order[-count:]].mean()),
    }


def checkpoint_drift(acquisition_path: Path, adaptation_path: Path) -> dict:
    """Recompute equal-count drift quartiles directly from exact checkpoints."""

    acquisition = torch.load(acquisition_path, map_location="cpu", weights_only=True)
    adaptation = torch.load(adaptation_path, map_location="cpu", weights_only=True)
    anchor = acquisition["model"]
    current = adaptation["model"]
    utility = adaptation["utility_profile"]
    expression_cells = current["expression_cells"].long()
    gain_delta = current["cell_gain"] - anchor["cell_gain"]
    bias_delta = current["cell_bias"] - anchor["cell_bias"]
    expression_delta = (gain_delta.square() + bias_delta.square()).mean(-1).sqrt()
    expression_utility = utility["measured_cell"][expression_cells]
    edge_delta = (current["graph.edge_logit"] - anchor["graph.edge_logit"]).abs()
    active = current["graph.active"].bool()
    genome_squared = []
    for name, value in current.items():
        if name.startswith(("graph.query.", "graph.key.", "graph.value.", "tissues.")):
            genome_squared.append((value - anchor[name]).square().flatten())
    return {
        "expression_rms": float(expression_delta.square().mean().sqrt()),
        "expression_by_measured_utility": _equal_quartile(
            expression_delta, expression_utility
        ),
        "active_edge_abs_mean": float(edge_delta[active].mean()),
        "edge_by_measured_utility": _equal_quartile(
            edge_delta[active], utility["measured_edge"][active]
        ),
        "genome_rms": float(torch.cat(genome_squared).mean().sqrt()),
    }


def analyze(root: Path, acquisition_checkpoint: Path | None = None) -> dict:
    payloads = {
        branch: json.loads((root / branch / "metrics.json").read_text())
        for branch in BRANCHES
    }
    rows = {}
    for branch, payload in payloads.items():
        a = _accuracy(payload, "A")
        b = _accuracy(payload, "B")
        drift = (
            checkpoint_drift(
                acquisition_checkpoint, root / branch / "adaptation.pt"
            )
            if acquisition_checkpoint is not None
            else payload["summary"]["final"]["drift"]
        )
        expression = drift["expression_by_measured_utility"]
        edge = drift["edge_by_measured_utility"]
        rows[branch] = {
            "a_accuracy": a,
            "b_accuracy": b,
            "min_ab_accuracy": min(a, b),
            "rejected_updates": int(payload["rejected_updates"]),
            "expression_low_high_drift_ratio": expression["low_mean"]
            / max(expression["high_mean"], 1e-12),
            "edge_low_high_drift_ratio": edge["low_mean"]
            / max(edge["high_mean"], 1e-12),
            "b_high_utility_lesion_penalty": _lesion_penalty(
                payload, "B", "freeze_high_utility"
            ),
            "b_low_utility_lesion_penalty": _lesion_penalty(
                payload, "B", "freeze_low_utility"
            ),
            "a_organ_unchanged": bool(payload["integrity"]["a_organ_unchanged"]),
        }

    plastic_min = rows["plastic"]["min_ab_accuracy"]
    consolidated_min = rows["consolidated"]["min_ab_accuracy"]
    shuffled_min = rows["shuffled"]["min_ab_accuracy"]
    measured_gain = consolidated_min - plastic_min
    shuffled_gain = shuffled_min - plastic_min
    gates = {
        "simultaneous_ab_80": (
            rows["consolidated"]["a_accuracy"] >= 0.80
            and rows["consolidated"]["b_accuracy"] >= 0.80
        ),
        "measured_gain_at_least_20pp": measured_gain >= 0.20,
        "shuffle_removes_half_gain": (
            measured_gain > 0.0 and shuffled_gain <= 0.5 * measured_gain
        ),
        "reserve_changes_more_expression": (
            rows["consolidated"]["expression_low_high_drift_ratio"] > 1.0
        ),
        "reserve_changes_more_edges": (
            rows["consolidated"]["edge_low_high_drift_ratio"] > 1.0
        ),
        "b_reuses_high_utility_more_than_low": (
            rows["consolidated"]["b_high_utility_lesion_penalty"]
            > rows["consolidated"]["b_low_utility_lesion_penalty"]
        ),
        "stable_and_integrity_clean": all(
            row["rejected_updates"] == 0 and row["a_organ_unchanged"]
            for row in rows.values()
        ),
    }
    return {
        "branches": rows,
        "contrasts": {
            "consolidated_minus_plastic_min_ab": measured_gain,
            "shuffled_minus_plastic_min_ab": shuffled_gain,
        },
        "gates": gates,
        "primary_success": all(
            gates[name]
            for name in (
                "simultaneous_ab_80",
                "measured_gain_at_least_20pp",
                "shuffle_removes_half_gain",
                "stable_and_integrity_clean",
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate S1-P3 consolidation branches")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--acquisition-checkpoint", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.acquisition_checkpoint)
    rendered = json.dumps(result, indent=2)
    if args.out is not None:
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
