"""Train-time diagnostics: curves and duration-influence verdicts for fable2 runs.

Under-training and over-training have both burned this project (sol S12's
schedule artifact; C1aa's tail collapse). Every run therefore renders its
trajectories and records a computed verdict on whether training duration is
influencing the headline result — a picture plus a number, not an impression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fixed identity→hue assignment (validated reference categorical palette; color
# follows the entity across every panel, never its rank in one legend).
ARM_COLORS = {
    "normal": "#2a78d6",
    "wrong_passage": "#eb6834",
    "no_exposure": "#1baf7a",
    "memory_lesion": "#eda100",
    "internal_lesion": "#e87ba4",
    "bare_floor": "#555555",
}
DEPTH_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def moving_average(values: list[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("window must be positive")
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def classify_series(
    updates: list[int],
    values: list[float],
    *,
    better: str,
) -> dict:
    """Classify one metric trajectory's relationship to training duration.

    Verdicts: `still_improving_at_stop` (undertraining signal — the best value is
    the final one and the tail is still moving), `degrading_tail` (overtraining
    signal — the best value happened earlier and the tail gave it back), or
    `plateau` (duration is no longer influencing the metric). The epsilon is 2%
    of the series range (floored at 1e-3): trajectories flatter than that are
    noise, not trend.
    """

    if better not in ("lower", "higher"):
        raise ValueError("better must be 'lower' or 'higher'")
    if len(updates) != len(values):
        raise ValueError("updates and values must align")
    if len(values) < 3:
        return {"verdict": "insufficient_evals", "count": len(values)}
    signed = [-v for v in values] if better == "lower" else list(values)
    epsilon = max(1e-3, 0.02 * (max(signed) - min(signed)))
    best_index = max(range(len(signed)), key=signed.__getitem__)
    final_gain = signed[-1] - signed[-2]
    report = {
        "best_update": updates[best_index],
        "best_value": values[best_index],
        "final_update": updates[-1],
        "final_value": values[-1],
        "epsilon": epsilon,
    }
    if best_index == len(signed) - 1 and final_gain > epsilon:
        report["verdict"] = "still_improving_at_stop"
    elif signed[best_index] - signed[-1] > epsilon:
        report["verdict"] = "degrading_tail"
    else:
        plateau_index = next(
            i for i in range(len(signed)) if signed[best_index] - signed[i] <= epsilon
        )
        report["verdict"] = "plateau"
        report["plateau_since_update"] = updates[plateau_index]
    return report


# Mode-line runs use "wrong_mode" for the counterfactual arm; the G-line uses
# "wrong_passage". Same role in every panel and verdict.
ARM_ALIASES = {"wrong_passage": ("wrong_passage", "wrong_mode")}


def _arm_block(evaluation_split: dict, arm: str) -> dict:
    for name in ARM_ALIASES.get(arm, (arm,)):
        if name in evaluation_split:
            return evaluation_split[name]
    raise KeyError(f"arm {arm!r} missing from evaluation block")


def _eval_series(result: dict, split: str, arm: str) -> tuple[list[int], list[float]]:
    updates, values = [], []
    for evaluation in result["evaluations"]:
        updates.append(evaluation["update"])
        values.append(_arm_block(evaluation[split], arm)["mean_loss"])
    return updates, values


def compute_trends(result: dict) -> dict:
    """Pure trend computation over a fable2 result payload (no rendering)."""

    telemetry = result["telemetry"]
    if not telemetry or not result["evaluations"]:
        raise ValueError("result carries no telemetry or evaluations to analyze")
    train_updates = [row["update"] for row in telemetry]
    train_loss = moving_average([row["loss"] for row in telemetry], 11)

    trends = {
        "train_loss_smoothed": classify_series(
            train_updates, train_loss, better="lower"
        )
    }
    for split in ("development", "heldout"):
        updates, normal = _eval_series(result, split, "normal")
        trends[f"{split}_normal_loss"] = classify_series(
            updates, normal, better="lower"
        )
        for arm in ("wrong_passage", "bare_floor", "no_exposure"):
            _, control = _eval_series(result, split, arm)
            margin = [c - n for c, n in zip(control, normal)]
            trends[f"{split}_margin_vs_{arm}"] = classify_series(
                updates, margin, better="higher"
            )

    # Silence-collapse flag (the C1x failure): per-depth control RMS that peaked
    # and then fell below 10% of its peak means an injection site went quiet.
    depth_count = len(telemetry[0]["per_depth_control_rms"])
    collapsed = []
    for depth_index in range(depth_count):
        series = [row["per_depth_control_rms"][depth_index] for row in telemetry]
        peak = max(series)
        if peak > 0 and series[-1] < 0.1 * peak:
            collapsed.append(depth_index)
    trends["silenced_depth_indices"] = collapsed
    return trends


def render_curves(result: dict, trends: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry = result["telemetry"]
    train_updates = [row["update"] for row in telemetry]
    depths = result["depths"]
    written = []

    def style(axis, title, ylabel):
        axis.set_title(title, fontsize=10, loc="left")
        axis.set_xlabel("update", fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.tick_params(labelsize=8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    raw = [row["loss"] for row in telemetry]
    axes[0, 0].plot(train_updates, raw, color="#2a78d6", alpha=0.25, linewidth=1.0)
    axes[0, 0].plot(
        train_updates, moving_average(raw, 11), color="#2a78d6", linewidth=1.8,
        label="paired loss (ma-11)",
    )
    axes[0, 0].legend(fontsize=8, frameon=False)
    style(
        axes[0, 0],
        f"train paired loss — {trends['train_loss_smoothed'].get('verdict', 'n/a')}",
        "loss",
    )
    axes[0, 1].plot(
        train_updates,
        moving_average([row["advantage_vs_wrong_passage"] for row in telemetry], 11),
        color=ARM_COLORS["wrong_passage"], linewidth=1.8, label="vs wrong passage",
    )
    axes[0, 1].plot(
        train_updates,
        moving_average([row["advantage_vs_no_exposure"] for row in telemetry], 11),
        color=ARM_COLORS["no_exposure"], linewidth=1.8, label="vs no exposure",
    )
    axes[0, 1].axhline(0.0, color="#555555", linewidth=0.8, linestyle=":")
    axes[0, 1].legend(fontsize=8, frameon=False)
    style(axes[0, 1], "train advantages (ma-11)", "log-prob advantage")
    for depth_index, depth in enumerate(depths):
        axes[1, 0].plot(
            train_updates,
            [row["per_depth_control_rms"][depth_index] for row in telemetry],
            color=DEPTH_COLORS[depth_index % len(DEPTH_COLORS)],
            linewidth=1.8, label=f"depth {depth}",
        )
    axes[1, 0].legend(fontsize=8, frameon=False)
    style(axes[1, 0], "control RMS by injection depth", "RMS")
    axes[1, 1].plot(
        train_updates, [row["grad_norm"] for row in telemetry],
        color="#4a3aa7", linewidth=1.2,
    )
    axes[1, 1].set_yscale("log")
    style(axes[1, 1], "gradient norm", "norm (log)")
    figure.tight_layout()
    path = out_dir / "train-dynamics.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for column, split in enumerate(("development", "heldout")):
        axis = axes[0, column]
        for arm, color in ARM_COLORS.items():
            updates, values = _eval_series(result, split, arm)
            axis.plot(
                updates, values, color=color, linewidth=1.8,
                linestyle=":" if arm == "bare_floor" else "-",
                marker="o", markersize=3, label=arm,
            )
        if column == 0:
            axis.legend(fontsize=7, frameon=False)
        verdict = trends[f"{split}_normal_loss"].get("verdict", "n/a")
        style(axis, f"{split}: mean loss by arm — normal {verdict}", "mean loss")
    for column, split in enumerate(("development", "heldout")):
        axis = axes[1, column]
        updates, normal = _eval_series(result, split, "normal")
        for arm in ("wrong_passage", "bare_floor", "no_exposure"):
            _, control = _eval_series(result, split, arm)
            axis.plot(
                updates, [c - n for c, n in zip(control, normal)],
                color=ARM_COLORS[arm], linewidth=1.8, marker="o", markersize=3,
                label=f"vs {arm}",
            )
        for depth_index, depth in enumerate(depths):
            lesion = [
                evaluation[split]["depth_lesions"][str(depth)]["mean_loss"]
                for evaluation in result["evaluations"]
            ]
            axis.plot(
                updates, [l - n for l, n in zip(lesion, normal)],
                color=DEPTH_COLORS[depth_index % len(DEPTH_COLORS)],
                linewidth=1.0, linestyle="--", marker=".", markersize=3,
                label=f"depth {depth} lesion",
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle=":")
        if column == 0:
            axis.legend(fontsize=7, frameon=False, ncol=2)
        style(
            axis,
            f"{split}: margin over normal (positive = normal better)",
            "control − normal loss",
        )
    figure.tight_layout()
    path = out_dir / "eval-curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    trends = compute_trends(result)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trend.json").write_text(json.dumps(trends, indent=2))
    written = render_curves(result, trends, out_dir)
    for key, value in trends.items():
        if isinstance(value, dict) and "verdict" in value:
            print(f"{key}: {value['verdict']} (best@{value.get('best_update')})")
        elif key == "silenced_depth_indices" and value:
            print(f"WARNING silence collapse at depth indices {value}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
