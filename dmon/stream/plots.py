"""Figures for experiments. Each one states its claim and its verdict on the figure.

A bpc printed in a log is easy to over-read — this session produced several confident
conclusions from single numbers that a curve would have contradicted at a glance (a
model still falling steeply, called converged; an ordering that reversed with training).
So a figure here is not decoration. It carries the hypothesis in the title and the
answer in the subtitle, and if it cannot state a verdict it says so.

    from dmon.stream.plots import training_curves, ablation_curves, scaling_curve
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
_TRAIN = re.compile(r"\[(\S+)\s+(\d+)\]\s+(?:ep=\s*([\d.]+)\s+)?train_bpc=\s*([\d.]+)")
_EVAL = re.compile(r"\[(\S+)\s+(\d+)\]\s+HELD-OUT bpc=([\d.]+)")


def parse_log(path: str | Path, chars_per_tick: int | None = None) -> dict:
    """Parse a train.py log into {name: {'train': [(x,bpc)], 'eval': [(x,bpc)]}}.

    x is epochs when the log records them, else ticks — the caller is told which via
    the returned 'x_unit', because silently mixing the two across runs with different
    batch sizes would make an unfair comparison look like a fair one.
    """
    runs: dict[str, dict] = {}
    x_unit = "ticks"
    for line in Path(path).read_text().splitlines():
        m = _TRAIN.search(line)
        if m:
            name, tick, ep, bpc = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
            r = runs.setdefault(name, {"train": [], "eval": []})
            if ep is not None:
                x_unit = "epochs"
                r["train"].append((float(ep), bpc))
            elif chars_per_tick:
                x_unit = "epochs"
                r["train"].append((tick * chars_per_tick / 1_003_854, bpc))
            else:
                r["train"].append((tick, bpc))
            continue
        m = _EVAL.search(line)
        if m:
            name, tick, bpc = m.group(1), int(m.group(2)), float(m.group(3))
            r = runs.setdefault(name, {"train": [], "eval": []})
            xs = r["train"]
            x = xs[-1][0] if xs else tick
            r["eval"].append((x, bpc))
    return {"runs": runs, "x_unit": x_unit}


def _finish(fig, ax, claim: str, verdict: str, out: Path, legend=True):
    ax.set_title(claim, fontsize=12, fontweight="bold", pad=14)
    fig.suptitle("")
    ax.text(
        0.5, 1.015, verdict, transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, style="italic", color="#374151", wrap=True,
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=9)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def training_curves(parsed: dict, claim: str, verdict: str, out: str | Path,
                    uniform: float | None = 6.022, floor: float | None = None,
                    zoom: tuple[float, float] | None = None):
    """`zoom` adds a second panel over a sub-range.

    Needed whenever runs have very different lengths: plotting a 23-epoch run and a
    382-epoch run on one axis squeezes the shorter one against the left edge and hides
    exactly the region where they can be compared. The wide view answers "did it
    converge"; the zoom answers "which was better where both ran".
    """
    if zoom:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax2 = None
    for i, (name, r) in enumerate(sorted(parsed["runs"].items())):
        c = PALETTE[i % len(PALETTE)]
        if r["train"]:
            xs, ys = zip(*r["train"])
            ax.plot(xs, ys, color=c, lw=1.6, label=f"{name} (train)")
        if r["eval"]:
            xs, ys = zip(*r["eval"])
            ax.plot(xs, ys, color=c, lw=0, marker="o", ms=6, mfc="white",
                    mew=1.8, label=f"{name} (held-out)")
        if ax2 is not None:
            if r["train"]:
                xs, ys = zip(*r["train"])
                ax2.plot(xs, ys, color=c, lw=1.8)
            if r["eval"]:
                xs, ys = zip(*r["eval"])
                ax2.plot(xs, ys, color=c, lw=0, marker="o", ms=7, mfc="white", mew=2)
    if uniform:
        ax.axhline(uniform, color="#9ca3af", ls=":", lw=1.2)
        ax.text(ax.get_xlim()[1], uniform, " uniform", va="center", fontsize=8,
                color="#6b7280")
    if floor:
        ax.axhline(floor, color="#111827", ls="--", lw=1.0)
    ax.set_xlabel(parsed["x_unit"])
    ax.set_ylabel("bits per character")
    if ax2 is not None:
        ax2.set_xlim(*zoom)
        ax2.set_xlabel(parsed["x_unit"])
        ax2.set_title("where both runs overlap", fontsize=10)
        ax2.grid(alpha=0.25, linewidth=0.6)
        ax2.spines[["top", "right"]].set_visible(False)
    return _finish(fig, ax, claim, verdict, out)


def ablation_curves(series: dict[str, list[tuple[int, float]]], claim: str,
                    verdict: str, out: str | Path, port_radius: int | None = 5):
    """bpc against how much of the lattice is kept. Flat = the periphery is unused."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, pts) in enumerate(series.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], lw=1.8, marker="o", ms=5,
                label=name)
    if port_radius:
        ax.axvline(port_radius, color="#9ca3af", ls=":", lw=1.2)
        ax.text(port_radius, ax.get_ylim()[1], " readout intact →", fontsize=8,
                va="top", color="#6b7280")
    ax.set_xlabel("radius kept (cells beyond this are zeroed every tick)")
    ax.set_ylabel("held-out bits per character")
    return _finish(fig, ax, claim, verdict, out)


def scaling_curve(series: dict[str, list[tuple[int, float]]], claim: str,
                  verdict: str, out: str | Path):
    """bpc against parameter count, log-x. The question capacity ceilings hide."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, pts) in enumerate(series.items()):
        pts = sorted(pts)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], lw=1.8, marker="o", ms=6,
                label=name)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("held-out bits per character")
    return _finish(fig, ax, claim, verdict, out)
