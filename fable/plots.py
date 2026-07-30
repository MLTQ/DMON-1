"""Loss-curve figures for fable runs — claim in the title, verdict in the
subtitle (the dmon/stream convention, kept).

Fixed categorical assignment (validated palette; color follows the entity
across every figure): creature=blue, gru=orange, transformer=aqua,
and in F1 figures born=blue / grown=orange / small=aqua.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK2 = "#5f5e56"
GRID = "#e6e5e0"
COLORS = {"creature": "#2a78d6", "gru": "#eb6834", "transformer": "#1baf7a",
          "born": "#2a78d6", "grown": "#eb6834", "small": "#1baf7a"}
SLOPE_WINDOW = 1500  # late-slope window in updates


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _series(result: dict, which: str):
    if which == "holdout":
        pts = [(e["update"], e["normal"]["bits_per_character"])
               for e in result.get("evals", [])]
    else:
        pts = [(h["update"], h["bpc"]) for h in result.get("history", [])]
    return [p[0] for p in pts], [p[1] for p in pts]


def late_slope(result: dict) -> float | None:
    """Δ held-out BPC across the final SLOPE_WINDOW updates."""
    us, bs = _series(result, "holdout")
    if not us:
        return None
    end = us[-1]
    ref = [b for u, b in zip(us, bs) if u <= end - SLOPE_WINDOW]
    return bs[-1] - ref[-1] if ref else None


def _load(root: Path, arm: str, kind: str) -> dict | None:
    f = root / arm / f"{kind}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _dodge(labels: list[tuple[float, str, str]], min_gap: float) -> list:
    """labels: (y, text, color) → same order with y nudged apart."""
    out = sorted(labels, key=lambda x: x[0])
    for i in range(1, len(out)):
        if out[i][0] - out[i - 1][0] < min_gap:
            out[i] = (out[i - 1][0] + min_gap, out[i][1], out[i][2])
    return out


def plot_f0(root: Path, out: Path) -> dict:
    seeds = [d.name for d in sorted(
        (p for p in root.iterdir()
         if p.is_dir() and (p / "creature.json").exists()),
        key=lambda p: int(p.name.lstrip("s")))]
    seeds = [p.name for p in [root / s for s in seeds]] if seeds and isinstance(seeds[0], str) else seeds
    kinds = ["creature", "gru", "transformer"]
    fig, axes = plt.subplots(2, len(seeds), figsize=(4.6 * len(seeds), 6.4),
                             dpi=150, facecolor=SURFACE, sharex=True)
    if len(seeds) == 1:
        axes = axes.reshape(2, 1)
    slopes: dict[str, list] = {k: [] for k in kinds}
    for col, seed in enumerate(seeds):
        for row, which in enumerate(("holdout", "train")):
            ax = axes[row][col]
            _style(ax)
            end_labels = []
            for kind in kinds:
                r = _load(root, seed, kind)
                if r is None:
                    continue
                us, bs = _series(r, which)
                if not us:
                    continue
                ax.plot(us, bs, color=COLORS[kind], linewidth=1.8,
                        solid_capstyle="round")
                if which == "holdout":
                    sl = late_slope(r)
                    if sl is not None:
                        slopes[kind].append(sl)
                    end_labels.append(
                        (bs[-1], f"{kind[:5]} {bs[-1]:.3f} ({sl:+.3f})",
                         COLORS[kind]))
            ax.set_xlim(0, 8200)
            if which == "holdout":
                for y, text, color in _dodge(end_labels, 0.09):
                    ax.annotate(text, (8200, y), xytext=(3, 0),
                                textcoords="offset points", fontsize=7,
                                color=INK, va="center",
                                annotation_clip=False)
                    ax.plot([8080], [y], marker="o", markersize=3,
                            color=color, clip_on=False)
                ax.axvspan(8000 - SLOPE_WINDOW, 8000, color=GRID, alpha=0.35,
                           zorder=0)
                ax.set_title(seed, fontsize=9, color=INK)
                ax.set_ylim(1.9, 3.4)
            else:
                ax.set_ylim(1.8, 3.2)
                ax.set_xlabel("update", fontsize=8, color=INK2)
            if col == 0:
                ax.set_ylabel("held-out BPC" if which == "holdout"
                              else "train-window BPC", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], color=COLORS[k], linewidth=2, label=k)
               for k in kinds]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8)
    mean_slopes = {k: (sum(v) / len(v) if v else None)
                   for k, v in slopes.items()}
    sub = " · ".join(f"{k} Δlate {mean_slopes[k]:+.3f}" for k in kinds
                     if mean_slopes[k] is not None)
    fig.suptitle("F0 — held-out loss over full training: "
                 "what is left on the table?", fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.945,
             f"shaded = late-slope window (last {SLOPE_WINDOW} updates) · "
             f"mean {sub} — still falling at the horizon: the budget binds, "
             "not the models", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 0.88, 0.93))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "f0_losses.png", facecolor=SURFACE)
    plt.close(fig)
    return mean_slopes


def plot_f1(root: Path, out: Path, grow_at: int = 2000) -> None:
    arms: dict[str, dict[str, dict]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        name, seed = d.name.rsplit("_s", 1)
        r = _load(root, d.name, "creature")
        if r is not None:
            arms.setdefault(seed, {})[name] = r
    if not arms:
        return
    seeds = sorted(arms)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.4 * len(seeds), 3.6),
                             dpi=150, facecolor=SURFACE, squeeze=False)
    for col, seed in enumerate(seeds):
        ax = axes[0][col]
        _style(ax)
        end_labels = []
        for name in ("small", "grown", "born"):
            r = arms[seed].get(name)
            if r is None:
                continue
            us, bs = _series(r, "holdout")
            ax.plot(us, bs, color=COLORS[name], linewidth=1.8)
            end_labels.append((bs[-1], f"{name} {bs[-1]:.3f}", COLORS[name]))
        ax.set_xlim(0, 8200)
        for y, text, color in _dodge(end_labels, 0.09):
            ax.annotate(text, (8200, y), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, color=INK,
                        va="center", annotation_clip=False)
            ax.plot([8080], [y], marker="o", markersize=3, color=color,
                    clip_on=False)
        ax.axvline(grow_at, color=INK2, linewidth=0.8, linestyle=":")
        ax.annotate("graft", (grow_at, ax.get_ylim()[1]), fontsize=7,
                    color=INK2, ha="left", va="top")
        ax.set_title(f"s{seed}", fontsize=9, color=INK)
        ax.set_ylim(1.9, 3.4)
        ax.set_xlabel("update", fontsize=8, color=INK2)
        if col == 0:
            ax.set_ylabel("held-out BPC", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], color=COLORS[k], linewidth=2, label=k)
               for k in ("small", "grown", "born")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8)
    fig.suptitle("F1 — growth: does grown tissue reach born-large?",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 0.93, 0.94))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "f1_losses.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0", default=None)
    ap.add_argument("--f1", default=None)
    ap.add_argument("--out", default="fable/experiments/figures")
    args = ap.parse_args()
    out = Path(args.out)
    if args.f0:
        print("f0 mean late slopes:", plot_f0(Path(args.f0), out))
    if args.f1:
        plot_f1(Path(args.f1), out)
    print("figures in", out)


if __name__ == "__main__":
    main()
