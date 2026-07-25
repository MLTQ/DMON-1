"""The null model: what the physics achieves with no spatial differentiation at all.

Both levers are held at fixed constants everywhere, the rule is inert, and nothing is
learned. Grid-searching those two constants answers two questions at once:

**1. Lever authority.** How much of the outcome do effort and conductance actually
command? If the best and worst constant policies differ by little, the levers are inert
and *no cell architecture can help* — a GRU or an attention head would compute better
values for controls that do not steer. If they differ a lot, the levers are powerful and
a rule that fails to beat constant is failing at computation, not affordance.

**2. The honest baseline.** Any claim that the learned rule does work must beat the
*best* constant policy. A zero-weight rule sitting at the init gate biases is one
arbitrary point in this space, not a baseline — and using it as one flattered the
network. Note that the M0 pass condition as written ("move the sources, get a different
morphology") is satisfied by this null model with no learning anywhere, which is why
this file exists.

    python -m dmon.baseline --geom west --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .substrate import Substrate, SubstrateConfig, descriptors_per_sample, make_sources


@torch.no_grad()
def run_constant(
    sub: Substrate,
    geom: str,
    effort: float,
    conduct: float,
    grid: int,
    steps: int,
    reps: int = 4,
    dense: bool = False,
    device: str = "cpu",
    dense_e: float | None = None,
) -> dict[str, float]:
    dev = torch.device(device)
    x, r = sub.seed(reps, grid, dev)
    if dense:
        # Every cell alive, but only just — a few multiples of the death threshold, not
        # a full tank. Giving each cell `seed_energy` makes the starvation timescale
        # ~270 steps, so across a 64-step horizon nothing dies and every policy scores
        # a full grid: the measurement reports 1.0x authority for reasons that have
        # nothing to do with the levers.
        x[:, sub.E] = dense_e or (3.0 * sub.cfg.e_death)
    src = make_sources(geom, reps, grid, dev, spread=sub.cfg.spread_end)
    for _ in range(steps):
        x, r = sub.step(x, r, src, gates=(effort, conduct))
    d = descriptors_per_sample(x, sub.cfg)
    return {k: v.mean().item() for k, v in d.items()}


def sweep_levers(
    geom: str = "west",
    grid: int = 64,
    steps: int = 64,
    reps: int = 4,
    n: int = 6,
    dense: bool = False,
    device: str = "cpu",
    cfg: SubstrateConfig | None = None,
    dense_e: float | None = None,
) -> dict:
    cfg = cfg or SubstrateConfig()
    sub = Substrate(cfg).to(device)
    sub.silence_rule()  # inert rule; physics only

    vals = [i / (n - 1) for i in range(n)]
    rows = []
    for e in vals:
        for c in vals:
            d = run_constant(sub, geom, e, c, grid, steps, reps, dense, device, dense_e)
            rows.append({"effort": e, "conduct": c, **d})

    masses = [r_["mass"] for r_ in rows]
    best = max(rows, key=lambda r_: r_["mass"])
    worst = min(rows, key=lambda r_: r_["mass"])
    span = (best["mass"] / worst["mass"]) if worst["mass"] > 0 else float("inf")

    return {
        "geom": geom,
        "grid": grid,
        "steps": steps,
        "dense": dense,
        "rows": rows,
        "best": best,
        "worst": worst,
        "mass_span": span,
        "mass_mean": sum(masses) / len(masses),
    }


def report(res: dict) -> str:
    b, w = res["best"], res["worst"]
    lines = [
        f"best constant policy : effort={b['effort']:.2f} conduct={b['conduct']:.2f} "
        f"-> mass {b['mass']:.1f} (dim {b['box_dim']:.2f})",
        f"worst                : effort={w['effort']:.2f} conduct={w['conduct']:.2f} "
        f"-> mass {w['mass']:.1f}",
    ]
    span = res["mass_span"]
    if span == float("inf") or span > 5:
        lines.append(
            f"LEVER AUTHORITY: HIGH ({span:.1f}x span). The controls steer the outcome, "
            "so a rule that cannot beat constant is failing at computation, not "
            "affordance. Richer cells are the right response."
        )
    elif span > 2:
        lines.append(
            f"LEVER AUTHORITY: MODERATE ({span:.1f}x span). Some headroom, but check "
            "whether the best policy is a corner of the space — a lever that only "
            "matters at its extreme is nearly a constant."
        )
    else:
        lines.append(
            f"LEVER AUTHORITY: LOW ({span:.1f}x span). The levers barely move the "
            "outcome. No cell architecture fixes this: a GRU would compute better "
            "values for controls that do not steer. Add a third lever."
        )
    lines.append(
        f"\nBaseline to beat: mass {b['mass']:.1f}. Any learned rule scoring below this "
        "is worse than having no network at all."
    )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--geom", default="west")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--n", type=int, default=6, help="grid resolution per lever")
    p.add_argument("--dense", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    for dense in ([True, False] if not a.dense else [True]):
        res = sweep_levers(a.geom, a.grid, a.steps, a.reps, a.n, dense, a.device)
        print(f"\n=== lever authority: geom={a.geom} init={'dense' if dense else 'seed'} ===")
        print(report(res))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
