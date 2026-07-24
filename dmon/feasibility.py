"""Can this ecology support a body at all? Answered in seconds, with no training.

`ARCHITECTURE.md` §4 states three design equations. Every one of them was violated by
the original defaults, and nobody noticed until a 40-minute GPU sweep came back with six
identical rows — a knob swept inside a regime where nothing could live.

This module makes that check cheap enough that there is no excuse for skipping it. It
simulates the *field alone*, with no creature and no rule, and reports what any creature
would have to work with.

    python -m dmon.feasibility --geom west --grid 64 --steps 64

Run it before spending GPU time. It costs about a second.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch

from .substrate import Substrate, SubstrateConfig, make_sources


@torch.no_grad()
def analyse(
    cfg: SubstrateConfig,
    geom: str = "west",
    grid: int = 64,
    steps: int = 64,
    seed_yx: tuple[int, int] | None = None,
    spread: float = 1.0,
) -> dict:
    """Field-only simulation. Returns what the ecology offers, independent of any rule."""
    sub = Substrate(cfg)
    seed_yx = seed_yx or (grid // 2, grid // 2)
    src = make_sources(geom, 1, grid, normalize=True, spread=spread)

    r = torch.zeros(1, 1, grid, grid)
    emitted_requested = 0.0
    emitted_actual = 0.0
    for _ in range(steps):
        r = r + cfg.field_diffusion * sub._laplace(r) - cfg.field_decay * r
        before = r.sum().item()
        r = r + cfg.source_rate * src
        emitted_requested += cfg.source_rate * src.sum().item()
        r = r.clamp(0.0, cfg.field_cap)
        emitted_actual += r.sum().item() - before

    # cost of merely being a cell that tries to eat
    cell_cost = cfg.maintenance + cfg.effort_cost
    r_star = cell_cost / cfg.uptake_rate

    inflow = emitted_actual / steps
    n_max = inflow / cell_cost
    usable = int((r >= r_star).sum().item())

    # distance from the seed to the nearest source cell
    ys, xs = torch.nonzero(src[0, 0], as_tuple=True)
    d_src = min(
        math.hypot(y.item() - seed_yx[0], x.item() - seed_yx[1]) for y, x in zip(ys, xs)
    )
    d_diff = 2 * math.sqrt(cfg.field_diffusion * steps)

    return {
        "r_star": r_star,
        "r_at_seed": r[0, 0, seed_yx[0], seed_yx[1]].item(),
        "r_max": r.max().item(),
        "inflow_per_step": inflow,
        "throttled_fraction": 1.0 - emitted_actual / max(emitted_requested, 1e-12),
        "n_max": n_max,
        "usable_cells": usable,
        "seed_to_source": d_src,
        "diffusive_reach": d_diff,
        "light_cone_ok": cfg.light_cone_ok(grid, steps),
    }


def verdict(a: dict, want_mass: int = 300) -> tuple[bool, list[str]]:
    """Each equation gets its own line, so a failure names itself."""
    out, ok = [], True

    if a["n_max"] < want_mass:
        ok = False
        out.append(
            f"SUPPLY  fail — inflow {a['inflow_per_step']:.3f}/step supports at most "
            f"{a['n_max']:.0f} cells at full effort; wanted ~{want_mass}. "
            "Raise source_rate, and field_cap if throttled."
        )
    else:
        out.append(f"SUPPLY  ok   — supports up to {a['n_max']:.0f} cells.")

    if a["throttled_fraction"] > 0.05:
        out.append(
            f"        note — {a['throttled_fraction']:.0%} of requested emission is "
            "discarded by the field_cap clamp. Raising source_rate further will do "
            "nothing until field_cap rises."
        )

    if a["seed_to_source"] > a["diffusive_reach"]:
        ok = False
        out.append(
            f"REACH   fail — source is {a['seed_to_source']:.0f} cells away but food "
            f"diffuses only {a['diffusive_reach']:.0f} in {'T'} steps. Food cannot arrive "
            "within a rollout. Move sources closer, lengthen the rollout, or use a "
            "distance curriculum."
        )
    else:
        out.append(
            f"REACH   ok   — source {a['seed_to_source']:.0f} cells away, diffusive "
            f"reach {a['diffusive_reach']:.0f}."
        )

    if a["r_at_seed"] < a["r_star"]:
        ok = False
        out.append(
            f"CONCEN  fail — r at seed {a['r_at_seed']:.5f} < break-even "
            f"{a['r_star']:.5f}. A cell there loses energy by trying to eat."
        )
    else:
        out.append(
            f"CONCEN  ok   — r at seed {a['r_at_seed']:.4f} vs break-even "
            f"{a['r_star']:.4f} ({a['usable_cells']} cells above it)."
        )

    if not a["light_cone_ok"]:
        ok = False
        out.append("LIGHT   fail — steps < grid; the far side is causally unreachable.")

    return ok, out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--geom", default="west")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--want-mass", type=int, default=300)
    p.add_argument("--spread", type=float, default=1.0)
    p.add_argument("--source-rate", type=float, default=None)
    p.add_argument("--field-cap", type=float, default=None)
    p.add_argument("--field-diffusion", type=float, default=None)
    a = p.parse_args()

    cfg = SubstrateConfig()
    over = {}
    if a.source_rate is not None:
        over["source_rate"] = a.source_rate
    if a.field_cap is not None:
        over["field_cap"] = a.field_cap
    if a.field_diffusion is not None:
        over["field_diffusion"] = a.field_diffusion
    cfg = replace(cfg, **over)

    res = analyse(cfg, a.geom, a.grid, a.steps, spread=a.spread)
    ok, lines = verdict(res, a.want_mass)
    print(f"\n=== feasibility: geom={a.geom} grid={a.grid} steps={a.steps} spread={a.spread} ===")
    print(f"source_rate={cfg.source_rate} field_cap={cfg.field_cap} "
          f"field_diffusion={cfg.field_diffusion}\n")
    for line in lines:
        print(line)
    print(f"\n{'FEASIBLE' if ok else 'NOT FEASIBLE — do not spend GPU time on this'}\n")


if __name__ == "__main__":
    main()
