"""The M0 verdict: is morphology contingent on ecology?

This replaces the version that lived in `train_m0.py`, which could not fail. It had two
separate defects:

1. It reported between-geometry spread with no baseline to exceed. Any two runs differ
   by *something*, so a bare spread number is always positive and always "looks like"
   contingency.
2. It computed cross-evaluation results and then discarded them. That is the half that
   catches the dangerous failure — a rule that memorised a shape and ignores the field
   looks exactly like a rule that learned to read gradients, and only cross-evaluation
   separates them.

Both are fixed here, and the fix has a cost that should be stated plainly: **the noise
floor must come from independent training runs, not repeated evaluation.** `seed()` is
deterministic — one live cell, empty field — so re-evaluating a single rule samples only
the firing mask and yields a floor far too low to fail against. So this trains
`len(geoms) × len(seeds)` rules, not `len(geoms)`.

Everything is reported in units of that noise floor. A separation of 1.0 means "exactly
as different as two runs of the same experiment", i.e. nothing.

    python -m dmon.contingency --iters 20000 --seeds 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import checkpoint
from .substrate import Substrate, descriptors_per_sample, make_sources
from .train_m0 import train

KEYS = ("mass", "compactness", "gyration", "box_dim")


@torch.no_grad()
def descriptor_vector(sub: Substrate, geom: str, grid: int, steps: int, device: str, reps: int = 8):
    """Mean descriptor vector for one rule under one geometry. Returns a (4,) tensor."""
    dev = torch.device(device)
    x, r = sub.seed(reps, grid, dev)
    src = make_sources(geom, reps, grid, dev)
    x, r, _ = sub.rollout(x, r, src, steps=steps)
    d = descriptors_per_sample(x, sub.cfg)
    return torch.stack([d[k].mean() for k in KEYS]).cpu()


def _pairwise(vs):
    """Mean pairwise Euclidean distance within a list of vectors."""
    if len(vs) < 2:
        return None
    ds = [
        (vs[i] - vs[j]).norm().item()
        for i in range(len(vs))
        for j in range(i + 1, len(vs))
    ]
    return sum(ds) / len(ds)


def effective_dim(vectors) -> dict:
    """Participation ratio of the descriptor point cloud.

    `ARCHITECTURE.md` §M0 requires this: it is what the bridge to M3 rests on. A cloud
    collapsing to ~1 means the ecology admits one morphology with a single axis of
    variation — the body is not steerable and the display must carry all expressivity.
    ~3-4 means the body itself has room to be selected over.
    """
    m = torch.stack(vectors)
    m = m - m.mean(0, keepdim=True)
    s = torch.linalg.svdvals(m.double())
    p = s**2
    if p.sum() <= 0:
        return {"participation_ratio": 0.0, "singular_values": s.tolist()}
    pr = (p.sum() ** 2 / (p**2).sum()).item()
    return {"participation_ratio": pr, "singular_values": s.tolist()}


def run(
    geoms=("center", "west", "poles", "corners"),
    seeds=(0, 1, 2, 3, 4),
    iters: int = 20000,
    grid: int = 64,
    steps: int = 64,
    batch: int = 32,
    lr: float = 2e-3,
    device: str = "cpu",
    reps: int = 8,
    outdir: Path = Path("runs/contingency"),
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- train len(geoms) x len(seeds) rules -------------------------------------
    vectors: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for g in geoms:
        for s in seeds:
            print(f"\n=== training geom={g} seed={s} ===")
            sub, _ = train(
                g, iters, grid, steps, batch, lr, device,
                log=max(1, iters // 10),
                ckpt=outdir / f"{g}_s{s}.pt",
                ckpt_every=max(1, iters // 2),
                seed=s,
            )
            # evaluate this rule under every geometry, not just its own
            vectors[(g, s)] = {h: descriptor_vector(sub, h, grid, steps, device, reps) for h in geoms}

    # --- noise floor: same geometry, different training seeds ---------------------
    # This is the baseline everything else is measured against. It must come from
    # retraining; repeated evaluation of one rule only samples the firing mask.
    within_raw = {g: _pairwise([vectors[(g, s)][g] for s in seeds]) for g in geoms}
    per_dim = torch.stack(
        [torch.stack([vectors[(g, s)][g] for s in seeds]).std(0) for g in geoms]
    ).mean(0)
    sigma = per_dim.clamp(min=1e-6)  # per-descriptor noise scale

    def z(v):
        return v / sigma

    within = {g: _pairwise([z(vectors[(g, s)][g]) for s in seeds]) for g in geoms}
    noise_floor = sum(v for v in within.values() if v is not None) / max(1, len(within))

    # --- signal: different geometries, each rule on its own field -----------------
    means = {g: torch.stack([z(vectors[(g, s)][g]) for s in seeds]).mean(0) for g in geoms}
    between = _pairwise(list(means.values()))

    # --- memoriser test: does a rule change body when moved to a new field? -------
    # A memoriser drags its shape along; a responder grows something else. Measured in
    # the same noise units, so <=1 means "indistinguishable from doing nothing".
    cross = {}
    for g in geoms:
        ds = []
        for s in seeds:
            own = z(vectors[(g, s)][g])
            for h in geoms:
                if h != g:
                    ds.append((own - z(vectors[(g, s)][h])).norm().item())
        cross[g] = sum(ds) / len(ds)
    responsiveness = sum(cross.values()) / len(cross)

    cloud = effective_dim([v for per in vectors.values() for v in per.values()])

    # Degeneracy guards. Every ratio above is a quotient by the noise floor, so a
    # collapsed floor manufactures enormous confidence out of nothing — four identically
    # dead creatures separate "infinitely well". HANDOFF.md: below ~50 cells the
    # descriptors are noise and box_dim in particular means nothing.
    self_mass = torch.stack([vectors[(g, s)][g][0] for g in geoms for s in seeds])
    degenerate = {
        "median_self_mass": self_mass.median().item(),
        "mass_too_small": bool(self_mass.median().item() < 50),
        "flat_descriptors": [
            k for k, v in zip(KEYS, sigma.tolist()) if v <= 1e-5
        ],
    }

    report = {
        "degeneracy": degenerate,
        "geoms": list(geoms),
        "seeds": list(seeds),
        "iters": iters,
        "grid": grid,
        "steps": steps,
        "sigma_per_descriptor": dict(zip(KEYS, sigma.tolist())),
        "noise_floor": noise_floor,
        "within_geometry": within,
        "within_geometry_raw": within_raw,
        "between_geometry": between,
        "separation_ratio": (between / noise_floor) if noise_floor else None,
        "responsiveness": responsiveness,
        "responsiveness_per_geom": cross,
        "morphospace": cloud,
        "vectors": {
            f"{g}_s{s}": {h: v.tolist() for h, v in per.items()}
            for (g, s), per in vectors.items()
        },
    }
    report["verdict"] = verdict(report)
    (outdir / "contingency.json").write_text(json.dumps(report, indent=2))
    return report


def verdict(rep: dict) -> str:
    sep = rep.get("separation_ratio")
    resp = rep.get("responsiveness")
    pr = rep["morphospace"]["participation_ratio"]
    deg = rep.get("degeneracy", {})

    if sep is None:
        return "INCONCLUSIVE — no noise floor. Need >=2 seeds per geometry."

    # Checked before anything else: every number below is divided by the noise floor,
    # so a degenerate floor turns nothing into a confident pass.
    if deg.get("mass_too_small"):
        return (
            f"INCONCLUSIVE — median body is {deg['median_self_mass']:.0f} cells. Below "
            "~50 the descriptors are noise and box_dim is meaningless (HANDOFF.md), so "
            "the separation and responsiveness ratios are quotients by an "
            "artificially collapsed noise floor. This is the 'everything starved' "
            "failure, not a result. Seed nearer a source, or walk the sources outward."
        )
    if deg.get("flat_descriptors"):
        return (
            f"INCONCLUSIVE — descriptors {deg['flat_descriptors']} are identical across "
            "independently trained rules. Either training collapsed to the same "
            "degenerate solution or the runs are not actually independent; check that "
            "--seeds varies the torch seed."
        )

    lines = []
    if sep < 1.5:
        lines.append(
            f"FAIL (not contingent) — between-geometry separation is {sep:.2f}x the "
            "seed-noise floor. Different ecologies are producing the same body."
        )
    elif resp < 1.5:
        lines.append(
            f"FAIL (memorised) — geometries separate ({sep:.2f}x noise) but a rule "
            f"moved to a new field barely changes ({resp:.2f}x noise). It learned a "
            "shape, not how to read a gradient. This is the failure that looks like "
            "success; see HANDOFF.md."
        )
    else:
        lines.append(
            f"PASS — separation {sep:.2f}x noise, responsiveness {resp:.2f}x noise. "
            "Morphology is contingent on ecology and rules react to a field they were "
            "not trained in."
        )
    lines.append(
        f"morphospace participation ratio {pr:.2f} of {len(KEYS)} — "
        + (
            "one axis of variation; the body is barely steerable and a display would "
            "have to carry all expressivity (ARCHITECTURE.md §M3)."
            if pr < 1.5
            else "the body itself has room to be selected over."
        )
    )
    return " ".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--geoms", nargs="+", default=["center", "west", "poles", "corners"])
    p.add_argument("--seeds", type=int, default=5, help="independent training runs per geometry")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--outdir", type=Path, default=Path("runs/contingency"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    n = len(a.geoms) * a.seeds
    print(f"{n} training runs ({len(a.geoms)} geometries x {a.seeds} seeds).")

    rep = run(
        a.geoms, tuple(range(a.seeds)), a.iters, a.grid, a.steps,
        a.batch, a.lr, a.device, a.reps, a.outdir,
    )

    print("\n=== contingency ===")
    print(f"noise floor (within-geometry, across training seeds): {rep['noise_floor']:.3f}")
    print(f"between-geometry separation:                          {rep['between_geometry']:.3f}")
    print(f"separation / noise:                                   {rep['separation_ratio']:.2f}x")
    print(f"responsiveness (rule moved to a new field):           {rep['responsiveness']:.2f}x")
    print(f"morphospace participation ratio:                      "
          f"{rep['morphospace']['participation_ratio']:.2f}")
    print(f"\n{rep['verdict']}\n")


if __name__ == "__main__":
    main()
