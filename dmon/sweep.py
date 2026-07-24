"""Diffusion-length sweep — the knob that decides whether M0 can work at all.

`ARCHITECTURE.md` §M0: there is a Goldilocks band in `field_diffusion` relative to grid
diameter. Too fast and the field is uniform, so there is no gradient and no reason to
reach — a blob. Too slow and only cells touching a source survive. Neither end can
produce contingent morphology, so this is swept *before* anything is interpreted.

Everything except `field_diffusion` is held fixed, including the torch seed, so the
only thing varying between runs is the physics.

The output is deliberately two things, because neither alone is sufficient: a
descriptor table (the verdict is always numbers) and a contact sheet of final frames
(a descriptor table cannot tell you the body never left the seed site).

    python -m dmon.sweep --iters 4000 --grid 64 --steps 64 --batch 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from . import checkpoint
from .render import render_state, rollout_frames, to_gif
from .substrate import SubstrateConfig, descriptors, make_sources
from .train_m0 import train

DEFAULT_VALUES = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80)


@torch.no_grad()
def _final_state(sub, geom, grid, steps, device):
    dev = torch.device(device)
    x, r = sub.seed(8, grid, dev)
    src = make_sources(geom, 8, grid, dev)
    x, r, _ = sub.rollout(x, r, src, steps=steps)
    return x, r


def contact_sheet(panels, path: str | Path, pad: int = 4):
    """One image, one panel per swept value. Labels are left to the JSON — burning text
    into the sheet would mean carrying a font dependency for no analytical gain."""
    import numpy as np
    from PIL import Image

    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + pad * (len(panels) - 1)
    sheet = np.zeros((h, w, 3), dtype=np.uint8)
    xoff = 0
    for p in panels:
        sheet[: p.shape[0], xoff : xoff + p.shape[1]] = p
        xoff += p.shape[1] + pad
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path)
    return path


def sweep(
    values=DEFAULT_VALUES,
    geom: str = "west",
    iters: int = 4000,
    grid: int = 64,
    steps: int = 64,
    batch: int = 32,
    lr: float = 2e-3,
    device: str = "cpu",
    seed: int = 0,
    outdir: Path = Path("runs/sweep"),
    scale: int = 4,
    gifs: bool = True,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results, panels = [], []

    for v in values:
        tag = f"d{v:g}".replace(".", "p")
        print(f"\n=== field_diffusion = {v:g} ===")
        cfg = replace(SubstrateConfig(), field_diffusion=v)
        sub, history = train(
            geom, iters, grid, steps, batch, lr, device,
            log=max(1, iters // 10),
            ckpt=outdir / f"{tag}.pt",
            ckpt_every=max(1, iters // 2),
            seed=seed,
            cfg=cfg,
        )
        x, r = _final_state(sub, geom, grid, steps, device)
        d = descriptors(x, sub.cfg)
        results.append({"field_diffusion": v, "tag": tag, **d})
        print(f"  -> {json.dumps({k: round(val, 3) for k, val in d.items()})}")

        panels.append(render_state(x, r, sub, scale))
        if gifs:
            to_gif(rollout_frames(sub, geom, grid, steps, device, scale), outdir / f"{tag}.gif")

    sheet = contact_sheet(panels, outdir / "contact_sheet.png")
    (outdir / "sweep.json").write_text(
        json.dumps(
            {"geom": geom, "grid": grid, "steps": steps, "iters": iters,
             "seed": seed, "results": results},
            indent=2,
        )
    )

    print("\n=== sweep ===")
    print(f"{'diffusion':>10}  {'mass':>8} {'compact':>8} {'gyration':>8} {'box_dim':>8}")
    for r_ in results:
        # HANDOFF.md: below ~50 occupied cells the descriptors — box_dim especially —
        # are noise. Flag it in the table rather than trusting the reader to remember.
        flag = "  << mass<50, descriptors unreliable" if r_["mass"] < 50 else ""
        print(
            f"{r_['field_diffusion']:>10.3g}  {r_['mass']:>8.1f} {r_['compactness']:>8.2f} "
            f"{r_['gyration']:>8.2f} {r_['box_dim']:>8.2f}{flag}"
        )
    print(f"\ncontact sheet: {sheet}")
    print("Panels are left-to-right in the order above. Read the sheet before the table:")
    print("a body that never left the seed site scores like one that grew and stalled.")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--values", type=float, nargs="+", default=list(DEFAULT_VALUES))
    p.add_argument("--geom", default="west")
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--no-gifs", action="store_true")
    p.add_argument("--outdir", type=Path, default=Path("runs/sweep"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    sweep(
        a.values, a.geom, a.iters, a.grid, a.steps, a.batch, a.lr,
        a.device, a.seed, a.outdir, a.scale, not a.no_gifs,
    )


if __name__ == "__main__":
    main()
