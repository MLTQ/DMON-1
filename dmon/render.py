"""Visualisation. For intuition only — never for the verdict.

`PROJECT.md` is explicit that descriptors decide pass/fail and renders do not, and this
file must not be allowed to quietly become the thing anyone judges a run by. Its actual
job is diagnostic: a starvation transition read off four scalars is exactly the
situation where you talk yourself into a trend that is not there.

Colour mapping is chosen so the three things that can go wrong are separable at a
glance:

    R  transport conductance (gated, body only)  — vasculature; is it building one?
    G  energy / e_max                            — the body and how fed it is
    B  resource field / field_cap                — where the food actually is

So: food is blue, body is green, a body sitting on food is cyan, and a conducting
network reads yellow-white. A body that never turns yellow never built transport.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .substrate import Substrate, make_sources


def render_state(x: torch.Tensor, r: torch.Tensor, sub: Substrate, scale: int = 8):
    """One frame from a batch of states. Renders batch element 0. Returns HxWx3 uint8."""
    import numpy as np

    cfg = sub.cfg
    alive = (x[:1, :1] > cfg.e_death).float()
    energy = (x[:1, :1] / cfg.e_max).clamp(0, 1)
    conduct = torch.sigmoid(x[:1, 2:3] + sub.gate_bias[1]) * alive
    field = (r[:1, :1] / cfg.field_cap).clamp(0, 1)

    rgb = torch.cat([conduct, energy, field], dim=1)  # (1,3,H,W)
    if scale > 1:
        rgb = F.interpolate(rgb, scale_factor=scale, mode="nearest")
    arr = (rgb[0].permute(1, 2, 0).detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    return np.asarray(arr)


def to_gif(frames, path: str | Path, fps: int = 20) -> Path:
    """Write frames as an animated GIF."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=max(1, int(1000 / fps)),
        loop=0,
    )
    return path


@torch.no_grad()
def rollout_frames(
    sub: Substrate,
    geom: str,
    grid: int,
    steps: int,
    device: str = "cpu",
    scale: int = 8,
    every: int = 1,
):
    """Run a rollout and collect frames. No gradient, single sample."""
    dev = torch.device(device)
    x, r = sub.seed(1, grid, dev)
    src = make_sources(geom, 1, grid, dev)
    frames = [render_state(x, r, sub, scale)]
    for i in range(steps):
        x, r = sub.step(x, r, src)
        if (i + 1) % every == 0:
            frames.append(render_state(x, r, sub, scale))
    return frames


def main():
    import argparse

    from .checkpoint import load

    p = argparse.ArgumentParser(description="Render a trained rule growing.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--geom", default="west")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default="runs/growth.gif")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    sub, meta = load(a.ckpt, a.device)
    frames = rollout_frames(sub, a.geom, a.grid, a.steps, a.device, a.scale, a.every)
    out = to_gif(frames, a.out, a.fps)
    print(f"wrote {out} ({len(frames)} frames, geom={a.geom}, trained_geom={meta.get('geom')})")


if __name__ == "__main__":
    main()
