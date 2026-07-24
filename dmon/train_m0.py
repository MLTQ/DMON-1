"""M0: train a rule to stay alive in a resource field, with no target image.

The only objective is sustained living mass. The falsification test is not "does it
look like a creature" but *contingency*: train the same architecture under different
source geometries and check that the resulting morphologies separate. If they do not,
shape is not coming from the ecology and M0 has failed no matter how good the renders
look.

    python -m dmon.train_m0 --geom west --iters 2000
    python -m dmon.train_m0 --contingency
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import checkpoint
from .substrate import Substrate, SubstrateConfig, descriptors, make_sources


class Pool:
    """Growing-NCA style sample pool: lets short BPTT windows compose into long
    horizons. Worst-performing sample is reset to seed each draw so the rule cannot
    forget how to start from nothing."""

    def __init__(self, sub: Substrate, size: int, grid: int, device):
        self.sub, self.grid, self.device = sub, grid, device
        self.x, self.r = sub.seed(size, grid, device)

    def sample(self, batch: int):
        idx = torch.randint(0, self.x.shape[0], (batch,), device=self.device)
        x, r = self.x[idx].clone(), self.r[idx].clone()
        mass = (x[:, :1] > self.sub.cfg.e_death).float().sum(dim=(1, 2, 3))
        worst = mass.argmin()
        sx, sr = self.sub.seed(1, self.grid, self.device)
        x[worst], r[worst] = sx[0], sr[0]
        return idx, x, r

    def commit(self, idx, x, r):
        self.x[idx], self.r[idx] = x.detach(), r.detach()


def train(
    geom: str,
    iters: int,
    grid: int,
    steps: int,
    batch: int,
    lr: float,
    device: str,
    log: int = 100,
    ckpt: Path | None = None,
    ckpt_every: int = 1000,
    seed: int | None = None,
):
    dev = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
    cfg = SubstrateConfig()
    if not cfg.light_cone_ok(grid, steps):
        print(
            f"[warn] light cone violated: steps={steps} < grid={grid}. Information "
            f"cannot cross the field; expect a mysteriously featureless result. "
            f"See ARCHITECTURE.md §4."
        )
    sub = Substrate(cfg).to(dev)
    opt = torch.optim.Adam(sub.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)

    pool = Pool(sub, 256, grid, dev)
    src_pool = make_sources(geom, 256, grid, dev)
    history = []

    def _save(i):
        if ckpt is None:
            return
        checkpoint.save(
            ckpt,
            sub,
            {
                "geom": geom,
                "grid": grid,
                "steps": steps,
                "iters": i + 1,
                "batch": batch,
                "lr": lr,
                "seed": seed,
                "history": history,
            },
        )

    for i in range(iters):
        idx, x, r = pool.sample(batch)
        src = src_pool[: x.shape[0]]
        n = int(torch.randint(steps // 2, steps + 1, (1,)).item())
        x, r, mass = sub.rollout(x, r, src, steps=n)

        loss = -mass.mean() / (grid * grid)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sub.parameters(), 1.0)
        opt.step()
        sched.step()
        pool.commit(idx, x, r)

        if i % log == 0 or i == iters - 1:
            d = descriptors(x, cfg)
            history.append({"iter": i, **d})
            print(
                f"[{i:5d}] mass={d['mass']:7.1f} compact={d['compactness']:5.2f} "
                f"gyr={d['gyration']:5.2f} dim={d['box_dim']:5.2f}"
            )
        if ckpt is not None and (i + 1) % ckpt_every == 0:
            _save(i)

    _save(iters - 1)
    if ckpt is not None:
        print(f"saved {ckpt}")
    return sub, history


@torch.no_grad()
def evaluate(sub: Substrate, geom: str, grid: int, steps: int, device: str, reps: int = 8):
    dev = torch.device(device)
    x, r = sub.seed(reps, grid, dev)
    src = make_sources(geom, reps, grid, dev)
    x, r, _ = sub.rollout(x, r, src, steps=steps)
    return descriptors(x, sub.cfg)


def contingency(geoms, iters, grid, steps, batch, lr, device, out: Path | None = None):
    """The actual M0 verdict. Train one rule per geometry, compare morphologies."""
    results = {}
    for g in geoms:
        print(f"\n=== training under '{g}' ===")
        sub, _ = train(g, iters, grid, steps, batch, lr, device)
        results[g] = {"self": evaluate(sub, g, grid, steps, device)}
        # cross-evaluate: does a rule trained on g behave differently elsewhere?
        for h in geoms:
            if h != g:
                results[g][h] = evaluate(sub, h, grid, steps, device)

    print("\n=== contingency ===")
    keys = ["mass", "compactness", "gyration", "box_dim"]
    for g in geoms:
        d = results[g]["self"]
        print(f"{g:10s} " + "  ".join(f"{k}={d[k]:7.2f}" for k in keys))

    spread = {
        k: max(results[g]["self"][k] for g in geoms) - min(results[g]["self"][k] for g in geoms)
        for k in keys
    }
    print("\nbetween-geometry spread:", {k: round(v, 3) for k, v in spread.items()})
    print("PASS if this spread exceeds within-geometry seed noise. Measure that too.")

    if out:
        out.write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geom", default="west")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--contingency", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    if a.contingency:
        contingency(
            ["center", "west", "poles", "corners"],
            a.iters, a.grid, a.steps, a.batch, a.lr, a.device, a.out,
        )
    else:
        train(
            a.geom, a.iters, a.grid, a.steps, a.batch, a.lr, a.device,
            log=a.log, ckpt=a.ckpt, ckpt_every=a.ckpt_every, seed=a.seed,
        )


if __name__ == "__main__":
    main()
