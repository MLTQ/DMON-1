"""M0: train a rule to stay alive in a resource field, with no target image.

The only objective is sustained living mass. The falsification test is not "does it
look like a creature" but *contingency*: train the same architecture under different
source geometries and check that the resulting morphologies separate. If they do not,
shape is not coming from the ecology and M0 has failed no matter how good the renders
look.

The verdict itself lives in `contingency.py`, not here — it needs many training runs
and its own analysis, and keeping it out of this file stops the trainer from being the
thing that also grades itself.

    python -m dmon.train_m0 --geom west --iters 2000 --ckpt runs/west.pt
    python -m dmon.contingency --iters 20000 --seeds 5
"""

from __future__ import annotations

import argparse
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

    def reseed(self, frac: float):
        """Reset a fraction of the pool to fresh seeds.

        Called when the curriculum moves the sources. Without this the pool's
        already-grown bodies simply follow the food outward and nothing is ever asked
        to solve seed-to-first-meal at the new distance — which is exactly how a rule
        came to report mass 1100 in training and mass 0 from a fresh seed."""
        n = max(1, int(frac * self.x.shape[0]))
        idx = torch.randperm(self.x.shape[0], device=self.device)[:n]
        sx, sr = self.sub.seed(n, self.grid, self.device)
        self.x[idx], self.r[idx] = sx, sr


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
    cfg: SubstrateConfig | None = None,
    reseed_frac: float = 0.5,
):
    dev = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
    cfg = cfg or SubstrateConfig()
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
    history = []
    # sources are rebuilt as the curriculum walks them outward; cached so we only pay
    # for it when the spread actually changes
    _src_cache: dict[float, torch.Tensor] = {}

    def sources_at(progress: float):
        sp = round(cfg.spread_at(progress), 4)
        if sp not in _src_cache:
            _src_cache.clear()
            _src_cache[sp] = make_sources(geom, batch, grid, dev, spread=sp)
            if _src_cache.get("_seen"):
                pool.reseed(reseed_frac)  # the world moved; re-face the hard part
            _src_cache["_seen"] = True
        return _src_cache[sp], sp

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
                "field_diffusion": cfg.field_diffusion,
                "spread_end": cfg.spread_end,
                "history": history,
            },
        )

    for i in range(iters):
        idx, x, r = pool.sample(batch)
        src, spread = sources_at(i / max(1, iters - 1))
        src = src[: x.shape[0]]
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
            # The pool's descriptors are NOT what evaluation measures: pool states are
            # the product of thousands of composed steps, while evaluation starts from
            # a bare seed. When those diverge the run is failing and the log alone
            # would never show it, so report both side by side.
            fresh = _fresh_mass(sub, geom, grid, steps, dev, spread)
            history.append({"iter": i, "fresh_mass": fresh, "spread": spread, **d})
            warn = "  <<< DIES FROM SEED" if fresh < 1.0 and d["mass"] > 50 else ""
            print(
                f"[{i:5d}] mass={d['mass']:7.1f} compact={d['compactness']:5.2f} "
                f"gyr={d['gyration']:5.2f} dim={d['box_dim']:5.2f} spread={spread:4.2f} "
                f"fresh={fresh:7.1f}{warn}"
            )
        if ckpt is not None and (i + 1) % ckpt_every == 0:
            _save(i)

    _save(iters - 1)
    if ckpt is not None:
        print(f"saved {ckpt}")
    return sub, history


@torch.no_grad()
def _fresh_mass(sub, geom, grid, steps, dev, spread, reps: int = 4) -> float:
    """Mass reached from a bare seed. The number the training log cannot fake."""
    was = sub.training
    sub.eval()
    x, r = sub.seed(reps, grid, dev)
    src = make_sources(geom, reps, grid, dev, spread=spread)
    x, r, _ = sub.rollout(x, r, src, steps=steps)
    sub.train(was)
    return (x[:, :1] > sub.cfg.e_death).float().sum().item() / reps


@torch.no_grad()
def evaluate(sub: Substrate, geom: str, grid: int, steps: int, device: str, reps: int = 8):
    """Evaluated at the curriculum's *final* spread — the world the rule was last
    trained in. Evaluating at spread 1.0 would score every rule on an ecology it never
    saw and that `feasibility.py` says is unsurvivable."""
    dev = torch.device(device)
    x, r = sub.seed(reps, grid, dev)
    src = make_sources(geom, reps, grid, dev, spread=sub.cfg.spread_end)
    x, r, _ = sub.rollout(x, r, src, steps=steps)
    return descriptors(x, sub.cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geom", default="west")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    train(
        a.geom, a.iters, a.grid, a.steps, a.batch, a.lr, a.device,
        log=a.log, ckpt=a.ckpt, ckpt_every=a.ckpt_every, seed=a.seed,
    )


if __name__ == "__main__":
    main()
