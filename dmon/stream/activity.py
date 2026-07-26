"""Where in the lattice is anything actually happening?

S2 failed with born-large scoring the same as small, which says extra cells buy nothing
however they are acquired. The first hypothesis to eliminate is the dullest one: the
periphery may simply be dead. New cells start at zero state, the rule is local, and
nothing in the design pushes activity outward — so if the region near the I/O port is
sufficient for the task, the rest of the lattice can sit at exactly zero forever and a
bigger grid is bigger only in the config file.

This measures state magnitude against distance from the port. If the active radius is
small compared to the grid, then size was never the variable it appeared to be, and no
amount of growth would have helped.

    python -m dmon.stream.activity --size 40 --ticks 20000 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .corpus import CharStream, History
from .lattice import LatticeConfig, StreamingLattice


def train_briefly(model, ticks, window, batch, lr, device, mirror_len, log=5000):
    dev = torch.device(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stream = CharStream(batch=batch, device=device, split="train")
    hist = History(mirror_len, batch, device)
    state = model.blank_state(batch, dev)
    acc = 0.0
    use_aux = model.cfg.local_pred > 0
    for t in range(ticks):
        tok = stream.next()
        hist.push(tok)
        if use_aux:
            state, logits, aux = model.tick(state, tok, hist.tokens, return_aux=True)
            acc = acc + F.cross_entropy(logits, stream.peek()) + model.cfg.local_pred * aux
        else:
            state, logits = model.tick(state, tok, hist.tokens)
            acc = acc + F.cross_entropy(logits, stream.peek())
        if (t + 1) % log == 0:
            # Collapse watch: if the local objective is being satisfied by going still,
            # state spread falls toward zero and the loss looks excellent while the
            # creature stops existing.
            print(f"  [{t+1:6d}] state std={state.std().item():.4f}")
        if (t + 1) % window == 0:
            opt.zero_grad()
            (acc / window).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc = 0.0
            state = state.detach()
    return state


@torch.no_grad()
def profile(model, state) -> dict:
    """Mean |state| per ring of Chebyshev distance from the port centre."""
    s = model.cfg.size
    cy = (model.in_slice[0].start + model.in_slice[0].stop) // 2
    cx = (model.out_slice[1].start + model.in_slice[1].start) // 2
    ys = torch.arange(s, device=state.device)[:, None]
    xs = torch.arange(s, device=state.device)[None, :]
    dist = torch.maximum((ys - cy).abs(), (xs - cx).abs())

    mag = state.abs().mean(dim=(0, 1))  # (S, S)
    rings = []
    for d in range(int(dist.max().item()) + 1):
        m = dist == d
        rings.append({"dist": d, "cells": int(m.sum().item()),
                      "mean_abs": float(mag[m].mean().item())})
    peak = max(r["mean_abs"] for r in rings) or 1e-12
    # "Active" = at least 1% of the strongest ring. Generous on purpose: the question
    # is whether the periphery does *anything*, not whether it does much.
    active = [r["dist"] for r in rings if r["mean_abs"] >= 0.01 * peak]
    return {"rings": rings, "peak": peak,
            "active_radius": max(active) if active else 0,
            "grid_radius": int(dist.max().item())}


@torch.no_grad()
def ablate_beyond(model, radius: int, ticks: int, batch: int, device: str,
                  mirror_len: int) -> float:
    """Held-out bpc with every cell beyond `radius` of the port held at zero.

    Activity magnitude is not contribution — a cell can be busy and irrelevant. This is
    the measurement that separates them: if bpc is unchanged with the periphery clamped
    off, the periphery was decorative, whatever its amplitude.
    """
    import math

    s = model.cfg.size
    cy = (model.in_slice[0].start + model.in_slice[0].stop) // 2
    cx = (model.out_slice[1].start + model.in_slice[1].start) // 2
    ys = torch.arange(s, device=device)[:, None]
    xs = torch.arange(s, device=device)[None, :]
    keep = (torch.maximum((ys - cy).abs(), (xs - cx).abs()) <= radius).float()
    # ALWAYS preserve the mirror row. The first version of this sweep did not, and the
    # mirror sits 4 rows off the port centre — so small radii were deleting the
    # creature's recent-history buffer at the same time as its peripheral compute, and
    # the two effects were inseparable. Holding the mirror fixed makes this a
    # measurement of computation only.
    keep[model.mirror_slice[0], model.mirror_slice[1]] = 1.0
    keep = keep[None, None]

    stream = CharStream(split="eval", batch=batch, device=device)
    hist = History(mirror_len, batch, device)
    state = model.blank_state(batch, device)
    tot, n = 0.0, 0
    for t in range(ticks):
        tok = stream.next()
        hist.push(tok)
        state, logits = model.tick(state, tok, hist.tokens)
        state = state * keep  # clamp the periphery off, every tick
        if t >= mirror_len:
            tot += F.cross_entropy(logits, stream.peek()).item()
            n += 1
    return (tot / max(n, 1)) / math.log(2.0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--micro", type=int, default=3)
    p.add_argument("--mirror-len", type=int, default=24)
    p.add_argument("--local-pred", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    vocab = CharStream(batch=1, device="cpu").vocab
    torch.manual_seed(0)
    model = StreamingLattice(
        LatticeConfig(size=a.size, channels=a.channels, hidden=a.hidden, vocab=vocab,
                      micro=a.micro, mirror_len=a.mirror_len,
                      local_pred=a.local_pred)
    ).to(a.device)

    state = train_briefly(model, a.ticks, a.window, a.batch, a.lr, a.device, a.mirror_len)
    res = profile(model, state)

    print(f"\n=== activity vs distance from port (size {a.size}) ===")
    for r in res["rings"]:
        bar = "#" * int(60 * r["mean_abs"] / max(res["peak"], 1e-12))
        print(f"  d={r['dist']:3d} cells={r['cells']:4d} mean|x|={r['mean_abs']:.5f} {bar}")
    print(f"\nactive radius {res['active_radius']} of {res['grid_radius']} available")
    frac = (2 * res["active_radius"] + 1) ** 2 / (a.size**2)
    print(f"that is ~{frac:.0%} of the lattice doing anything at all")
    if res["active_radius"] < res["grid_radius"] * 0.6:
        print("\nThe periphery is dead. Grid size was never the variable it looked "
              "like — S2's result is explained without any claim about growth.")

    print(f"\n=== ablation: compute cells clamped off, mirror always intact ===")
    abl = []
    for radius in (2, 3, 5, 8, 12, 20, a.size):
        bpc = ablate_beyond(model, radius, 3000, a.batch, a.device, a.mirror_len)
        abl.append({"radius": radius, "bpc": bpc})
        tag = "  (whole lattice)" if radius >= res["grid_radius"] else ""
        print(f"  keep r<={radius:3d}  bpc={bpc:.3f}{tag}")
    full = abl[-1]["bpc"]
    small = next(x for x in abl if x["radius"] == 5)["bpc"]
    print(f"\ncost of discarding everything beyond r=5: {small - full:+.3f} bpc")
    if abs(small - full) < 0.05:
        print("The periphery is ACTIVE BUT NOT USED. Extra cells carry no task-relevant "
              "information, which explains S2 completely: growth cannot help when the "
              "cells already present contribute nothing.")
    else:
        print("The periphery does contribute — so S2's failure is about growth "
              "dynamics or optimisation, not about capacity being unusable.")
    res["ablation"] = abl

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
