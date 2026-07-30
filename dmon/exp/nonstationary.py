"""Experiment 1: does a continually-living model adapt to a changing world?

Every result so far — mine and the SOL spike's — was measured on a stationary corpus
scored by held-out BPC. On that task nothing the architecture exists for can pay:
there is no distribution change, so continual adaptation earns nothing; reward is dense
and immediate, so delayed-credit machinery is overhead; task complexity is fixed, so
capacity can never bind. The SOL S0 table reads exactly like that — frozen synapses beat
trainable ones and no-metabolism beats metabolism, each by ~0.03 bpc. Those are taxes on
mechanisms that had no opportunity to earn.

So this measures something a stationary benchmark structurally cannot.

**The task.** The stream alternates between two regimes every `block` characters.
Regime B is regime A under a fixed permutation of the vocabulary: identical text,
relabelled symbols.

That choice is the point. A different corpus would confound adaptation with difficulty —
if B is intrinsically harder, a model looks slow to adapt when it is merely facing a
worse task. Under a permutation the two regimes have **identical entropy, identical
unigram statistics, and identical sequential structure**, so the only thing that changes
at a switch is the symbol mapping. Every bit of excess loss after a switch is
adaptation cost and nothing else.

**The metric is not final BPC.** It is BPC as a function of characters-since-switch,
averaged over many switches. A model that adapts shows a spike that decays; a model that
cannot shows a flat elevated line at the blend of both regimes. Reporting a single
number here would average the two into the same value and hide the entire effect —
which is the failure mode this file exists to avoid.

    python -m dmon.exp.nonstationary --ticks 200000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

LOG2E = 1.0 / math.log(2.0)
DEFAULT_SOURCE = Path.home() / "Code/petridish/data/tinyshakespeare/input.txt"


def build_corpus(source: Path, block: int, seed: int = 0):
    """Alternating regimes. Returns (tokens, vocab_size, regime_of_position)."""
    text = Path(source).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    base = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(chars), generator=g)
    # A derangement-ish check: if the permutation happens to fix most symbols the two
    # regimes are nearly identical and the experiment measures nothing.
    fixed = int((perm == torch.arange(len(chars))).sum())
    if fixed > len(chars) // 8:
        raise RuntimeError(f"permutation fixes {fixed} symbols; regimes too similar")

    idx = torch.arange(len(base))
    regime = ((idx // block) % 2).to(torch.uint8)
    tokens = torch.where(regime.bool(), perm[base], base)
    return tokens, len(chars), regime


class SwitchStream:
    """Endless reader over the alternating corpus, tracking distance since each switch."""

    def __init__(self, tokens, block, batch, device, split="train", holdout=0.1, seed=0):
        n = len(tokens)
        cut = int(n * (1.0 - holdout))
        # Split on a block boundary so the held-out tail contains whole regimes rather
        # than starting mid-block with an unknown amount of adaptation already done.
        cut -= cut % block
        self.data = (tokens[:cut] if split == "train" else tokens[cut:]).to(device)
        self.block = block
        g = torch.Generator().manual_seed(seed)
        self.pos = torch.randint(0, len(self.data) - 1, (batch,), generator=g).to(device)
        self.batch = batch

    def next(self):
        tok = self.data[self.pos]
        since = (self.pos % self.block).clone()
        self.pos = (self.pos + 1) % (len(self.data) - 1)
        return tok, since

    def peek(self):
        return self.data[self.pos % (len(self.data) - 1)]


def adaptation_curve(since, loss, block, bins=32):
    """Mean bpc per bin of characters-since-switch, on LOG-SPACED edges.

    Linear bins cannot see fast adaptation. With 32 linear bins over a 4000-character
    block the first bin covers characters 0-125, so a model that infers the new symbol
    mapping in-context within a few dozen characters is already adapted for almost all
    of that bin, and the curve reads flat for precisely the opposite of the real reason.
    Since the whole experiment is "how fast does it recover", the resolution has to be
    where the recovery happens.
    """
    edges = torch.unique(torch.cat([
        torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                      384, 512, 768, 1024, 1536, 2048, 3072]),
        torch.tensor([block]),
    ]))
    edges = edges[edges <= block]
    if edges[-1] < block:
        edges = torch.cat([edges, torch.tensor([block])])
    out = []
    for i in range(len(edges) - 1):
        m = (since >= edges[i]) & (since < edges[i + 1])
        if m.any():
            out.append((int(edges[i]), (loss[m].mean() * LOG2E).item()))
    return out


def run_arm(model, name, tokens, vocab, block, ticks, window, batch, lr, device,
            mirror_len, log=20000):
    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stream = SwitchStream(tokens, block, batch, device, "train")
    from ..stream.corpus import History

    hist = History(mirror_len, batch, device)
    state = model.blank_state(batch, dev)

    acc, t0 = 0.0, time.time()
    sinces, losses = [], []
    for t in range(ticks):
        tok, since = stream.next()
        hist.push(tok)
        state, logits = model.tick(state, tok, hist.tokens)
        per = F.cross_entropy(logits, stream.peek(), reduction="none")
        acc = acc + per.mean()
        # Second half only: the first half is the model learning the task at all, and
        # mixing that into an adaptation curve would read as slow adaptation.
        if t >= ticks // 2:
            sinces.append(since.cpu())
            losses.append(per.detach().cpu())
        if (t + 1) % window == 0:
            opt.zero_grad()
            (acc / window).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc = 0.0
            state = model.detach(state)
        if (t + 1) % log == 0:
            print(f"[{name:9s} {t + 1:7d}] {(t + 1) / (time.time() - t0):6.1f} tick/s")

    since = torch.cat(sinces)
    loss = torch.cat(losses)
    curve = adaptation_curve(since, loss, block)
    # First 16 characters vs the last quarter of the block. "Just after a switch" has to
    # mean genuinely just after, not "in the first 3% of the block".
    early = (loss[since < 16].mean() * LOG2E).item()
    late = (loss[since >= block * 3 // 4].mean() * LOG2E).item()
    # Keep a raw sample so any later question about binning is answerable without
    # re-running. Discarding it cost this experiment one arm already.
    keep = torch.randperm(len(since))[:200_000]
    return {
        "arm": name,
        "params": model.param_count(),
        "curve": curve,
        "raw_since": since[keep].tolist(),
        "raw_loss": loss[keep].tolist(),
        "bpc_just_after_switch": early,
        "bpc_well_after_switch": late,
        "adaptation_gain": early - late,
        "mean_bpc": (loss.mean() * LOG2E).item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--block", type=int, default=4000, help="chars between switches")
    p.add_argument("--ticks", type=int, default=200000)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--cells", type=int, default=64)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--dendrites", type=int, default=8)
    p.add_argument("--sensory", type=int, default=8)
    p.add_argument("--outputs", type=int, default=8)
    p.add_argument("--no-metabolism", action="store_true")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--mirror-len", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("runs/exp1"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    tokens, vocab, _ = build_corpus(a.source, a.block)
    print(f"vocab={vocab}  block={a.block}  corpus={len(tokens):,} chars")

    from .adapters import SolArm, TensorStateArm, sol_matched_to
    from ..organism.model import SolConfig
    from ..stream.baselines import gru_matched_to

    # SOL sets the parameter budget; every other arm is matched down to it.
    sol = SolArm(SolConfig(vocab_size=vocab, cells=a.cells, channels=a.channels,
                           dendrites=a.dendrites,
                           sensory_cells=a.sensory, output_cells=a.outputs))
    n = sol.param_count()
    arms = [("sol", sol), ("gru", TensorStateArm(gru_matched_to(n, vocab)))]
    if a.no_metabolism:
        from dataclasses import replace as _rep
        flat = SolArm(_rep(sol.cfg, basal_cost=0.0, activity_cost=0.0,
                           stimulation_gain=0.0))
        arms.insert(1, ("sol-no-metabolism", flat))
    print(f"params: " + "  ".join(f"{k}={m.param_count()}" for k, m in arms))

    results = []
    for name, model in arms:
        results.append(run_arm(model, name, tokens, vocab, a.block, a.ticks, a.window,
                               a.batch, a.lr, a.device, a.mirror_len))
        r = results[-1]
        print(f"[{name}] just-after-switch={r['bpc_just_after_switch']:.3f}  "
              f"well-after={r['bpc_well_after_switch']:.3f}  "
              f"adaptation gain={r['adaptation_gain']:+.3f} bpc")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(results, indent=2))

    from ..stream.plots import adaptation_curves
    series = {r["arm"]: r["curve"] for r in results}
    best = max(results, key=lambda r: r["adaptation_gain"])
    any_gain = best["adaptation_gain"] > 0.05
    verdict = (
        f"{best['arm']} recovers {best['adaptation_gain']:.3f} bpc across a block"
        if any_gain else
        "NO arm shows meaningful recovery within a block - none adapts to the switch"
    )
    fig = adaptation_curves(series,
        "Does anything adapt to an unannounced change of regime?",
        verdict, a.out / "adaptation.png", block=a.block)
    print(f"\nwrote {a.out / 'results.json'} and {fig}")


if __name__ == "__main__":
    main()
