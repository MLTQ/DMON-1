"""S0: learn character prediction from a stream that never stops.

The loop has no episodes. State persists across the entire run; only the autograd graph
is truncated, every `window` ticks. Nothing is ever reseeded and nothing is reset — if
the creature had to stop in order to learn, it would not be asynchronous and S0 would
not have been passed, whatever the loss curve said.

The metric is bits-per-character on a held-out tail of the corpus, which is a hard
number comparable to any other character model. Baselines run on the identical stream
with a matched parameter budget (see `baselines.py`); they are reported next to the
lattice, never afterwards.

    python -m dmon.stream.train --ticks 20000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .baselines import gru_matched_to, transformer_matched_to
from .corpus import CharStream, History
from .lattice import LatticeConfig, StreamingLattice

LOG2E = 1.0 / math.log(2.0)


@torch.no_grad()
def evaluate(model, vocab_stream_args: dict, ticks: int, mirror_len: int, device: str) -> float:
    """Bits-per-character on the held-out tail, from a cold state.

    Cold rather than warm on purpose: a warm state carries information from the training
    stream, and the number would flatter the model for reasons unrelated to what it
    learned. The first `mirror_len` ticks are burn-in and excluded.
    """
    stream = CharStream(split="eval", device=device, **vocab_stream_args)
    hist = History(mirror_len, stream.batch, device)
    state = model.blank_state(stream.batch, device)
    total, n = 0.0, 0
    for t in range(ticks):
        tok = stream.next()
        hist.push(tok)
        state, logits = model.tick(state, tok, hist.tokens)
        target = stream.peek()
        if t >= mirror_len:  # burn-in
            total += F.cross_entropy(logits, target).item()
            n += 1
    return (total / max(n, 1)) * LOG2E


def train_stream(
    model,
    name: str,
    ticks: int,
    window: int,
    batch: int,
    lr: float,
    device: str,
    mirror_len: int,
    log: int = 1000,
    eval_ticks: int = 2000,
    eval_every: int = 0,
    ckpt: Path | None = None,
    ckpt_every: int = 0,
):
    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    stream = CharStream(batch=batch, device=device, split="train")
    hist = History(mirror_len, batch, device)
    state = model.blank_state(batch, dev)

    running, seen, history_log = 0.0, 0, []
    chars_per_tick = batch

    def _snapshot(tick):
        if ckpt is None:
            return
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tick": tick, "state": model.state_dict(), "log": history_log}, ckpt)
    t0 = time.time()
    loss_acc = 0.0

    for t in range(ticks):
        tok = stream.next()
        hist.push(tok)
        state, logits = model.tick(state, tok, hist.tokens)
        target = stream.peek()
        step_loss = F.cross_entropy(logits, target)
        loss_acc = loss_acc + step_loss
        running += step_loss.item()
        seen += 1

        if (t + 1) % window == 0:
            opt.zero_grad()
            (loss_acc / window).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_acc = 0.0
            # The graph is truncated here. The STATE is not: it carries forward
            # detached, so the creature keeps its context across the boundary. Calling
            # blank_state() here instead would silently make this episodic.
            state = state.detach() if state is not None else None

        if (t + 1) % log == 0:
            bpc = (running / seen) * LOG2E
            ep = (t + 1) * chars_per_tick / 1_003_854
            rec = {"tick": t + 1, "train_bpc": bpc, "epochs": ep}
            print(
                f"[{name:9s} {t + 1:8d}] ep={ep:6.1f} train_bpc={bpc:6.3f}  "
                f"({(t + 1) / (time.time() - t0):6.1f} tick/s)"
            )
            running, seen = 0.0, 0
            # Held-out eval on a schedule, not only at the end. A 15-hour run whose
            # only measurement is its last line cannot be diagnosed, cannot be stopped
            # early on a plateau, and cannot show whether a late-emerging effect —
            # which is precisely what this run exists to look for — ever appeared.
            if eval_every and (t + 1) % eval_every == 0:
                ev = evaluate(model, {"batch": batch}, eval_ticks, mirror_len, device)
                rec["eval_bpc"] = ev
                print(f"[{name:9s} {t + 1:8d}] HELD-OUT bpc={ev:.4f}")
            history_log.append(rec)
        if ckpt_every and (t + 1) % ckpt_every == 0:
            _snapshot(t + 1)

    _snapshot(ticks)
    eval_bpc = evaluate(model, {"batch": batch}, eval_ticks, mirror_len, device)
    print(f"[{name:9s}  final ] held-out bpc = {eval_bpc:.3f}")
    return {"name": name, "params": model.param_count(), "eval_bpc": eval_bpc,
            "log": history_log}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--window", type=int, default=32, help="graph truncation length")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--micro", type=int, default=3)
    p.add_argument("--mirror-len", type=int, default=24)
    p.add_argument("--log", type=int, default=1000)
    p.add_argument("--eval-ticks", type=int, default=2000)
    p.add_argument("--only", default=None,
                   choices=["lattice", "gru", "inert", "transformer"])
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    vocab = CharStream(batch=1, device="cpu").vocab
    cfg = LatticeConfig(size=a.size, channels=a.channels, hidden=a.hidden,
                        vocab=vocab, micro=a.micro, mirror_len=a.mirror_len)
    lattice = StreamingLattice(cfg)
    n = lattice.param_count()
    print(f"vocab={vocab}  lattice params={n}")

    runs = []
    want = a.only
    if want in (None, "lattice"):
        runs.append(("lattice", StreamingLattice(cfg)))
    if want in (None, "inert"):
        runs.append(("inert", StreamingLattice(cfg).silence()))
    if want in (None, "gru"):
        g = gru_matched_to(n, vocab)
        print(f"gru params={g.param_count()} (budget {n})")
        runs.append(("gru", g))
    if want in (None, "transformer"):
        tr = transformer_matched_to(n, vocab, a.mirror_len)
        print(f"transformer params={tr.param_count()} (budget {n}, ctx {a.mirror_len})")
        runs.append(("transformer", tr))

    results = []
    for name, model in runs:
        results.append(
            train_stream(model, name, a.ticks, a.window, a.batch, a.lr, a.device,
                         a.mirror_len, a.log, a.eval_ticks, a.eval_every,
                         (a.ckpt.with_name(a.ckpt.stem + f"_{name}.pt") if a.ckpt else None),
                         a.ckpt_every)
        )

    print("\n=== S0: held-out bits per character ===")
    for r in sorted(results, key=lambda r_: r_["eval_bpc"]):
        print(f"  {r['name']:9s} params={r['params']:8d}  bpc={r['eval_bpc']:.3f}")
    print("\nUniform over vocab would be "
          f"{math.log2(vocab):.3f} bpc. Anything near that has learned nothing.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
