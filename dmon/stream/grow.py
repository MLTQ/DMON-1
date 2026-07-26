"""S2: can the creature be made bigger at runtime, and does it use the room?

This is the claim that justifies cellular automata over any other recurrent
architecture. The rule is shared and local, so adding cells does not change the
parameter count — capacity can be added to a *running* creature without retraining. A
transformer cannot do that. If it turns out the added cells are inert, the main
architectural argument for this substrate is gone and we should know early.

Three arms, one stream, one seed, identical total ticks:

    small   trained at `size` throughout — the control for "did it just train longer"
    grown   trained at `size`, then extended to `big` mid-run and trained on
    born    trained at `big` from tick zero — the ceiling, for "is growth as good as
            having always been large"

`grown` beating `small` is the pass. `grown` approaching `born` is the stronger result:
it says growth costs little. `grown` collapsing at the transition is a real failure and
is reported separately from the final number, because a creature that has to be
rebooted to grow has not grown.

    python -m dmon.stream.grow --ticks 40000 --grow-at 20000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .corpus import CharStream, History
from .lattice import LatticeConfig, StreamingLattice
from .train import LOG2E, evaluate


def run_arm(
    name: str,
    size: int,
    grow_to: int | None,
    grow_at: int | None,
    ticks: int,
    window: int,
    batch: int,
    lr: float,
    device: str,
    cfg_kw: dict,
    log: int,
    eval_ticks: int,
):
    dev = torch.device(device)
    cfg = LatticeConfig(size=size, **cfg_kw)
    model = StreamingLattice(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    stream = CharStream(batch=batch, device=device, split="train", seed=0)
    hist = History(cfg.mirror_len, batch, device)
    state = model.blank_state(batch, dev)

    curve, running, seen = [], 0.0, 0
    loss_acc, t0 = 0.0, time.time()
    grew_at_tick = None

    for t in range(ticks):
        if grow_to and grow_at and t == grow_at:
            # Flush the pending window first: the graph belongs to the old lattice.
            if seen and isinstance(loss_acc, torch.Tensor):
                opt.zero_grad()
                (loss_acc / max(1, (t % window) or window)).backward()
                opt.step()
            loss_acc = 0.0
            n_before = model.param_count()
            model, migrate = model.grown(grow_to)
            model = model.to(dev)
            state = migrate(state.detach())
            # The optimiser must be rebuilt around the new parameter objects, but the
            # VALUES are the old ones — this is not a reinitialisation.
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            assert model.param_count() == n_before, (
                f"growth changed the parameter count ({n_before} -> "
                f"{model.param_count()}); it is supposed to be free"
            )
            grew_at_tick = t
            print(f"[{name:6s}] grew {size} -> {grow_to} at tick {t}, params unchanged "
                  f"({n_before})")

        tok = stream.next()
        hist.push(tok)
        state, logits = model.tick(state, tok, hist.tokens)
        step_loss = F.cross_entropy(logits, stream.peek())
        loss_acc = loss_acc + step_loss
        running += step_loss.item()
        seen += 1

        if (t + 1) % window == 0:
            opt.zero_grad()
            (loss_acc / window).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_acc = 0.0
            state = state.detach()

        if (t + 1) % log == 0:
            bpc = (running / seen) * LOG2E
            curve.append({"tick": t + 1, "train_bpc": bpc})
            print(f"[{name:6s} {t + 1:7d}] train_bpc={bpc:6.3f} "
                  f"size={model.cfg.size:3d} ({(t + 1) / (time.time() - t0):6.1f} tick/s)")
            running, seen = 0.0, 0

    eval_bpc = evaluate(model, {"batch": batch}, eval_ticks, cfg.mirror_len, device)
    print(f"[{name:6s}  final ] held-out bpc = {eval_bpc:.3f}")
    return {
        "arm": name,
        "final_size": model.cfg.size,
        "params": model.param_count(),
        "eval_bpc": eval_bpc,
        "grew_at": grew_at_tick,
        "curve": curve,
    }


def transition_cost(curve: list[dict], grow_at: int) -> float | None:
    """How much bpc got worse in the log window straddling the growth event.

    Reported separately from the final score on purpose. A creature that has to be
    rebooted to get bigger has not grown, and a final number alone would hide a
    catastrophic dip followed by recovery.
    """
    before = [c for c in curve if c["tick"] <= grow_at]
    after = [c for c in curve if c["tick"] > grow_at]
    if not before or not after:
        return None
    return after[0]["train_bpc"] - before[-1]["train_bpc"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ticks", type=int, default=40000)
    p.add_argument("--grow-at", type=int, default=20000)
    p.add_argument("--size", type=int, default=24)
    p.add_argument("--big", type=int, default=40)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--micro", type=int, default=3)
    p.add_argument("--mirror-len", type=int, default=24)
    p.add_argument("--log", type=int, default=5000)
    p.add_argument("--eval-ticks", type=int, default=3000)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    vocab = CharStream(batch=1, device="cpu").vocab
    cfg_kw = dict(channels=a.channels, hidden=a.hidden, vocab=vocab,
                  micro=a.micro, mirror_len=a.mirror_len)

    arms = [
        ("small", a.size, None, None),
        ("grown", a.size, a.big, a.grow_at),
        ("born", a.big, None, None),
    ]
    results = []
    for name, size, grow_to, grow_at in arms:
        torch.manual_seed(0)
        results.append(
            run_arm(name, size, grow_to, grow_at, a.ticks, a.window, a.batch, a.lr,
                    a.device, cfg_kw, a.log, a.eval_ticks)
        )

    by = {r["arm"]: r for r in results}
    print("\n=== S2: growth ===")
    for r in results:
        print(f"  {r['arm']:6s} size={r['final_size']:3d} params={r['params']:8d} "
              f"bpc={r['eval_bpc']:.3f}")

    g, s, b = by["grown"], by["small"], by["born"]
    dip = transition_cost(g["curve"], a.grow_at)
    print(f"\ngrowth transition cost: "
          f"{'n/a' if dip is None else f'{dip:+.3f} bpc in the window after growing'}")

    if g["eval_bpc"] < s["eval_bpc"] - 0.02:
        gap = (g["eval_bpc"] - b["eval_bpc"])
        print(f"PASS — growing beat staying small by {s['eval_bpc'] - g['eval_bpc']:.3f} "
              f"bpc at identical parameter count. Gap to born-large: {gap:+.3f}.")
    elif abs(g["eval_bpc"] - s["eval_bpc"]) <= 0.02:
        print("FAIL — added cells are inert. Growth changed nothing, so lattice "
              "capacity is not a usable axis and the main architectural argument for "
              "this substrate does not hold as stated.")
    else:
        print("FAIL — growing made it worse. The creature did not absorb the new "
              "substrate; check the transition cost above for whether it ever "
              "recovered.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
