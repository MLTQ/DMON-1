"""What actually fits, and on which axis. Measured, not estimated.

Parameters are nearly free here — 94K of them is 0.4 MB. The memory is retained BPTT
activations, and it scales as

    window x micro x batch x hidden x size^2

which is **linear in `hidden` and quadratic in `size`**. Those are the two capacity axes
from ARCHITECTURE.md §4, and they do not cost the same: tripling `hidden` roughly
triples memory, while tripling the grid side is a factor of nine. Reasoning about
"tripling the network" without saying which axis will overshoot by 3x.

`window x micro` is the credit-assignment depth and is the cheapest lever of all — it
buys memory back linearly without touching capacity at all, at the cost of how far back
the creature can assign blame.

    python -m dmon.stream.capacity --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .corpus import CharStream, History
from .lattice import LatticeConfig, StreamingLattice


def probe_config(
    size: int, channels: int, hidden: int, batch: int, window: int, micro: int,
    vocab: int, device: str, mirror_len: int = 24,
) -> dict:
    """One truncation window's worth of forward + backward. Returns peak GB."""
    dev = torch.device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = LatticeConfig(size=size, channels=channels, hidden=hidden, vocab=vocab,
                        micro=micro, mirror_len=mirror_len)
    model = StreamingLattice(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stream = CharStream(batch=batch, device=device, split="train")
    hist = History(mirror_len, batch, device)
    state = model.blank_state(batch, dev)

    loss = 0.0
    for _ in range(window):
        tok = stream.next()
        hist.push(tok)
        state, logits = model.tick(state, tok, hist.tokens)
        loss = loss + F.cross_entropy(logits, stream.peek())
    opt.zero_grad()
    (loss / window).backward()
    opt.step()

    peak = torch.cuda.max_memory_allocated() / 1e9
    out = {"size": size, "channels": channels, "hidden": hidden, "batch": batch,
           "window": window, "micro": micro, "params": model.param_count(),
           "peak_gb": peak}
    del model, opt, state, loss
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--budget-gb", type=float, default=22.0, help="usable VRAM")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    vocab = CharStream(batch=1, device="cpu").vocab
    base = dict(size=32, channels=32, hidden=256, batch=16, window=32, micro=3)

    trials = [("baseline (S0)", base)]
    for k, mult, label in [
        ("hidden", 3, "hidden x3"),
        ("hidden", 8, "hidden x8"),
        ("channels", 3, "channels x3"),
        ("size", 2, "grid side x2"),
        ("size", 3, "grid side x3"),
        ("batch", 3, "batch x3"),
    ]:
        cfg = dict(base)
        cfg[k] = base[k] * mult
        trials.append((label, cfg))
    cheap = dict(base)
    cheap["window"] = 8
    trials.append(("window 32->8", cheap))

    rows = []
    for label, cfg in trials:
        try:
            r = probe_config(vocab=vocab, device=a.device, **cfg)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            r = {**cfg, "params": None, "peak_gb": float("inf")}
        r["label"] = label
        rows.append(r)
        fits = "" if r["peak_gb"] <= a.budget_gb else "   << does not fit"
        pk = "OOM" if r["peak_gb"] == float("inf") else f"{r['peak_gb']:6.2f}"
        print(f"{label:16s} size={r['size']:3d} ch={r['channels']:3d} "
              f"hid={r['hidden']:5d} batch={r['batch']:3d} win={r['window']:3d}  "
              f"params={str(r['params']):>9s}  peak={pk} GB{fits}")

    b = rows[0]["peak_gb"]
    print(f"\nbaseline peak {b:.2f} GB of {a.budget_gb:.0f} GB usable "
          f"({a.budget_gb / max(b, 1e-9):.1f}x headroom)")
    print("Params are not the constraint — retained activations are. Memory is linear "
          "in hidden/batch/window and QUADRATIC in grid side.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
