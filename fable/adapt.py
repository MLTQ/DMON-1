"""F2: adaptability under regime cycling — stream, trainer, and report.

See fable/experiments/f2-adaptability.md. The scored quantities are lifetime
properties (savings across regime revisits, interference across regime
absence), which in-context inference cannot satisfy — the lesson of
dmon/exp/nonstationary.py, whose block-switch design the GRU won by
re-inferring the symbol map in ~10 characters.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import FableConfig
from .evaluate import LN2
from .model import Fable
from .stream import load_corpus
from .train import add_config_args, build_model, build_optimizer, build_schedule, config_from_args

LOG_BIN_EDGES = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
                 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
                 8192, 12288, 16384, 24576]


def sample_permutation(vocab: int, seed: int) -> torch.Tensor:
    """Vocabulary permutation with fewer than vocab/8 fixed points (exp1's
    guard: a near-identity permutation would measure nothing)."""
    gen = torch.Generator().manual_seed(seed)
    while True:
        perm = torch.randperm(vocab, generator=gen)
        if int((perm == torch.arange(vocab)).sum()) < vocab // 8:
            return perm


class RegimeStream:
    """Aligned regime-cycling stream.

    Corpus positions tile into blocks of `block` chars; odd blocks are
    emitted under `perm`. The training region is truncated to a multiple of
    2*block so wrapping preserves block parity, and lanes are spaced exactly
    2*block apart so every lane is in the same regime at every update — a
    staggered mixture would feed the weights a stationary blend and no
    coherent regime switch would ever reach the learner.

    `next_chunk` returns (inputs, targets, pos_in_block, parity), all
    [n_lanes, L]; targets share the input position's regime encoding.
    """

    def __init__(self, ids: torch.Tensor, n_lanes: int, block: int,
                 perm: torch.Tensor | None, device: str):
        period = 2 * block
        n_eff = (len(ids) // period) * period
        if n_eff < period:
            raise ValueError(f"corpus ({len(ids)}) shorter than one A/B period ({period})")
        if n_lanes * period > n_eff:
            raise ValueError("lanes would overlap within one period")
        self.ids = ids[:n_eff].to(device)
        self.n = n_eff
        self.block = block
        self.perm = perm.to(device) if perm is not None else None
        self.cursors = (torch.arange(n_lanes, dtype=torch.long) * period).to(device)
        self.device = device

    def next_chunk(self, length: int):
        offsets = torch.arange(length, device=self.device)
        idx = (self.cursors.unsqueeze(1) + offsets.unsqueeze(0)) % self.n
        raw_in = self.ids[idx]
        raw_tg = self.ids[(idx + 1) % self.n]
        pos = idx % self.block
        parity = (idx // self.block) % 2
        if self.perm is not None:
            permuted_in = self.perm[raw_in]
            permuted_tg = self.perm[raw_tg]
            inputs = torch.where(parity.bool(), permuted_in, raw_in)
            # a target is encoded under ITS OWN position's regime
            tg_parity = ((idx + 1) % self.n // self.block) % 2
            targets = torch.where(tg_parity.bool(), permuted_tg, raw_tg)
        else:
            inputs, targets = raw_in, raw_tg
        self.cursors = (self.cursors + length) % self.n
        return inputs, targets, pos, parity


def run_arm(kind: str, cycled: bool, cfg: FableConfig, out_dir: Path,
            device: str, block: int) -> dict:
    """Online training with per-token raw loss records."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    cfg = cfg.scaled(vocab_size=corpus.vocab_size, device=device)
    model = build_model(kind, cfg, device)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, build_schedule(cfg))
    perm = sample_permutation(corpus.vocab_size, seed=99) if cycled else None
    stream = RegimeStream(corpus.train_ids, cfg.batch_size, block, perm, device)
    state = model.initial_state(cfg.batch_size, device)

    rec_pos, rec_par, rec_loss = [], [], []
    skipped = 0
    consecutive = 0
    t0 = time.time()
    for update in range(1, cfg.updates + 1):
        tokens, targets, pos, parity = stream.next_chunk(cfg.chunk_length)
        losses = []
        for t in range(tokens.shape[1]):
            if isinstance(model, Fable):
                logits, state, _ = model.step(tokens[:, t], state)
            else:
                logits, state = model.step(tokens[:, t], state)
            losses.append(F.cross_entropy(logits, targets[:, t],
                                          reduction="none"))     # [lanes]
        per_token = torch.stack(losses, dim=1)                   # [lanes, T]
        loss = per_token.mean()
        state = state.detach() if isinstance(model, Fable) else state.detach()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
        if torch.isfinite(total_norm):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            consecutive = 0
        else:
            skipped += 1
            consecutive += 1
            if consecutive >= 200:
                raise SystemExit(f"[{kind}] aborted: {consecutive} consecutive "
                                 f"non-finite updates at u{update}")
        sched.step()

        rec_pos.append(pos.cpu().to(torch.int32))
        rec_par.append(parity.cpu().to(torch.uint8))
        rec_loss.append(per_token.detach().cpu().to(torch.float32))

        if update % cfg.log_every == 0:
            print(f"[{kind}{'/cycled' if cycled else '/aonly'}] u{update} "
                  f"bpc={float(loss.detach()) / LN2:.4f} skip={skipped} "
                  f"tok/s={update * cfg.batch_size * cfg.chunk_length / (time.time() - t0):.0f}",
                  flush=True)

    torch.save({"pos": torch.cat(rec_pos, 1), "parity": torch.cat(rec_par, 1),
                "loss": torch.cat(rec_loss, 1), "block": block,
                "config": dataclasses.asdict(cfg), "kind": kind,
                "cycled": cycled, "skipped": skipped},
               out_dir / "raw.pt")
    return {"skipped": skipped}


# ---------------------------------------------------------------- reporting

def analyze(raw_path: Path) -> dict:
    """Savings + adaptation curves from one arm's raw records.

    Only the second half of the run is scored (initial task learning must not
    be read as slow adaptation — exp1's discipline).
    """
    raw = torch.load(raw_path, weights_only=True)
    pos, parity, loss = raw["pos"], raw["parity"], raw["loss"]
    block = raw["block"]
    half = pos.shape[1] // 2
    pos, parity, loss = pos[:, half:], parity[:, half:], loss[:, half:]
    bpc = loss / LN2

    out: dict = {"kind": raw["kind"], "cycled": raw["cycled"],
                 "skipped": raw["skipped"]}

    # log-binned adaptation curve per regime
    curves = {}
    for reg in (0, 1) if raw["cycled"] else (0,):
        mask = parity == reg
        curve = []
        for lo, hi in zip(LOG_BIN_EDGES[:-1], LOG_BIN_EDGES[1:]):
            m = mask & (pos >= lo) & (pos < hi)
            curve.append(float(bpc[m].mean()) if int(m.sum()) else None)
        curves[f"regime_{reg}"] = curve
    out["adaptation_curves"] = curves
    out["bin_edges"] = LOG_BIN_EDGES

    # savings: per (lane, visit) — a visit starts when pos wraps backwards
    early_cut, late_cut = 2048, (3 * block) // 4
    savings: dict[int, list] = {}
    n_lanes = pos.shape[0]
    for lane in range(n_lanes):
        p, par, b = pos[lane], parity[lane], bpc[lane]
        starts = [0] + (torch.where(p[1:] < p[:-1])[0] + 1).tolist() + [len(p)]
        visit_index = {0: 0, 1: 0}
        for s, e in zip(starts[:-1], starts[1:]):
            reg = int(par[s])
            seg_pos, seg = p[s:e], b[s:e]
            if int(seg_pos[0]) > 0 or int(seg_pos[-1]) < block - 1:
                continue  # partial block at the scoring boundary
            early = seg[seg_pos < early_cut]
            steady = seg[seg_pos >= late_cut]
            if len(early) and len(steady):
                cost = float(early.mean() - steady.mean())
                savings.setdefault(reg, []).append(
                    {"lane": lane, "visit": visit_index[reg], "cost": cost,
                     "steady_bpc": float(steady.mean())})
            visit_index[reg] += 1
    out["visits"] = savings
    for reg, rows in savings.items():
        by_visit: dict[int, list] = {}
        for r in rows:
            by_visit.setdefault(r["visit"], []).append(r["cost"])
        out[f"savings_curve_regime_{reg}"] = {
            v: sum(c) / len(c) for v, c in sorted(by_visit.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one F2 adaptability arm")
    ap.add_argument("--kind", choices=("creature", "gru"), required=True)
    ap.add_argument("--stream", choices=("cycled", "aonly"), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--block", type=int, default=24576)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)
    out_dir = Path(args.out_dir)
    run_arm(args.kind, args.stream == "cycled", cfg, out_dir, args.device,
            args.block)
    summary = analyze(out_dir / "raw.pt")
    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=1))
    for reg in (0, 1):
        key = f"savings_curve_regime_{reg}"
        if key in summary:
            print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
