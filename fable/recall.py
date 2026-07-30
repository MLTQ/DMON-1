"""F6: the pointer stream — cued associative recall under load.

Generator, online trainer, frozen-tape evaluation, and the conditional
freeze probe that asks the experiment's real question: when the task demands
state the ports cannot hold, does information move into the bulk?

See fable/experiments/f6-state-pressure.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import FableConfig
from .evaluate import LN2
from .model import Fable
from .stream import load_corpus
from .train import (add_config_args, build_model, build_optimizer,
                    build_schedule, config_from_args)

KEY_SENT, QUERY_SENT = "{", "}"
KEYS = "abcdefghijklmnopqrstuvwxyz"
DELAY_BUCKETS = [16, 32, 128, 512, 1025]
CHANCE_BITS = math.log2(len(KEYS))


def build_vocab(corpus_text_chars: list[str]) -> list[str]:
    extra = [c for c in (KEY_SENT, QUERY_SENT) if c not in corpus_text_chars]
    return corpus_text_chars + extra


def generate_lane(ids: torch.Tensor, start: int, n_chars: int, rng: random.Random,
                  stoi_key: dict, key_sent: int, query_sent: int,
                  n_pairs: int = 8, d_min: int = 16, d_max: int = 1024,
                  inject_prob: float = 0.06,
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (tape [n], recall_mask [n], delay [n], special [n]).

    recall_mask[i] is True iff the *loss at position i* (predicting tape[i+1])
    is a recall score — i.e. tape[i] is a query's key char and tape[i+1] is
    the value. delay[i] holds the emitted-char distance from that pair's
    injection.

    special[i] marks episode-overhead losses that are irreducibly
    unpredictable (sentinel timing, which key, the fresh value at inject) —
    excluded from "natural" so LM quality is not polluted by episode noise.
    """
    tape, mask, delay, special = [], [], [], []
    cursor = start % len(ids)
    live: dict[str, tuple[str, int]] = {}      # key -> (value, inject_pos)
    due: list[tuple[int, str]] = []            # (due_pos, key), kept sorted

    def emit(ch_id: int, m: bool = False, d: int = 0, sp: bool = False):
        tape.append(ch_id)
        mask.append(m)
        delay.append(d)
        special.append(sp)

    def mark_prev_special():
        if special:                            # loss predicting the sentinel
            special[-1] = True

    while len(tape) < n_chars:
        pos = len(tape)
        if due and due[0][0] <= pos:
            _, k = due.pop(0)
            v, injected = live.pop(k)
            mark_prev_special()
            emit(query_sent, sp=True)          # loss here predicts the key
            emit(stoi_key[k], m=True, d=pos + 1 - injected)
            emit(stoi_key[v], sp=True)         # loss here resumes natural text
            continue
        if len(live) < n_pairs and rng.random() < inject_prob:
            free = [k for k in KEYS if k not in live]
            k = rng.choice(free)
            v = rng.choice(KEYS)
            mark_prev_special()
            emit(key_sent, sp=True)
            emit(stoi_key[k], sp=True)         # loss here predicts fresh value
            emit(stoi_key[v], sp=True)
            live[k] = (v, len(tape) - 1)       # position of the value char
            d = int(math.exp(rng.uniform(math.log(d_min), math.log(d_max))))
            due.append((len(tape) + d, k))
            due.sort()
            continue
        emit(int(ids[cursor]))
        cursor = (cursor + 1) % len(ids)

    return (torch.tensor(tape[:n_chars], dtype=torch.long),
            torch.tensor(mask[:n_chars], dtype=torch.bool),
            torch.tensor(delay[:n_chars], dtype=torch.long),
            torch.tensor(special[:n_chars], dtype=torch.bool))


class PointerCorpus:
    """Corpus with sentinels appended to the vocabulary."""

    def __init__(self):
        base = load_corpus()
        self.itos = build_vocab(base.itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        remap = torch.tensor([self.stoi[c] for c in base.itos])
        self.train_ids = remap[base.train_ids]
        self.holdout_ids = remap[base.holdout_ids]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


def _step(model, tok, state):
    if isinstance(model, Fable):
        logits, state, _ = model.step(tok, state)
    else:
        logits, state = model.step(tok, state)
    return logits, state


def run_arm(kind: str, cfg: FableConfig, out_dir: Path, device: str,
            n_pairs: int, d_min: int, d_max: int,
            inject_prob: float = 0.06) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = PointerCorpus()
    cfg = cfg.scaled(vocab_size=corpus.vocab_size, device=device)
    model = build_model(kind, cfg, device)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, build_schedule(cfg))

    lanes = cfg.batch_size
    total = cfg.updates * cfg.chunk_length
    gen = random.Random(cfg.seed)
    tapes, masks, delays, specials = [], [], [], []
    stride = len(corpus.train_ids) // lanes
    for lane in range(lanes):
        t, m, d, sp = generate_lane(
            corpus.train_ids, lane * stride, total + 1, gen,
            corpus.stoi, corpus.stoi[KEY_SENT], corpus.stoi[QUERY_SENT],
            n_pairs, d_min, d_max, inject_prob)
        tapes.append(t), masks.append(m), delays.append(d), specials.append(sp)
    tape = torch.stack(tapes).to(device)
    # tape has total+1 chars (the final loss needs a target); losses have
    # exactly `total`, so the score masks are trimmed to match
    mask = torch.stack(masks)[:, :total]
    dly = torch.stack(delays)[:, :total]
    spc = torch.stack(specials)[:, :total]

    state = model.initial_state(lanes, device)
    rec_loss = torch.zeros(lanes, total)
    skipped, consecutive = 0, 0
    t0 = time.time()
    for update in range(1, cfg.updates + 1):
        s = (update - 1) * cfg.chunk_length
        e = s + cfg.chunk_length
        losses = []
        for i in range(s, e):
            logits, state = _step(model, tape[:, i], state)
            losses.append(F.cross_entropy(logits, tape[:, i + 1],
                                          reduction="none"))
        per_tok = torch.stack(losses, dim=1)
        loss = per_tok.mean()
        state = state.detach()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm = torch.norm(torch.stack([g.norm() for g in grads]))
        if torch.isfinite(norm):
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
        rec_loss[:, s:e] = per_tok.detach().cpu()
        if update % cfg.log_every == 0:
            half = mask[:, :e] & (torch.arange(e)[None] >= e // 2)
            rb = (rec_loss[:, :e][half].mean() / LN2) if int(half.sum()) else float("nan")
            print(f"[{kind}] u{update} bpc={float(loss.detach()) / LN2:.4f} "
                  f"recall_bpc={float(rb):.3f} skip={skipped} "
                  f"tok/s={update * lanes * cfg.chunk_length / (time.time() - t0):.0f}",
                  flush=True)

    torch.save({"loss": rec_loss, "mask": mask, "delay": dly, "special": spc,
                "config": dataclasses.asdict(cfg), "kind": kind,
                "skipped": skipped}, out_dir / "raw.pt")
    ckpt_cfg = model.cfg if isinstance(model, Fable) else cfg
    torch.save({"model": model.state_dict(),
                "config": dataclasses.asdict(ckpt_cfg)},
               out_dir / f"{kind}.pt")

    analysis = {"kind": kind, "skipped": skipped,
                "train": bucketize(rec_loss, mask, dly, spc, half_only=True)}
    analysis["eval"] = frozen_eval(model, corpus, cfg, device,
                                   n_pairs, d_min, d_max, inject_prob)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(json.dumps(analysis["eval"], indent=1), flush=True)


def bucketize(loss: torch.Tensor, mask: torch.Tensor, delay: torch.Tensor,
              special: torch.Tensor | None = None,
              half_only: bool = False) -> dict:
    out = {}
    n = loss.shape[1]
    scope = mask.clone()
    if half_only:
        scope &= torch.arange(n)[None] >= n // 2
    natural = ~mask if special is None else ~mask & ~special
    if half_only:
        natural = natural & (torch.arange(n)[None] >= n // 2)
    out["natural_bpc"] = float(loss[natural].mean() / LN2)
    out["recall_bpc_all"] = (float(loss[scope].mean() / LN2)
                             if int(scope.sum()) else None)
    out["n_recall"] = int(scope.sum())
    for lo, hi in zip(DELAY_BUCKETS[:-1], DELAY_BUCKETS[1:]):
        m = scope & (delay >= lo) & (delay < hi)
        out[f"recall_bpc_d{lo}_{hi - 1}"] = (
            float(loss[m].mean() / LN2) if int(m.sum()) else None)
        out[f"n_d{lo}_{hi - 1}"] = int(m.sum())
    return out


@torch.no_grad()
def _tape_losses(model, tape: torch.Tensor, device: str,
                 frozen_idx=None) -> torch.Tensor:
    state = model.initial_state(1, device)
    losses = []
    for i in range(len(tape) - 1):
        tok = tape[i].reshape(1).to(device)
        if isinstance(model, Fable):
            logits, state, _ = model.step(tok, state, frozen_idx=frozen_idx)
        else:
            logits, state = model.step(tok, state)
        losses.append(float(F.cross_entropy(logits,
                                            tape[i + 1].reshape(1).to(device))))
    return torch.tensor(losses).unsqueeze(0)


def frozen_eval(model, corpus: PointerCorpus, cfg: FableConfig, device: str,
                n_pairs: int, d_min: int, d_max: int,
                inject_prob: float = 0.06, tape_len: int = 16384) -> dict:
    """Held-out tape with fresh pairs; creature additionally probed with
    internal tissue frozen, split by position type."""
    gen = random.Random(cfg.seed + 777)
    tape, mask, delay, special = generate_lane(
        corpus.holdout_ids, 0, tape_len, gen, corpus.stoi,
        corpus.stoi[KEY_SENT], corpus.stoi[QUERY_SENT],
        n_pairs, d_min, d_max, inject_prob)
    model.eval()
    losses = _tape_losses(model, tape, device)
    m, d = mask[:-1].unsqueeze(0), delay[:-1].unsqueeze(0)
    sp = special[:-1].unsqueeze(0)
    out = {"normal": bucketize(losses, m, d, sp)}
    if isinstance(model, Fable):
        frozen = _tape_losses(model, tape, device,
                              frozen_idx=model.internal_idx)
        fr = bucketize(frozen, m, d, sp)
        out["freeze_internal"] = fr
        out["freeze_recall_delta"] = (
            fr["recall_bpc_all"] - out["normal"]["recall_bpc_all"]
            if fr["recall_bpc_all"] is not None else None)
        out["freeze_natural_delta"] = (fr["natural_bpc"]
                                       - out["normal"]["natural_bpc"])
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="F6 pointer-stream arm")
    ap.add_argument("--kind", choices=("creature", "gru", "transformer"),
                    required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--dmin", type=int, default=16)
    ap.add_argument("--dmax", type=int, default=1024)
    ap.add_argument("--inject-prob", type=float, default=0.06)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)
    run_arm(args.kind, cfg, Path(args.out_dir), args.device,
            args.pairs, args.dmin, args.dmax, args.inject_prob)


if __name__ == "__main__":
    main()
