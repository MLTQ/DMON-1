"""Chunked-BPTT training on the continuous stream, for every arm.

One loop serves creature, matched GRU, and the head-only bypass control, so
"matched" means matched: same stream object, same chunking, same optimizer
family, same warmup+cosine schedule, same gradient guard, same eval cadence.

Stability machinery (the chase-1 lesson):
  - RMSNorm on messages (in the model) keeps per-micro-step signal scale fixed
  - warmup + cosine LR (sol S12's largest-gain fix, applied to BOTH arms)
  - non-finite gradient guard: a chunk whose gradients overflow is skipped and
    *counted*, never stepped — 128 sequential GRU applications can overflow
    fp32 before clipping, and clip_grad_norm_ of an inf norm poisons every
    parameter with NaN in one step (the likely chase-1 failure mode)
  - state health (max |h|, message RMS, logit scale, raw grad norm) logged
    next to BPC so a blowup is visible while it is still finite
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from .baselines import MatchedGRU, gru_param_count, match_hidden
from .config import FableConfig
from .evaluate import LN2, evaluate_model, evaluate_with_ablations
from .model import Fable, count_parameters
from .stream import LaneStream, load_corpus


def build_schedule(cfg: FableConfig):
    floor = cfg.lr_min / cfg.lr

    def factor(update: int) -> float:
        if update < cfg.warmup_updates:
            return (update + 1) / cfg.warmup_updates
        span = max(cfg.updates - cfg.warmup_updates, 1)
        progress = min((update - cfg.warmup_updates) / span, 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def build_optimizer(model: torch.nn.Module, cfg: FableConfig) -> torch.optim.AdamW:
    """Weight decay on genuine weight matrices only — not on biases, norms,
    embeddings, per-edge logits, or the sensory affine (grok decayed all of
    them, including double-decaying its duplicated edge logit, debt #9/#27)."""
    no_decay_names = ("embed", "logit", "in_gain", "in_bias", "norm")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(k in name for k in no_decay_names):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)


def build_model(kind: str, cfg: FableConfig, device: str):
    if kind == "creature":
        torch.manual_seed(cfg.seed)
        return Fable(cfg).to(device)
    if kind == "bypass":
        torch.manual_seed(cfg.seed)
        model = Fable(cfg).to(device)
        trainable = ("readout", "out_norm")
        for name, p in model.named_parameters():
            p.requires_grad = any(name.startswith(t) for t in trainable)
        return model
    if kind == "gru":
        torch.manual_seed(cfg.seed + 1000)
        target = count_parameters(Fable(FableConfig(**dataclasses.asdict(cfg))))
        hidden = match_hidden(cfg.vocab_size, target)
        model = MatchedGRU(cfg.vocab_size, hidden).to(device)
        return model
    raise ValueError(f"unknown kind {kind!r}")


def run(kind: str, cfg: FableConfig, out_dir: Path, device: str,
        corpus_path=None, on_update=None) -> dict:
    """Train one arm to completion; write <kind>.json and <kind>.pt.

    `on_update(update, model, opt, state) -> state | None` runs before each
    update — grow.py uses it for the F1 graft.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(corpus_path) if corpus_path else load_corpus()
    cfg = cfg.scaled(vocab_size=corpus.vocab_size, device=device)

    model = build_model(kind, cfg, device)
    params = count_parameters(model)
    opt = build_optimizer(model, cfg)
    schedule = build_schedule(cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    stream = LaneStream(corpus.train_ids, cfg.batch_size, cfg.seed, device)
    state = model.initial_state(cfg.batch_size, device)

    history, evals = [], []
    skipped_total = 0
    window_nll, window_tokens, t0 = 0.0, 0, time.time()
    health = None

    for update in range(1, cfg.updates + 1):
        if on_update is not None:
            migrated = on_update(update, model, opt, state)
            if migrated is not None:
                state = migrated
        tokens, targets = stream.next_chunk(cfg.chunk_length)
        loss, state, health = model.forward_chunk(tokens, targets, state)
        state = state.detach()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
        if torch.isfinite(total_norm):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        else:
            skipped_total += 1
        sched.step()

        window_nll += float(loss.detach()) * tokens.numel()
        window_tokens += tokens.numel()

        if update % cfg.log_every == 0:
            entry = {
                "update": update,
                "bpc": window_nll / max(window_tokens, 1) / LN2,
                "lr": sched.get_last_lr()[0],
                "tokens_per_s": window_tokens / max(time.time() - t0, 1e-9),
                "grad_norm": float(total_norm),
                "skipped_total": skipped_total,
            }
            if health is not None:
                entry.update(h_max=health.h_max, msg_rms=health.msg_rms,
                             logit_absmax=health.logit_absmax)
            history.append(entry)
            print(f"[{kind}] u{update} bpc={entry['bpc']:.4f} "
                  f"lr={entry['lr']:.2e} tok/s={entry['tokens_per_s']:.0f} "
                  f"gnorm={entry['grad_norm']:.2f} skip={skipped_total}"
                  + (f" hmax={health.h_max:.2f} mrms={health.msg_rms:.2f}"
                     if health else ""), flush=True)
            window_nll, window_tokens, t0 = 0.0, 0, time.time()

        if update % cfg.eval_every == 0 or update == cfg.updates:
            model.eval()
            if kind == "gru":
                ev = {"normal": evaluate_model(
                    model, corpus.holdout_ids, cfg.eval_warmup_tokens,
                    cfg.eval_tokens, device)}
            else:
                ev = evaluate_with_ablations(
                    model, corpus.holdout_ids, cfg.eval_warmup_tokens,
                    cfg.eval_tokens, device)
            model.train()
            ev["update"] = update
            evals.append(ev)
            print(f"[{kind}] eval u{update} "
                  f"holdout_bpc={ev['normal']['bits_per_character']:.4f}",
                  flush=True)

    result = {
        "kind": kind, "params": params,
        "config": dataclasses.asdict(cfg),
        "history": history, "evals": evals,
        "final_eval": evals[-1] if evals else None,
        "skipped_updates_total": skipped_total,
    }
    if kind == "gru":
        result["gru_hidden"] = model.hidden
        result["gru_params_closed_form"] = gru_param_count(cfg.vocab_size, model.hidden)
    (out_dir / f"{kind}.json").write_text(json.dumps(result, indent=1))
    # checkpoint the model's own config, not the run's initial one — a grown
    # model's field is larger than the config this run started from
    ckpt_cfg = model.cfg if isinstance(model, Fable) else cfg
    torch.save({"model": model.state_dict(),
                "config": dataclasses.asdict(ckpt_cfg)}, out_dir / f"{kind}.pt")
    return result


def add_config_args(ap: argparse.ArgumentParser) -> None:
    defaults = FableConfig()
    for field in dataclasses.fields(FableConfig):
        if field.name in ("vocab_size", "device"):
            continue
        flag = "--" + field.name.replace("_", "-")
        ap.add_argument(flag, type=type(getattr(defaults, field.name)),
                        default=getattr(defaults, field.name))


def config_from_args(args: argparse.Namespace) -> FableConfig:
    kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(FableConfig)
              if f.name not in ("vocab_size", "device") and hasattr(args, f.name)}
    return FableConfig(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one fable arm")
    ap.add_argument("--model", choices=("creature", "gru", "bypass"),
                    default="creature")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="fable/runs/dev")
    add_config_args(ap)
    args = ap.parse_args()
    run(args.model, config_from_args(args), Path(args.out_dir), args.device)


if __name__ == "__main__":
    main()
