"""Continuous online trainer for the streaming creature (S0)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .baselines import MatchedGRU, match_hidden_for_budget
from .config import TrainConfig
from .corpus import CharCorpus
from .model import StreamingCreature


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DMON-1 grok/ S0 streaming char LM")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n-cells", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n-dendrites", type=int, default=None)
    p.add_argument("--n-input", type=int, default=None)
    p.add_argument("--n-output", type=int, default=None)
    p.add_argument("--n-mirror", type=int, default=None)
    p.add_argument("--steps-per-token", type=int, default=None)
    p.add_argument("--truncate-every", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="grok/runs/latest")
    p.add_argument("--baseline", action="store_true", help="Also train matched GRU")
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--smoke-text", type=str, default="", help="Override corpus with this text")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    mapping = {
        "steps": "steps",
        "n_cells": "n_cells",
        "hidden": "hidden",
        "n_dendrites": "n_dendrites",
        "n_input": "n_input",
        "n_output": "n_output",
        "n_mirror": "n_mirror",
        "steps_per_token": "steps_per_token",
        "truncate_every": "truncate_every",
        "batch_size": "batch_size",
        "lr": "lr",
        "device": "device",
        "seed": "seed",
        "log_every": "log_every",
        "eval_every": "eval_every",
        "data_dir": "data_dir",
    }
    for arg_name, field in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(cfg, field, val)
    if args.no_attention:
        cfg.use_attention = False
    return cfg


@torch.no_grad()
def eval_window(
    model: StreamingCreature,
    corpus: CharCorpus,
    *,
    start: int,
    length: int,
    device: torch.device,
    warmup: int = 256,
) -> dict[str, float]:
    """Eval on a contiguous window after optional stream warmup (no loss during warmup)."""

    model.eval()
    warm_start = max(0, start - warmup)
    x, y = corpus.window(warm_start, warmup + length, device)
    state = model.initial_state(1, device)
    # Warm recurrent state on preceding tokens; do not score them.
    for i in range(min(warmup, x.shape[0])):
        _, state = model.step(x[i : i + 1], state)
    total = 0.0
    correct = 0
    scored = 0
    for i in range(warmup, x.shape[0]):
        logits, state = model.step(x[i : i + 1], state)
        loss = F.cross_entropy(logits, y[i : i + 1])
        total += float(loss.item())
        pred = int(logits.argmax(dim=-1).item())
        if pred == int(y[i].item()):
            correct += 1
        scored += 1
    nll = total / max(scored, 1)
    bpc = nll / math.log(2.0)
    model.train()
    return {"nll": nll, "bpc": bpc, "acc": correct / max(scored, 1), "scored": scored, "warmup": warmup}


def _train_online(
    *,
    name: str,
    step_fn,
    parameters,
    initial_state,
    corpus: CharCorpus,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[list[dict], object]:
    """Shared truncated-online loop for creature and baseline."""

    opt = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    corpus.reset_cursors(cfg.batch_size, seed=cfg.seed)
    state = initial_state
    history: list[dict] = []
    opt.zero_grad(set_to_none=True)

    run_loss = 0.0
    run_n = 0
    window_loss: torch.Tensor | None = None
    window_n = 0
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        x, y = corpus.next_batch(cfg.batch_size, device)
        logits, state = step_fn(x, state)
        loss = F.cross_entropy(logits, y)
        window_loss = loss if window_loss is None else window_loss + loss
        window_n += 1
        run_loss += float(loss.item())
        run_n += 1

        if step % cfg.truncate_every == 0:
            assert window_loss is not None
            (window_loss / window_n).backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            state = state.detach() if hasattr(state, "detach") else state.detach()
            window_loss = None
            window_n = 0

        if step % cfg.log_every == 0:
            nll = run_loss / max(run_n, 1)
            bpc = nll / math.log(2.0)
            elapsed = time.time() - t0
            tok = step * cfg.batch_size
            row = {
                "step": step,
                "nll": nll,
                "bpc": bpc,
                "tokens": tok,
                "tokens_per_s": tok / max(elapsed, 1e-6),
            }
            history.append(row)
            print(
                f"[{name}] step {step:6d}  nll={nll:.4f}  bpc={bpc:.4f}  "
                f"({row['tokens_per_s']:.0f} tok/s)"
            )
            run_loss = 0.0
            run_n = 0

    if window_n and window_loss is not None:
        (window_loss / window_n).backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
        opt.step()

    return history, state


def train(cfg: TrainConfig, *, out_dir: Path, baseline: bool, smoke_text: str) -> dict:
    cfg.validate()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    if smoke_text:
        corpus = CharCorpus.from_text(smoke_text)
    else:
        corpus = CharCorpus.from_shakespeare(cfg.data_dir)

    model = StreamingCreature(cfg, corpus.vocab_size).to(device)
    chance_nll = math.log(corpus.vocab_size)
    chance_bpc = chance_nll / math.log(2.0)

    print(
        f"creature params={model.count_parameters():,}  vocab={corpus.vocab_size}  "
        f"cells={cfg.n_cells}  hidden={cfg.hidden}  batch={cfg.batch_size}  "
        f"attn={cfg.use_attention}  chance_bpc={chance_bpc:.3f}  device={device}"
    )

    model.train()
    history, _ = _train_online(
        name="creature",
        step_fn=model.step,
        parameters=list(model.parameters()),  # materialize: generator is single-use
        initial_state=model.initial_state(cfg.batch_size, device),
        corpus=corpus,
        cfg=cfg,
        device=device,
    )

    metrics = eval_window(
        model,
        corpus,
        start=len(corpus.data) // 2,
        length=min(cfg.eval_tokens, len(corpus.data) - 1),
        device=device,
    )
    print(
        f"[creature] eval  nll={metrics['nll']:.4f}  bpc={metrics['bpc']:.4f}  "
        f"acc={metrics['acc']:.3f}"
    )

    result: dict = {
        "config": cfg.to_dict(),
        "params": model.count_parameters(),
        "vocab_size": corpus.vocab_size,
        "chance_bpc": chance_bpc,
        "history": history,
        "eval": metrics,
    }

    if baseline:
        gru_h = match_hidden_for_budget(corpus.vocab_size, model.count_parameters())
        gru = MatchedGRU(corpus.vocab_size, hidden=gru_h).to(device)
        print(f"gru hidden={gru_h} params={gru.count_parameters():,} (target ~{model.count_parameters():,})")
        gru.train()
        g_hist, _ = _train_online(
            name="gru",
            step_fn=gru.step,
            parameters=list(gru.parameters()),
            initial_state=gru.initial_state(cfg.batch_size, device),
            corpus=corpus,
            cfg=cfg,
            device=device,
        )
        result["baseline_gru"] = {
            "hidden": gru_h,
            "params": gru.count_parameters(),
            "history": g_hist,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab_size": corpus.vocab_size,
            "itos": corpus.itos,
        },
        ckpt,
    )
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {ckpt} and metrics.json")
    return result


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    if cfg.device == "cpu" and torch.cuda.is_available():
        # Prefer GPU when present unless user forced cpu.
        pass
    train(cfg, out_dir=Path(args.out_dir), baseline=args.baseline, smoke_text=args.smoke_text)


if __name__ == "__main__":
    main()
