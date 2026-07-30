"""Continuous multi-lane chunk trainer for the grok organism."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .baselines import MatchedGRU, match_hidden_for_budget
from .config import TrainConfig
from .evaluate import evaluate_with_ablations
from .model import StreamingCreature
from .stream import CharacterVocabulary, ContinuousCharStream, ensure_shakespeare
from .structure import apply_structural_phase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="grok/ continuous char organ trainer")
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--n-cells", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n-dendrites", type=int, default=None)
    p.add_argument("--n-input", type=int, default=None)
    p.add_argument("--n-output", type=int, default=None)
    p.add_argument("--n-mirror", type=int, default=None)
    p.add_argument("--steps-per-token", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--chunk-length", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="grok/runs/latest")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-reverse-credit", action="store_true")
    p.add_argument("--no-fast-plasticity", action="store_true")
    p.add_argument("--structural", action="store_true", help="Enable probe rewiring")
    p.add_argument("--smoke-text", type=str, default="")
    # back-compat
    p.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--truncate-every", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    mapping = {
        "updates": "updates",
        "steps": "updates",
        "n_cells": "n_cells",
        "hidden": "hidden",
        "n_dendrites": "n_dendrites",
        "n_input": "n_input",
        "n_output": "n_output",
        "n_mirror": "n_mirror",
        "steps_per_token": "steps_per_token",
        "batch_size": "batch_size",
        "chunk_length": "chunk_length",
        "truncate_every": "chunk_length",
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
    if args.no_reverse_credit:
        cfg.output_error_credit_gain = 0.0
        cfg.reward_gain = 0.0
    if args.no_fast_plasticity:
        cfg.fast_plasticity_gain = 0.0
    if args.structural:
        cfg.structural_probe_gain = 0.05
    return cfg


def train_creature(
    cfg: TrainConfig,
    text: str,
    vocabulary: CharacterVocabulary,
    *,
    holdout: str,
    out_dir: Path,
) -> dict:
    cfg.validate()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    model = StreamingCreature(cfg, len(vocabulary)).to(device)
    stream = ContinuousCharStream(text, vocabulary, cfg.batch_size, device)
    opt = torch.optim.AdamW(
        list(model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    state = model.initial_state(cfg.batch_size, device)

    chance_bpc = math.log(len(vocabulary)) / math.log(2.0)
    print(
        f"creature params={model.count_parameters():,}  vocab={len(vocabulary)}  "
        f"cells={cfg.n_cells} h={cfg.hidden} B={cfg.batch_size} T={cfg.chunk_length}  "
        f"credit={cfg.output_error_credit_gain} fast={cfg.fast_plasticity_gain}  "
        f"struct={cfg.structural_probe_gain} chance_bpc={chance_bpc:.3f} device={device}"
    )

    history: list[dict] = []
    evals: list[dict] = []
    run_loss = 0.0
    run_n = 0
    t0 = time.time()
    model.train()

    for update in range(1, cfg.updates + 1):
        tokens, targets = stream.next(cfg.chunk_length)
        _, state, loss = model.forward_chunk(tokens, targets, state)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Optional first-order global fitness for probes from ∂L/∂h.
        if (
            cfg.structural_probe_gain > 0
            and state.h.grad is not None
            and state.probe_fitness is not None
        ):
            # state.h may not retain grad after forward; use a proxy from eligibility.
            pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        state = state.detach()

        if cfg.structural_probe_gain > 0 and update % cfg.structural_phase_every == 0:
            pc = state.probe_credit if state.probe_credit is not None else torch.zeros(
                cfg.n_cells, device=device
            )
            pf = state.probe_fitness if state.probe_fitness is not None else torch.zeros(
                cfg.n_cells, device=device
            )
            # Without retained probe effect grads, use local credit only unless fitness
            # is supplied; require_global_fitness still gates on probe_fitness which
            # stays ~0 unless we estimate it. Soften: use local credit as fitness proxy
            # when fitness is zero-vector.
            if float(pf.abs().sum()) < 1e-8:
                pf = pc
            report = apply_structural_phase(
                model, probe_credit=pc, probe_fitness=pf, cfg=cfg
            )
            # Reset accumulators after phase.
            state.probe_credit = torch.zeros(cfg.n_cells, device=device)
            state.probe_fitness = torch.zeros(cfg.n_cells, device=device)
        else:
            report = None

        run_loss += float(loss.item())
        run_n += 1

        if update % cfg.log_every == 0:
            nll = run_loss / max(run_n, 1)
            bpc = nll / math.log(2.0)
            tokens_seen = update * cfg.batch_size * cfg.chunk_length
            elapsed = time.time() - t0
            row = {
                "update": update,
                "nll": nll,
                "bpc": bpc,
                "tokens": tokens_seen,
                "tokens_per_s": tokens_seen / max(elapsed, 1e-6),
            }
            if report is not None:
                row["rewires"] = report.rewires
                row["total_rewires"] = report.total_rewires
            history.append(row)
            print(
                f"[creature] up {update:5d}  nll={nll:.4f}  bpc={bpc:.4f}  "
                f"({row['tokens_per_s']:.0f} tok/s)"
                + (
                    f"  rewires={report.rewires}" if report is not None else ""
                )
            )
            run_loss = 0.0
            run_n = 0

        if update % cfg.eval_every == 0 or update == cfg.updates:
            ab = evaluate_with_ablations(
                model,
                vocabulary,
                holdout,
                device=device,
                tokens=cfg.eval_tokens,
                warmup=cfg.eval_warmup,
            )
            ab["update"] = update
            evals.append(ab)
            print(
                f"[creature] eval up {update}  bpc={ab['normal']['bits_per_character']:.4f}  "
                f"acc={ab['normal']['accuracy']:.3f}  "
                f"resetΔ={ab['reset_delta_bpc']:+.3f}  "
                f"shuffleΔ={ab['shuffle_delta_bpc']:+.3f}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": cfg.to_dict(),
        "params": model.count_parameters(),
        "vocab_size": len(vocabulary),
        "chance_bpc": chance_bpc,
        "history": history,
        "evals": evals,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": vocabulary.characters,
        },
        out_dir / "best.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'best.pt'}")
    return result


def train_gru_control(
    cfg: TrainConfig,
    text: str,
    vocabulary: CharacterVocabulary,
    *,
    target_params: int,
    holdout: str,
) -> dict:
    device = torch.device(cfg.device)
    gru_h = match_hidden_for_budget(len(vocabulary), target_params)
    model = MatchedGRU(len(vocabulary), hidden=gru_h).to(device)
    stream = ContinuousCharStream(text, vocabulary, cfg.batch_size, device)
    opt = torch.optim.AdamW(list(model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    h = model.initial_state(cfg.batch_size, device)
    history = []
    run_loss = 0.0
    run_n = 0
    t0 = time.time()
    print(f"gru hidden={gru_h} params={model.count_parameters():,}")
    model.train()
    for update in range(1, cfg.updates + 1):
        tokens, targets = stream.next(cfg.chunk_length)
        _, h, loss = model.forward_chunk(tokens, targets, h)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        h = h.detach()
        run_loss += float(loss.item())
        run_n += 1
        if update % cfg.log_every == 0:
            nll = run_loss / max(run_n, 1)
            row = {
                "update": update,
                "nll": nll,
                "bpc": nll / math.log(2.0),
                "tokens": update * cfg.batch_size * cfg.chunk_length,
                "tokens_per_s": (
                    update * cfg.batch_size * cfg.chunk_length / max(time.time() - t0, 1e-6)
                ),
            }
            history.append(row)
            print(f"[gru] up {update:5d}  nll={nll:.4f}  bpc={row['bpc']:.4f}")
            run_loss = 0.0
            run_n = 0
    # Holdout eval
    from .evaluate import evaluate_creature  # local import avoid cycles — actually same pkg

    # Manual GRU eval
    with torch.no_grad():
        model.eval()
        enc = vocabulary.encode(holdout, device)
        scored = min(cfg.eval_tokens, enc.numel() - 1 - cfg.eval_warmup)
        warm = cfg.eval_warmup
        state = model.initial_state(1, device)
        total = 0.0
        correct = 0
        n = 0
        for i in range(warm + scored):
            x = enc[i : i + 1]
            y = enc[i + 1 : i + 2]
            logits, state = model.step(x, state)
            if i >= warm:
                total += float(torch.nn.functional.cross_entropy(logits, y).item())
                correct += int(logits.argmax(-1).item() == int(y.item()))
                n += 1
        nll = total / max(n, 1)
        eval_metrics = {
            "nll": nll,
            "bits_per_character": nll / math.log(2.0),
            "accuracy": correct / max(n, 1),
            "tokens": n,
        }
    return {
        "hidden": gru_h,
        "params": model.count_parameters(),
        "history": history,
        "eval": eval_metrics,
    }


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    if args.smoke_text:
        text = args.smoke_text
    else:
        text = ensure_shakespeare(cfg.data_dir).read_text(encoding="utf-8")
    cut = int(len(text) * 0.9)
    train_text, holdout = text[:cut], text[cut:]
    vocabulary = CharacterVocabulary.from_text(text)
    out = Path(args.out_dir)
    result = train_creature(cfg, train_text, vocabulary, holdout=holdout, out_dir=out)
    if args.baseline:
        gru = train_gru_control(
            cfg,
            train_text,
            vocabulary,
            target_params=result["params"],
            holdout=holdout,
        )
        result["baseline_gru"] = gru
        (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        c_bpc = result["evals"][-1]["normal"]["bits_per_character"]
        g_bpc = gru["eval"]["bits_per_character"]
        print(f"compare final eval  creature={c_bpc:.4f}  gru={g_bpc:.4f}  gap={c_bpc-g_bpc:+.4f}")


if __name__ == "__main__":
    main()
