"""One resumable continuous trainer for SOL2 and all matched controls."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .baselines import (
    MatchedGRU,
    MatchedTransformer,
    match_gru_hidden,
    match_transformer_hidden,
)
from .checkpoint import load_checkpoint, restore_rng_state, save_checkpoint, unpack_state
from .config import Sol2Config
from .evaluate import evaluate_model, evaluate_with_ablations
from .model import Sol2, count_parameters
from .optim import build_optimizer, guarded_step, set_learning_rate
from .state import OrganismState
from .stream import LaneStream, load_corpus

LN2 = math.log(2.0)


def build_model(kind: str, cfg: Sol2Config, device: str):
    if kind == "creature":
        torch.manual_seed(cfg.seed)
        return Sol2(cfg).to(device)
    torch.manual_seed(cfg.seed + 10_000)
    target = count_parameters(Sol2(cfg))
    if kind == "gru":
        return MatchedGRU(
            cfg.vocab_size, match_gru_hidden(cfg.vocab_size, target)
        ).to(device)
    if kind == "transformer":
        hidden = match_transformer_hidden(
            cfg.vocab_size,
            target,
            max_len=cfg.transformer_seq_len,
        )
        return MatchedTransformer(
            cfg.vocab_size,
            hidden,
            max_len=cfg.transformer_seq_len,
        ).to(device)
    raise ValueError(f"unknown model kind {kind!r}")


def _detach_state(state):
    return state.detach()


def _set_state_version(state, version: int):
    if isinstance(state, OrganismState):
        state.weight_version = version


def _write_summary(
    path: Path,
    *,
    kind: str,
    cfg: Sol2Config,
    params: int,
    update: int,
    accepted: int,
    rejected: int,
    history: list[dict],
    evaluations: list[dict],
) -> None:
    payload = {
        "kind": kind,
        "config": cfg.to_dict(),
        "params": params,
        "update": update,
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "history": history,
        "evaluations": evaluations,
    }
    path.write_text(json.dumps(payload, indent=2))


def run(
    kind: str,
    cfg: Sol2Config,
    out_dir: Path,
    device: str,
    *,
    corpus_path: Path | str | None = None,
    resume: Path | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(corpus_path) if corpus_path else load_corpus()

    checkpoint_payload = load_checkpoint(resume, device) if resume else None
    if checkpoint_payload is not None:
        kind = checkpoint_payload["kind"]
        cfg = Sol2Config.from_dict(checkpoint_payload["config"])
        if cfg.device != device:
            cfg = cfg.scaled(device=device)
    else:
        cfg = cfg.scaled(vocab_size=corpus.vocab_size, device=device)

    model = build_model(kind, cfg, device)
    params = count_parameters(model)
    optimizer = build_optimizer(model, cfg)

    batch = cfg.batch_size
    chunk = cfg.chunk_length
    if kind == "transformer":
        chunk = cfg.transformer_seq_len
        batch = max(1, cfg.batch_size * cfg.chunk_length // chunk)
    stream = LaneStream(corpus.train_ids, batch, cfg.seed, device)
    state = model.initial_state(batch, device)
    start_update = 0
    accepted_updates = 0
    rejected_updates = 0
    history: list[dict] = []
    evaluations: list[dict] = []

    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model"])
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        state = unpack_state(checkpoint_payload["state"], device)
        stream.load_state_dict(checkpoint_payload["stream"])
        start_update = int(checkpoint_payload["update"])
        accepted_updates = int(checkpoint_payload.get("accepted_updates", start_update))
        rejected_updates = int(checkpoint_payload.get("rejected_updates", 0))
        history = list(checkpoint_payload.get("history", []))
        evaluations = list(checkpoint_payload.get("evaluations", []))
        restore_rng_state(checkpoint_payload)

    model.train()
    window_nll = 0.0
    window_tokens = 0
    window_started = time.time()
    consecutive_rejections = 0
    latest_update = start_update

    for update in range(start_update + 1, cfg.updates + 1):
        latest_update = update
        learning_rate = set_learning_rate(optimizer, cfg, update)
        inputs, targets = stream.next_chunk(chunk)
        collect_health = kind == "creature" and update % cfg.log_every == 0
        loss, next_state, step_health = model.forward_chunk(
            inputs, targets, state, collect_health=collect_health
        )
        next_state = _detach_state(next_state)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        update_health = guarded_step(model, optimizer, cfg)
        if update_health.accepted:
            accepted_updates += 1
            consecutive_rejections = 0
        else:
            rejected_updates += 1
            consecutive_rejections += 1
        _set_state_version(next_state, accepted_updates)
        state = next_state

        if consecutive_rejections >= 200:
            raise RuntimeError(
                f"{kind} reached 200 consecutive rejected gradients at update {update}"
            )

        tokens_seen = inputs.numel()
        window_nll += float(loss.detach()) * tokens_seen
        window_tokens += tokens_seen

        if update % cfg.log_every == 0 or update == cfg.updates:
            entry = {
                "update": update,
                "train_bpc": window_nll / max(window_tokens, 1) / LN2,
                "lr": learning_rate,
                "tokens_per_second": window_tokens
                / max(time.time() - window_started, 1e-9),
                "gradient_norm": update_health.total_norm,
                "gradient_modules": update_health.module_norms,
                "cell_gradient_utilization": update_health.cell_gradient_utilization,
                "accepted_updates": accepted_updates,
                "rejected_updates": rejected_updates,
            }
            if step_health is not None:
                entry["health"] = {
                    "hidden_absmax": step_health.hidden_absmax,
                    "message_rms": step_health.message_rms,
                    "message_absmax": step_health.message_absmax,
                    "logit_absmax": step_health.logit_absmax,
                    "alpha_mean": step_health.alpha_mean,
                    "operator_norm_max": step_health.operator_norm_max,
                    "organ_attention_entropy": step_health.organ_attention_entropy,
                    "organ_attention_max": step_health.organ_attention_max,
                    "organ_effective_cells": step_health.organ_effective_cells,
                }
            history.append(entry)
            print(
                f"[{kind}] u{update} bpc={entry['train_bpc']:.4f} "
                f"grad={entry['gradient_norm']:.2f} rejects={rejected_updates}",
                flush=True,
            )
            window_nll = 0.0
            window_tokens = 0
            window_started = time.time()

        if update % cfg.eval_every == 0 or update == cfg.updates:
            model.eval()
            if kind == "creature":
                metrics = evaluate_with_ablations(
                    model,
                    corpus.holdout_ids,
                    cfg.eval_warmup_tokens,
                    cfg.eval_tokens,
                    device,
                )
            else:
                metrics = {
                    "normal": evaluate_model(
                        model,
                        corpus.holdout_ids,
                        cfg.eval_warmup_tokens,
                        cfg.eval_tokens,
                        device,
                    )
                }
            evaluations.append({"update": update, **metrics})
            model.train()
            print(
                f"[{kind}] eval u{update} "
                f"bpc={metrics['normal']['bits_per_character']:.4f}",
                flush=True,
            )
            _write_summary(
                out_dir / "metrics.json",
                kind=kind,
                cfg=cfg,
                params=params,
                update=update,
                accepted=accepted_updates,
                rejected=rejected_updates,
                history=history,
                evaluations=evaluations,
            )
            save_checkpoint(
                out_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                state=state,
                stream=stream,
                cfg=cfg,
                kind=kind,
                update=update,
                accepted_updates=accepted_updates,
                rejected_updates=rejected_updates,
                history=history,
                evaluations=evaluations,
            )

    return {
        "kind": kind,
        "params": params,
        "update": latest_update,
        "accepted_updates": accepted_updates,
        "rejected_updates": rejected_updates,
        "history": history,
        "evaluations": evaluations,
        "config": cfg.to_dict(),
    }


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--updates", type=int, default=24_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--compute-cells", type=int, default=64)
    parser.add_argument("--relay-cells", type=int, default=16)
    parser.add_argument("--steps-per-token", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument(
        "--bounded-operators", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cell-identity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-tokens", type=int, default=2_048)


def config_from_arguments(args: argparse.Namespace) -> Sol2Config:
    return Sol2Config(
        updates=args.updates,
        batch_size=args.batch_size,
        chunk_length=args.chunk_length,
        hidden=args.hidden,
        n_compute=args.compute_cells,
        n_relay=args.relay_cells,
        steps_per_token=args.steps_per_token,
        lr=args.lr,
        lr_min=args.lr_min,
        schedule=args.schedule,
        bounded_operators=args.bounded_operators,
        cell_identity=args.cell_identity,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_tokens=args.eval_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one SOL2 comparison arm")
    parser.add_argument("--model", choices=("creature", "gru", "transformer"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--resume", type=Path)
    add_config_arguments(parser)
    args = parser.parse_args()
    run(
        args.model,
        config_from_arguments(args),
        args.out_dir,
        args.device,
        corpus_path=args.corpus,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
