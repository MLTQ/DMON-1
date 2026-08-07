"""M0: paired mode-acquisition training of the Broca graft (language modes)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .backbone import MultiDepthBackbone
from .config import Fable2Config
from .modes import (
    MODES,
    ModeBank,
    load_mode_corpus,
    paired_mode_loss,
    run_mode_arm,
    sha_seed,
)
from .system import BrocaSystem, build_broca_system
from .train import (
    dtype_from_name,
    gradient_norm_or_raise,
    load_checkpoint,
    save_checkpoint,
    trainable_parameter_groups,
)

LN2 = math.log(2.0)

EVAL_ARMS = (
    "normal",
    "wrong_mode",
    "no_exposure",
    "memory_lesion",
    "internal_lesion",
    "bare_floor",
)


@torch.no_grad()
def evaluate_split(
    system: BrocaSystem, bank: ModeBank, split: str, *, demo_k: int
) -> dict:
    """Arm battery over item x exposure-direction episodes, curves-compatible.

    `mean_loss` is the negative mean log-likelihood of the *exposed-mode* twin,
    so lower is better and every downstream instrument (margins, strict wins,
    trend verdicts) reads exactly as in the G-line.
    """

    was_training = system.organism.training
    system.organism.eval()
    try:
        items = bank.split_items(split)
        episodes = [
            (item, mode) for item in items for mode in MODES
        ]
        report: dict = {"count": len(episodes)}
        losses: dict[str, list[float]] = {}
        for arm in EVAL_ARMS:
            arm_losses, margins, control = [], [], []
            for item, mode in episodes:
                sample_seed = sha_seed("eval", split, item.id, mode)
                score = run_mode_arm(
                    system, bank, item, mode, arm,
                    demo_k=demo_k, sample_seed=sample_seed,
                )
                arm_losses.append(-float(score["log_likelihoods"][mode].detach()))
                margins.append(float(score["margin"].detach()))
                control.append(score["control_rms"])
            if not arm_losses:
                raise RuntimeError(f"evaluation produced no scores for arm {arm!r}")
            losses[arm] = arm_losses
            report[arm] = {
                "mean_loss": sum(arm_losses) / len(arm_losses),
                "mean_bits": sum(arm_losses) / len(arm_losses) / LN2,
                "mean_mode_margin": sum(margins) / len(margins),
                "accuracy": sum(margin > 0 for margin in margins) / len(margins),
                "control_rms": sum(control) / len(control),
                "per_question_loss": arm_losses,
            }
        for arm in EVAL_ARMS:
            if arm == "normal":
                continue
            report[arm]["normal_strict_wins"] = sum(
                normal < other for normal, other in zip(losses["normal"], losses[arm])
            )
            report[arm]["normal_ties"] = sum(
                normal == other for normal, other in zip(losses["normal"], losses[arm])
            )
        report["depth_lesions"] = {}
        for depth in system.depths:
            depth_losses = []
            for item, mode in episodes:
                sample_seed = sha_seed("eval", split, item.id, mode)
                score = run_mode_arm(
                    system, bank, item, mode, "normal",
                    demo_k=demo_k, sample_seed=sample_seed,
                    lesion_depths=frozenset({depth}),
                )
                depth_losses.append(-float(score["log_likelihoods"][mode].detach()))
            report["depth_lesions"][str(depth)] = {
                "mean_loss": sum(depth_losses) / len(depth_losses),
                "lesion_hurts_count": sum(
                    lesioned > normal
                    for lesioned, normal in zip(depth_losses, losses["normal"])
                ),
            }
        return report
    finally:
        system.organism.train(was_training)


def run_training(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to train modes") from error

    cfg = Fable2Config().scaled(
        updates=args.updates,
        lr=args.lr,
        organ_lr_multiplier=args.organ_lr_multiplier,
        seed=args.seed,
        control_gain=args.control_gain,
        contrast_margin=args.contrast_margin,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )
    backbone = MultiDepthBackbone.from_pretrained(
        args.model,
        device=device,
        dtype=dtype_from_name(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    items, demos = load_mode_corpus(Path(args.corpus))
    bank = ModeBank(items, demos, tokenizer, backbone, device)
    if args.demo_k * 56 > cfg.n_memory:
        raise ValueError("demonstration stream cannot exceed the memory FIFO span")
    system = build_broca_system(backbone, cfg, device=device)
    optimizer = torch.optim.AdamW(
        trainable_parameter_groups(system, cfg), weight_decay=0.01
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "broca.pt"
    start_update = 0
    if args.resume and checkpoint_path.is_file():
        start_update = load_checkpoint(checkpoint_path, system, optimizer)

    train_items = bank.split_items("meta_train")
    telemetry: list[dict] = []
    evaluations: list[dict] = []
    order_generator = torch.Generator().manual_seed(cfg.seed + 11_000 + start_update)
    started = time.time()

    for update in range(start_update, cfg.updates):
        item = train_items[
            int(torch.randint(len(train_items), (1,), generator=order_generator))
        ]
        exposed_mode = MODES[update % 2]
        sample_seed = sha_seed("train", cfg.seed, update)
        with torch.no_grad():
            neutral = run_mode_arm(
                system, bank, item, exposed_mode, "no_exposure",
                demo_k=args.demo_k, sample_seed=sample_seed,
            )
        wrong = run_mode_arm(
            system, bank, item, exposed_mode, "wrong_mode",
            demo_k=args.demo_k, sample_seed=sample_seed,
        )
        exposed = run_mode_arm(
            system, bank, item, exposed_mode, "normal",
            demo_k=args.demo_k, sample_seed=sample_seed,
        )
        loss, advantages = paired_mode_loss(
            exposed, wrong, neutral, margin=cfg.contrast_margin
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = gradient_norm_or_raise(system, reject_above=cfg.reject_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            cfg.grad_clip,
        )
        optimizer.step()

        telemetry.append(
            {
                "update": update + 1,
                "question": f"{item.id}:{exposed_mode}",
                "loss": float(loss.detach()),
                "advantage_vs_no_exposure": float(advantages[0]),
                "advantage_vs_wrong_passage": float(advantages[1]),
                "wrong_vs_no_exposure": float(advantages[2]),
                "grad_norm": grad_norm,
                "control_rms": exposed["control_rms"],
                "per_depth_control_rms": exposed["per_depth_control_rms"],
            }
        )
        finished = update + 1
        if finished % cfg.eval_every == 0 or finished == cfg.updates:
            evaluations.append(
                {
                    "update": finished,
                    "development": evaluate_split(
                        system, bank, "development", demo_k=args.demo_k
                    ),
                    "heldout": evaluate_split(
                        system, bank, "heldout", demo_k=args.demo_k
                    ),
                }
            )
        if finished % cfg.checkpoint_every == 0 or finished == cfg.updates:
            save_checkpoint(checkpoint_path, system, optimizer, cfg, finished)
            save_checkpoint(
                out_dir / f"broca-u{finished}.pt", system, optimizer, cfg, finished
            )
            _write_result(
                out_dir, cfg, system, telemetry, evaluations, started, device
            )
    return _write_result(out_dir, cfg, system, telemetry, evaluations, started, device)


def _write_result(out_dir, cfg, system, telemetry, evaluations, started, device):
    result = {
        "schema": 1,
        "experiment": "m0-acquired-language-mode",
        "config": cfg.to_dict(),
        "depths": list(system.depths),
        "elapsed_seconds": time.time() - started,
        "telemetry": telemetry,
        "evaluations": evaluations,
    }
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--organ-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-gain", type=float, default=1.0)
    parser.add_argument("--contrast-margin", type=float, default=0.1)
    parser.add_argument("--demo-k", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
