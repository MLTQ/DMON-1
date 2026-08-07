"""G1: paired-behavior training of the Broca graft on the wiki corpus."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from sol2.wiki_memory import load_wiki_memory_corpus, verify_wiki_sources

from .backbone import MultiDepthBackbone
from .config import Fable2Config
from .episodes import EVAL_ARMS, EpisodeBank, paired_contrast_loss, run_arm, run_depth_lesion
from .system import BrocaSystem, build_broca_system

LN2 = math.log(2.0)


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(table)}")
    return table[name]


def trainable_parameter_groups(system: BrocaSystem, cfg: Fable2Config) -> list[dict]:
    """Two groups: the organism kernel, and the Broca organ at its own rate."""

    organ = system.organism.attached_organs[system.organ_name]
    organ_ids = {id(parameter) for parameter in organ.parameters()}
    organism_params = [
        parameter
        for parameter in system.organism.parameters()
        if id(parameter) not in organ_ids and parameter.requires_grad
    ]
    organ_params = [p for p in organ.parameters() if p.requires_grad]
    if not organism_params or not organ_params:
        raise ValueError("both organism and organ must contribute trainable parameters")
    return [
        {"params": organism_params, "lr": cfg.lr, "name": "organism"},
        {"params": organ_params, "lr": cfg.lr * cfg.organ_lr_multiplier, "name": "organ"},
    ]


def gradient_norm_or_raise(system: BrocaSystem, *, reject_above: float) -> float:
    """Global gradient norm; raises on non-finite values or a rejected magnitude."""

    total = 0.0
    for name, parameter in system.organism.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"non-finite gradient in {name}")
        total += float(parameter.grad.detach().float().pow(2).sum())
    norm = math.sqrt(total)
    if norm > reject_above:
        raise RuntimeError(
            f"gradient norm {norm:.1f} above rejection bound {reject_above}"
        )
    return norm


@torch.no_grad()
def evaluate_split(system: BrocaSystem, bank: EpisodeBank, split: str) -> dict:
    """Full arm battery with per-question losses, strict wins, and depth lesions.

    Runs in eval mode: EffectiveLinear's power-iteration buffer advances during
    training-mode forwards, so a training-mode evaluation would make measured
    numbers depend on arm order.
    """

    was_training = system.organism.training
    system.organism.eval()
    try:
        return _evaluate_split_inner(system, bank, split)
    finally:
        system.organism.train(was_training)


def _evaluate_split_inner(system: BrocaSystem, bank: EpisodeBank, split: str) -> dict:
    items = bank.split_items(split)
    report: dict = {"count": len(items)}
    losses: dict[str, list[float]] = {}
    for arm in EVAL_ARMS:
        arm_losses, correct, control = [], 0, []
        for item in items:
            score = run_arm(system, bank, item, arm)
            arm_losses.append(float(score.loss.detach()))
            correct += int(score.correct)
            control.append(score.control_rms)
        if not arm_losses:
            raise RuntimeError(f"evaluation produced no scores for arm {arm!r}")
        losses[arm] = arm_losses
        report[arm] = {
            "mean_loss": sum(arm_losses) / len(arm_losses),
            "mean_bits": sum(arm_losses) / len(arm_losses) / LN2,
            "accuracy": correct / len(items),
            "control_rms": sum(control) / len(control),
            "per_question_loss": arm_losses,
        }
    for arm in EVAL_ARMS:
        if arm == "normal":
            continue
        wins = sum(
            normal < other for normal, other in zip(losses["normal"], losses[arm])
        )
        ties = sum(
            normal == other for normal, other in zip(losses["normal"], losses[arm])
        )
        report[arm]["normal_strict_wins"] = wins
        report[arm]["normal_ties"] = ties

    report["depth_lesions"] = {}
    for depth in system.depths:
        depth_losses = [
            float(run_depth_lesion(system, bank, item, depth).loss.detach())
            for item in items
        ]
        report["depth_lesions"][str(depth)] = {
            "mean_loss": sum(depth_losses) / len(depth_losses),
            "lesion_hurts_count": sum(
                lesioned > normal
                for lesioned, normal in zip(depth_losses, losses["normal"])
            ),
        }
    return report


def save_checkpoint(
    path: Path,
    system: BrocaSystem,
    optimizer: torch.optim.Optimizer,
    cfg: Fable2Config,
    update: int,
) -> None:
    payload = {
        "schema": 1,
        "config": cfg.to_dict(),
        "depths": list(system.depths),
        "update": update,
        "organism": system.organism.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, system: BrocaSystem, optimizer: torch.optim.Optimizer
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != 1:
        raise ValueError("unsupported fable2 checkpoint schema")
    if tuple(payload["depths"]) != system.depths:
        raise ValueError("checkpoint depths do not match the current system")
    system.organism.load_state_dict(payload["organism"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if "cuda_rng" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["update"])


def run_training(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to train the Broca graft") from error

    cfg = Fable2Config().scaled(
        updates=args.updates,
        lr=args.lr,
        organ_lr_multiplier=args.organ_lr_multiplier,
        seed=args.seed,
        control_gain=args.control_gain,
        depth_fractions=tuple(float(f) for f in args.depths.split(",")),
        contrast_margin=args.contrast_margin,
        task_weight=args.task_weight,
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
    corpus = load_wiki_memory_corpus(Path(args.corpus))
    if not args.skip_source_verification:
        verify_wiki_sources(corpus)
    bank = EpisodeBank(
        corpus,
        tokenizer,
        backbone,
        device,
        permutation_seed=cfg.permutation_seed,
        max_memory_tokens=cfg.max_memory_tokens,
        max_question_tokens=cfg.max_question_tokens,
    )
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
    order_generator = torch.Generator().manual_seed(cfg.seed + 9_000 + start_update)
    started = time.time()

    for update in range(start_update, cfg.updates):
        item = train_items[
            int(torch.randint(len(train_items), (1,), generator=order_generator))
        ]
        with torch.no_grad():
            neutral = run_arm(system, bank, item, "no_exposure")
        # The wrong-passage arm stays live so common-mode target inflation cancels
        # exactly in the differential term; see paired_contrast_loss.
        wrong = run_arm(system, bank, item, "wrong_passage")
        exposed = run_arm(system, bank, item, "normal")
        loss, advantages = paired_contrast_loss(
            exposed, wrong, neutral, margin=cfg.contrast_margin
        )
        if cfg.task_weight > 0:
            loss = loss + cfg.task_weight * exposed.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = gradient_norm_or_raise(
            system, reject_above=cfg.reject_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            cfg.grad_clip,
        )
        optimizer.step()

        telemetry.append(
            {
                "update": update + 1,
                "question": item.question_id,
                "loss": float(loss.detach()),
                "advantage_vs_no_exposure": float(advantages[0]),
                "advantage_vs_wrong_passage": float(advantages[1]),
                "wrong_vs_no_exposure": float(advantages[2]),
                "grad_norm": grad_norm,
                "control_rms": exposed.control_rms,
                "per_depth_control_rms": list(exposed.per_depth_control_rms),
                "internal_activity": exposed.internal_activity,
                "memory_activity": exposed.memory_activity,
            }
        )
        finished_update = update + 1
        if finished_update % cfg.eval_every == 0 or finished_update == cfg.updates:
            evaluation = {
                "update": finished_update,
                "development": evaluate_split(system, bank, "development"),
                "heldout": evaluate_split(system, bank, "heldout"),
            }
            evaluations.append(evaluation)
        if finished_update % cfg.checkpoint_every == 0 or finished_update == cfg.updates:
            save_checkpoint(checkpoint_path, system, optimizer, cfg, finished_update)
            # Retain a per-eval copy: G1's best-update model (u150) was destroyed
            # by its own degrading tail because the rolling checkpoint overwrote it.
            save_checkpoint(
                out_dir / f"broca-u{finished_update}.pt",
                system,
                optimizer,
                cfg,
                finished_update,
            )
            _write_result(out_dir, cfg, system, telemetry, evaluations, started, device)

    return _write_result(out_dir, cfg, system, telemetry, evaluations, started, device)


def _write_result(
    out_dir: Path,
    cfg: Fable2Config,
    system: BrocaSystem,
    telemetry: list[dict],
    evaluations: list[dict],
    started: float,
    device: torch.device,
) -> dict:
    result = {
        "schema": 1,
        "experiment": "g1-broca-paired-behavior",
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
    parser.add_argument(
        "--corpus", default="sol2/experiments/l0c1-wiki-memory-corpus.json"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-source-verification", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--organ-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-gain", type=float, default=1.0)
    parser.add_argument("--depths", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--contrast-margin", type=float, default=0.1)
    parser.add_argument("--task-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
