"""G0: interface audit of the Broca graft against a frozen backbone.

Runs before any training (the one-second check before the forty-minute sweep):
exact zero-init no-op, per-depth label sensitivity to bounded random controls,
and two-step gradient recruitment through the paired loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol2.wiki_memory import load_wiki_memory_corpus, verify_wiki_sources

from .backbone import MultiDepthBackbone
from .config import Fable2Config
from .episodes import EpisodeBank, paired_contrast_loss, run_arm
from .system import BrocaSystem, build_broca_system
from .train import dtype_from_name


def gradient_group_norms(system: BrocaSystem) -> dict[str, float]:
    """Squared-sum gradient norms bucketed by anatomical group."""

    prefix = f"attached_organs.{system.organ_name}."
    groups: dict[str, float] = {}
    for name, parameter in system.organism.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().pow(2).sum())
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix.startswith("depth_heads."):
                key = f"organ.head.{suffix.split('.')[1]}"
            elif suffix.startswith("depth_bases"):
                key = "organ.bases"
            elif suffix.startswith("trunk."):
                key = "organ.trunk"
            elif suffix.startswith("sensor") or suffix.startswith("sensory"):
                key = "organ.sensor"
            else:
                key = "organ.other"
        elif name.startswith("tissues."):
            key = "organism.tissues"
        elif name.startswith("graph."):
            key = "organism.graph"
        elif name.startswith("cell_"):
            key = "organism.expression"
        else:
            key = "organism.other"
        groups[key] = groups.get(key, 0.0) + value
    return {key: value**0.5 for key, value in groups.items()}


def backbone_gradient_tensors(backbone: torch.nn.Module) -> int:
    return sum(
        1 for parameter in backbone.parameters() if parameter.grad is not None
    )


def run_audit(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to audit the Broca graft") from error

    cfg = Fable2Config().scaled(
        seed=args.seed,
        control_gain=args.control_gain,
        depth_fractions=tuple(float(f) for f in args.depths.split(",")),
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
    items = bank.split_items("development")

    report: dict = {
        "schema": 1,
        "experiment": "g0-broca-interface-audit",
        "model": args.model,
        "config": cfg.to_dict(),
        "n_layers": backbone.n_layers,
        "depths": list(system.depths),
        "native_logit_error": float(
            backbone.native_logit_error(items[0].question_ids)
        ),
    }

    # Gate 1: a fresh graft is exactly silent on the frozen model. This runs with
    # gradients enabled on purpose: the no-grad path short-circuits zero controls
    # by contract, so only the live-graph math path can prove the claim.
    noop_deltas = []
    for item in items:
        with torch.no_grad():
            bare = backbone.bare_logits(item.question_ids)[:, -1]
        state = system.initial_state(1, bank.device)
        state = system.observe_feature_sequence(item.exposure_features, state)
        controls, _ = system._evolve(
            item.question_features, state, write_memory=False
        )
        injected = backbone.injected_logits(
            item.question_ids, system.depth_controls(controls)
        )[:, -1]
        noop_deltas.append(float((injected.detach() - bare).abs().max()))
    report["zero_init_noop_max_delta"] = max(noop_deltas)
    report["gate_zero_init_noop"] = max(noop_deltas) == 0.0

    # Gate 2: bounded random controls at each depth alone move the label logits.
    sensitivity: dict[str, float] = {}
    generator = torch.Generator().manual_seed(cfg.seed + 31_000)
    with torch.no_grad():
        for depth in system.depths:
            deltas = []
            for item in items:
                bare = backbone.bare_logits(item.question_ids)[:, -1]
                random_bank = cfg.control_gain * torch.tanh(
                    torch.randn(
                        1,
                        cfg.n_control_tokens,
                        backbone.width,
                        generator=generator,
                    )
                ).to(device)
                injected = backbone.injected_logits(
                    item.question_ids, {depth: random_bank}
                )[:, -1]
                label_delta = (
                    (injected.index_select(-1, bank.labels) - bare.index_select(-1, bank.labels))
                    .float()
                    .abs()
                    .mean()
                )
                deltas.append(float(label_delta))
            sensitivity[str(depth)] = sum(deltas) / len(deltas)
    report["per_depth_label_sensitivity"] = sensitivity
    positive = [value for value in sensitivity.values() if value > 0.0]
    report["gate_depth_sensitivity"] = len(positive) == len(system.depths)
    if positive:
        report["depth_sensitivity_ratio"] = max(positive) / max(min(positive), 1e-12)

    # Gate 3: gradient recruitment. At exact zero-init only the depth heads can
    # receive gradient; after one optimizer step the trunk, sensor, and organism
    # must come live. Both stages must leave the backbone with zero grad tensors.
    optimizer = torch.optim.AdamW(
        [p for p in system.organism.parameters() if p.requires_grad], lr=1e-3
    )
    stages = {}
    for stage in ("init", "after_one_step"):
        item = items[0]
        with torch.no_grad():
            neutral = run_arm(system, bank, item, "no_exposure")
        wrong = run_arm(system, bank, item, "wrong_passage")
        exposed = run_arm(system, bank, item, "normal")
        loss, _ = paired_contrast_loss(
            exposed, wrong, neutral, margin=cfg.contrast_margin
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        stages[stage] = gradient_group_norms(system)
        stages[stage]["backbone_grad_tensors"] = backbone_gradient_tensors(backbone)
        if stage == "init":
            optimizer.step()
    report["gradient_stages"] = stages
    head_keys = [f"organ.head.{i}" for i in range(len(system.depths))]
    report["gate_init_heads_live"] = all(
        stages["init"].get(key, 0.0) > 0.0 for key in head_keys
    )
    report["gate_recruited_after_one_step"] = all(
        stages["after_one_step"].get(key, 0.0) > 0.0
        for key in ("organ.trunk", "organ.sensor", "organism.tissues", "organism.graph")
    )
    report["gate_backbone_frozen"] = all(
        stages[stage]["backbone_grad_tensors"] == 0 for stage in stages
    )

    report["gates_all_pass"] = all(
        report[key]
        for key in (
            "gate_zero_init_noop",
            "gate_depth_sensitivity",
            "gate_init_heads_live",
            "gate_recruited_after_one_step",
            "gate_backbone_frozen",
        )
    )
    if device.type == "cuda":
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k.startswith("gate")}, indent=2))
    return report


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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-gain", type=float, default=1.0)
    parser.add_argument("--depths", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
