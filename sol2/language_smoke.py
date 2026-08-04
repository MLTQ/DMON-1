"""One-update pretrained-backbone smoke test for the DMON-L0 language graft."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import Sol2Config
from .language_backbone import HuggingFaceFrozenBackbone
from .living_language import graft_language_backbone
from .model import Sol2, count_parameters


def dtype_from_name(name: str) -> torch.dtype | str:
    values: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"unsupported dtype {name!r}")
    return values[name]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_smoke(args: argparse.Namespace) -> dict:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to run the language smoke") from error

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    backbone = HuggingFaceFrozenBackbone.from_pretrained(
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
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("smoke prompt must tokenize to at least two tokens")

    cfg = Sol2Config(
        n_input=args.input_cells,
        n_memory=args.memory_cells,
        n_compute=args.compute_cells,
        n_relay=args.relay_cells,
        n_output=args.output_cells,
        hidden=args.organism_hidden,
        n_dendrites=args.dendrites,
        initial_active_dendrites=args.active_dendrites,
        steps_per_token=args.microsteps,
        organ_queries=args.organ_queries,
        # The legacy discrete A port is unused by the continuous language graft.
        vocab_size=2,
        seed=args.seed,
        device=str(device),
    )
    torch.manual_seed(args.seed)
    organism = Sol2(cfg).to(device)
    system = graft_language_backbone(
        organism,
        backbone,
        n_control_tokens=args.control_tokens,
        control_rank=args.control_rank,
        seed=args.seed + 1,
    )
    prefix = input_ids[:, :-1]
    target = input_ids[:, -1]
    native_error = backbone.native_logit_error(prefix)
    if native_error > args.parity_tolerance:
        raise RuntimeError(
            f"generic output head differs from native logits by {native_error:.6g}; "
            "use a model-specific language adapter"
        )

    synchronize(device)
    step_started = time.perf_counter()
    step = system.advance(prefix, system.initial_state(1, device), collect_health=True)
    loss = F.cross_entropy(step.logits, target)
    organism.zero_grad(set_to_none=True)
    loss.backward()
    synchronize(device)
    step_seconds = time.perf_counter() - step_started

    zero_controls = int(torch.count_nonzero(step.controls.detach())) == 0
    decoder = organism.attached_organs["language"].output.decoder
    decoder_grad = (
        0.0
        if decoder.weight.grad is None
        else float(decoder.weight.grad.detach().abs().sum())
    )
    backbone_gradient_tensors = sum(
        parameter.grad is not None for parameter in backbone.parameters()
    )
    result = {
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "prompt_tokens": int(input_ids.shape[1]),
        "backbone_width": backbone.width,
        "backbone_vocab_size": backbone.vocab_size,
        "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
        "backbone_trainable_parameters": sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        ),
        "backbone_gradient_tensors": backbone_gradient_tensors,
        "organism_trainable_parameters": count_parameters(organism),
        "native_logit_error": native_error,
        "zero_controls": zero_controls,
        "control_decoder_gradient_l1": decoder_grad,
        "loss": float(loss.detach()),
        "step_seconds": step_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
        "health": None if step.health is None else step.health.__dict__,
    }
    if not zero_controls or decoder_grad <= 0 or backbone_gradient_tensors != 0:
        raise RuntimeError("language graft failed its zero-up or frozen-gradient contract")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="User: Hello. What should I call you?\nAssistant:")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input-cells", type=int, default=8)
    parser.add_argument("--memory-cells", type=int, default=32)
    parser.add_argument("--compute-cells", type=int, default=128)
    parser.add_argument("--relay-cells", type=int, default=32)
    parser.add_argument("--output-cells", type=int, default=8)
    parser.add_argument("--organism-hidden", type=int, default=192)
    parser.add_argument("--dendrites", type=int, default=16)
    parser.add_argument("--active-dendrites", type=int, default=12)
    parser.add_argument("--microsteps", type=int, default=5)
    parser.add_argument("--organ-queries", type=int, default=8)
    parser.add_argument("--control-tokens", type=int, default=8)
    parser.add_argument("--control-rank", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
