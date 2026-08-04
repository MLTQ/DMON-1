"""Screen preregistered L0-C1 wiki questions against frozen Llama knowledge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .config import Sol2Config
from .language_backbone import HuggingFaceFrozenBackbone
from .language_smoke import dtype_from_name
from .living_language import graft_language_backbone
from .model import Sol2
from .wiki_memory import (
    label_token_ids,
    load_wiki_memory_corpus,
    score_frozen_baseline,
    verify_wiki_sources,
)


def run_baseline(args: argparse.Namespace) -> dict:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to screen wiki memory") from error

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    corpus = load_wiki_memory_corpus(args.corpus)
    verify_wiki_sources(corpus, args.wiki_root)
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
    labels = label_token_ids(tokenizer, device)
    cfg = Sol2Config(vocab_size=2, seed=args.seed, device=str(device))
    torch.manual_seed(args.seed)
    organism = Sol2(cfg).to(device)
    system = graft_language_backbone(organism, backbone, seed=args.seed + 1)

    rows = []
    split_counts: dict[str, dict[str, int]] = {}
    for document in corpus.documents:
        counts = split_counts.setdefault(
            document.split, {"count": 0, "correct": 0, "admitted": 0}
        )
        for question in document.questions:
            result = score_frozen_baseline(
                system,
                tokenizer,
                question,
                permutation_seed=args.permutation_seed,
                max_question_tokens=args.max_question_tokens,
            )
            admitted = not result.correct
            counts["count"] += 1
            counts["correct"] += int(result.correct)
            counts["admitted"] += int(admitted)
            rows.append(
                {
                    "document_id": document.id,
                    "question_id": question.id,
                    "split": document.split,
                    "correct": result.correct,
                    "admitted": admitted,
                    "predicted_label": result.predicted_label,
                    "correct_label": result.correct_label,
                    "label_log_probs": result.label_log_probs,
                }
            )
    splits = {
        split: {
            **counts,
            "accuracy": counts["correct"] / counts["count"],
            "admission_rate": counts["admitted"] / counts["count"],
        }
        for split, counts in split_counts.items()
    }
    return {
        "schema": 1,
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": (
            torch.cuda.get_device_name() if device.type == "cuda" else None
        ),
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "wiki_root": str(args.wiki_root),
        "permutation_seed": args.permutation_seed,
        "label_token_ids": labels.tolist(),
        "backbone_trainable_parameters": sum(
            parameter.numel()
            for parameter in backbone.parameters()
            if parameter.requires_grad
        ),
        "splits": splits,
        "rows": rows,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=(
            Path(__file__).with_name("experiments")
            / "l0c1-wiki-memory-corpus.json"
        ),
    )
    parser.add_argument("--wiki-root", type=Path, default=Path("/home/m/wiki"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--permutation-seed", type=int, default=41)
    parser.add_argument("--max-question-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_baseline(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["splits"], indent=2))


if __name__ == "__main__":
    main()
