"""Audit whether a frozen language model is a usable DMON language organ."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .language_backbone import HuggingFaceFrozenBackbone
from .language_smoke import dtype_from_name, synchronize
from .wiki_memory import (
    LABELS,
    WikiDocument,
    WikiQuestion,
    encode_text,
    format_wiki_question,
    label_token_ids,
    load_wiki_memory_corpus,
)
from .wiki_memory_train import counterfactual_training_pair


@torch.no_grad()
def score_visible_binding(
    backbone: HuggingFaceFrozenBackbone,
    tokenizer: Any,
    document: WikiDocument,
    question: WikiQuestion,
    *,
    permutation_seed: int,
    max_tokens: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Score one temporary binding while its memory card remains visible."""

    device = next(backbone.parameters()).device
    formatted = format_wiki_question(question, permutation_seed=permutation_seed)
    input_ids = encode_text(
        tokenizer,
        f"{document.memory}\n\n{formatted.prompt}",
        device,
        max_tokens=max_tokens,
    )
    vocab_logits = backbone.controlled_logits(input_ids)[0, -1]
    labels = label_token_ids(tokenizer, device)
    label_logits = vocab_logits.index_select(0, labels).float()
    label_log_probs = F.log_softmax(label_logits, dim=0)
    correct = formatted.correct_index
    other = torch.cat((label_logits[:correct], label_logits[correct + 1 :]))
    predicted = int(label_logits.argmax())
    row = {
        "document_id": document.id,
        "question_id": question.id,
        "correct_label": formatted.correct_label,
        "predicted_label": LABELS[predicted],
        "correct": predicted == correct,
        "correct_label_probability": float(label_log_probs[correct].exp()),
        "correct_label_margin": float(label_logits[correct] - other.max()),
        "label_log_probs": {
            label: float(value)
            for label, value in zip(LABELS, label_log_probs)
        },
        "tokens": int(input_ids.shape[1]),
    }
    return row, vocab_logits.float().cpu()


def summarize_binding_rows(
    rows: list[dict[str, Any]], pair_separations: list[float]
) -> dict[str, float]:
    """Aggregate binding fidelity without hiding adverse individual examples."""

    if not rows or not pair_separations:
        raise ValueError("binding audit requires non-empty rows and pairs")
    return {
        "count": float(len(rows)),
        "accuracy": sum(float(row["correct"]) for row in rows) / len(rows),
        "mean_correct_label_probability": sum(
            float(row["correct_label_probability"]) for row in rows
        )
        / len(rows),
        "mean_correct_label_margin": sum(
            float(row["correct_label_margin"]) for row in rows
        )
        / len(rows),
        "mean_pair_logit_separation_rms": sum(pair_separations)
        / len(pair_separations),
        "minimum_pair_logit_separation_rms": min(pair_separations),
    }


def audit_prefix(
    backbone: HuggingFaceFrozenBackbone,
    tokenizer: Any,
    question: WikiQuestion,
    *,
    permutation_seed: int,
    control_tokens: int,
    perturbation_scale: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Measure prefix credit and response around the anchored-prefix floor."""

    device = next(backbone.parameters()).device
    formatted = format_wiki_question(question, permutation_seed=permutation_seed)
    input_ids = encode_text(
        tokenizer, formatted.prompt, device, max_tokens=max_tokens
    )
    labels = label_token_ids(tokenizer, device)
    correct_id = labels[formatted.correct_index]
    controls = torch.zeros(
        1,
        control_tokens,
        backbone.width,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    started = time.perf_counter()
    logits = backbone.prefix_logits(input_ids, controls)[0, -1].float()
    loss = -F.log_softmax(logits, dim=0)[correct_id]
    loss.backward()
    synchronize(device)
    elapsed = time.perf_counter() - started
    gradient = controls.grad
    if gradient is None:
        raise RuntimeError("continuous prefix did not receive a gradient")

    generator = torch.Generator().manual_seed(1701)
    direction = torch.randn(
        controls.shape, generator=generator, dtype=torch.float32
    ).to(device)
    direction = direction / direction.square().mean().sqrt()
    embedding = backbone.model.get_input_embeddings()(input_ids).detach()
    embedding_rms = embedding.float().square().mean().sqrt()
    perturbation = direction * (perturbation_scale * embedding_rms)
    with torch.no_grad():
        perturbed = backbone.prefix_logits(input_ids, perturbation)[0, -1].float()
    label_delta = (perturbed - logits.detach()).index_select(0, labels)
    log_p = F.log_softmax(logits.detach(), dim=0)
    log_q = F.log_softmax(perturbed, dim=0)
    p = log_p.exp()
    return {
        "question_id": question.id,
        "tokens": int(input_ids.shape[1]),
        "loss": float(loss.detach()),
        "seconds": elapsed,
        "gradient_rms": float(gradient.float().square().mean().sqrt()),
        "gradient_absmax": float(gradient.float().abs().max()),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "backbone_gradient_tensors": sum(
            parameter.grad is not None for parameter in backbone.parameters()
        ),
        "embedding_rms": float(embedding_rms),
        "perturbation_relative_scale": perturbation_scale,
        "full_logit_delta_rms": float(
            (perturbed - logits.detach()).square().mean().sqrt()
        ),
        "label_logit_delta_rms": float(label_delta.square().mean().sqrt()),
        "forward_kl": float((p * (log_p - log_q)).sum()),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to audit a language backbone") from error

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
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
    corpus = load_wiki_memory_corpus(args.corpus)
    examples = [
        (document, question)
        for document in corpus.split("meta_train")
        for question in document.questions
    ]
    if not examples:
        raise ValueError("meta-training split is empty")

    first_document, first_question = examples[0]
    formatted = format_wiki_question(
        first_question, permutation_seed=args.permutation_seed
    )
    parity_ids = encode_text(
        tokenizer,
        formatted.prompt,
        device,
        max_tokens=args.max_tokens,
    )
    native_error = backbone.native_logit_error(parity_ids)
    prefix = audit_prefix(
        backbone,
        tokenizer,
        first_question,
        permutation_seed=args.permutation_seed,
        control_tokens=args.control_tokens,
        perturbation_scale=args.perturbation_scale,
        max_tokens=args.max_tokens,
    )

    rows: list[dict[str, Any]] = []
    pair_separations: list[float] = []
    binding_started = time.perf_counter()
    for document, question in examples:
        pair = counterfactual_training_pair(
            document, question, epoch=0, compact=True
        )
        pair_logits = []
        for branch, (bound_document, bound_question) in zip(("left", "right"), pair):
            row, vocab_logits = score_visible_binding(
                backbone,
                tokenizer,
                bound_document,
                bound_question,
                permutation_seed=args.permutation_seed,
                max_tokens=args.max_tokens,
            )
            row["branch"] = branch
            rows.append(row)
            pair_logits.append(vocab_logits)
        pair_separations.append(
            float((pair_logits[0] - pair_logits[1]).square().mean().sqrt())
        )
    synchronize(device)
    result = {
        "schema": 1,
        "kind": "l0c1q-modern-backbone-audit",
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "backbone_class": type(backbone.model).__name__,
        "backbone_width": backbone.width,
        "backbone_vocab_size": backbone.vocab_size,
        "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
        "backbone_trainable_parameters": sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        ),
        "native_logit_error": native_error,
        "prefix": prefix,
        "binding_summary": summarize_binding_rows(rows, pair_separations),
        "binding_seconds": time.perf_counter() - binding_started,
        "binding_rows": rows,
        "pair_logit_separations_rms": pair_separations,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
        ),
        "total_seconds": time.perf_counter() - started,
    }
    if not math.isfinite(native_error):
        raise RuntimeError("native-logit parity is non-finite")
    if not prefix["gradient_finite"] or prefix["gradient_rms"] <= 0:
        raise RuntimeError("continuous-prefix gradient gate failed")
    if prefix["backbone_gradient_tensors"] != 0:
        raise RuntimeError("frozen backbone received gradients")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("sol2/experiments/l0c1-wiki-memory-corpus.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--permutation-seed", type=int, default=101)
    parser.add_argument("--max-tokens", type=int, default=528)
    parser.add_argument("--control-tokens", type=int, default=4)
    parser.add_argument("--perturbation-scale", type=float, default=0.01)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.control_tokens < 1:
        parser.error("token limits must be positive")
    if args.perturbation_scale <= 0:
        parser.error("perturbation scale must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
