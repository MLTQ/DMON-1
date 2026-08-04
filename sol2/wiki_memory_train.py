"""Checkpointed meta-training and causal evaluation for L0-C1 wiki memory."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import pack_state, restore_rng_state, unpack_state
from .config import Sol2Config
from .language_backbone import HuggingFaceFrozenBackbone
from .language_smoke import dtype_from_name
from .living_language import LivingLanguageSystem, graft_language_backbone
from .model import Sol2, count_parameters
from .state import OrganismState
from .wiki_memory import (
    LABELS,
    WikiDocument,
    WikiEpisodeResult,
    WikiMemoryCorpus,
    WikiQuestion,
    encode_text,
    format_wiki_question,
    load_wiki_memory_corpus,
    run_wiki_memory_episode,
    summarize_results,
    verify_wiki_sources,
)


@dataclass(frozen=True)
class WikiMemoryTrainConfig:
    updates: int = 300
    lr: float = 2e-3
    grad_clip: float = 1.0
    eval_every: int = 50
    checkpoint_every: int = 50
    permutation_seed: int = 101
    max_memory_tokens: int = 256
    max_question_tokens: int = 256
    seed: int = 7
    paired_counterfactual: bool = False
    binding_margin: float = 1.0
    binding_weight: float = 1.0

    def validate(self) -> None:
        if self.updates < 1 or self.eval_every < 1 or self.checkpoint_every < 1:
            raise ValueError("update and interval counts must be positive")
        if self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.binding_margin <= 0 or self.binding_weight < 0:
            raise ValueError("binding margin must be positive and weight nonnegative")


def counterfactual_training_record(
    document: WikiDocument,
    question: WikiQuestion,
    *,
    epoch: int,
) -> tuple[WikiDocument, WikiQuestion]:
    """Bind one meta-training question to an episode-varying answer choice."""

    answer = (question.answer + 1 + epoch) % 4
    annotation = (
        "\n\nTemporary archive erratum for this episode: if asked \""
        f"{question.question}\" the designated answer is \"{question.choices[answer]}\". "
        "Use this temporary annotation even if it conflicts with prior knowledge."
    )
    return (
        dataclasses.replace(document, memory=document.memory + annotation),
        dataclasses.replace(question, answer=answer),
    )


def counterfactual_training_pair(
    document: WikiDocument,
    question: WikiQuestion,
    *,
    epoch: int,
) -> tuple[tuple[WikiDocument, WikiQuestion], tuple[WikiDocument, WikiQuestion]]:
    """Create two incompatible bindings for one identical question and start state."""

    return (
        counterfactual_training_record(document, question, epoch=epoch),
        counterfactual_training_record(document, question, epoch=epoch + 1),
    )


def paired_binding_margin_loss(
    left: WikiEpisodeResult,
    right: WikiEpisodeResult,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require each passage branch to prefer its own target over its mate's target."""

    if left.label_logits is None or right.label_logits is None:
        raise ValueError("paired binding loss requires live label logits")
    left_target = LABELS.index(left.correct_label)
    right_target = LABELS.index(right.correct_label)
    if left_target == right_target:
        raise ValueError("paired binding targets must be incompatible")
    left_logits = left.label_logits.float()
    right_logits = right.label_logits.float()
    preferences = torch.stack(
        (
            left_logits[left_target] - left_logits[right_target],
            right_logits[right_target] - right_logits[left_target],
        )
    )
    loss = torch.relu(preferences.new_tensor(margin) - preferences).mean()
    return loss, preferences


def save_wiki_memory_checkpoint(
    path: Path,
    *,
    system: LivingLanguageSystem,
    optimizer: torch.optim.Optimizer,
    lane_state: OrganismState,
    organism_config: Sol2Config,
    train_config: WikiMemoryTrainConfig,
    model_name: str,
    dtype: str,
    corpus_sha256: str,
    update: int,
    history: list[dict],
    evaluations: list[dict],
) -> None:
    payload = {
        "schema": 1,
        "kind": "l0c1-wiki-memory",
        "model_name": model_name,
        "dtype": dtype,
        "corpus_sha256": corpus_sha256,
        "organism_config": organism_config.to_dict(),
        "train_config": dataclasses.asdict(train_config),
        "organism": system.organism.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lane_state": pack_state(lane_state),
        "update": update,
        "history": history,
        "evaluations": evaluations,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_wiki_memory_checkpoint(
    path: Path,
    *,
    system: LivingLanguageSystem,
    optimizer: torch.optim.Optimizer,
    device: str,
    expected_corpus_sha256: str,
) -> tuple[dict, OrganismState]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != 1 or payload.get("kind") != "l0c1-wiki-memory":
        raise ValueError("unsupported wiki-memory checkpoint")
    if payload["corpus_sha256"] != expected_corpus_sha256:
        raise ValueError("wiki-memory corpus changed since checkpoint")
    system.organism.load_state_dict(payload["organism"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload)
    return payload, unpack_state(payload["lane_state"], device)


def _flatten_split(corpus: WikiMemoryCorpus, split: str):
    return [
        (document, question)
        for document in corpus.split(split)
        for question in document.questions
    ]


@torch.no_grad()
def evaluate_wiki_memory(
    system: LivingLanguageSystem,
    tokenizer: Any,
    corpus: WikiMemoryCorpus,
    *,
    split: str,
    permutation_seed: int,
    max_memory_tokens: int,
    max_question_tokens: int,
    admitted_questions: set[str] | None = None,
) -> dict:
    documents = corpus.split(split)
    rows = []
    arm_results: dict[str, list] = {
        name: []
        for name in (
            "normal",
            "no_exposure",
            "zero_control",
            "reset_after_exposure",
            "wrong_passage",
            "memory_lesion",
            "internal_lesion",
            "question_paraphrase",
        )
    }
    device = next(system.organism.parameters()).device
    memory_features = {
        document.id: system.backbone.encode(
            encode_text(
                tokenizer,
                document.memory,
                device,
                max_tokens=max_memory_tokens,
            )
        )
        for document in documents
    }
    question_features = {}
    for document in documents:
        for question in document.questions:
            for paraphrase in (False, True):
                formatted = format_wiki_question(
                    question,
                    permutation_seed=permutation_seed,
                    paraphrase=paraphrase,
                )
                ids = encode_text(
                    tokenizer,
                    formatted.prompt,
                    device,
                    max_tokens=max_question_tokens,
                )
                question_features[(question.id, paraphrase)] = system.backbone.encode(ids)
    for document_index, document in enumerate(documents):
        wrong_document = documents[(document_index + 1) % len(documents)]
        for question in document.questions:
            common = {
                "system": system,
                "tokenizer": tokenizer,
                "document": document,
                "question": question,
                "state": system.initial_state(1, device),
                "permutation_seed": permutation_seed,
                "max_memory_tokens": max_memory_tokens,
                "max_question_tokens": max_question_tokens,
            }
            results = {
                "normal": run_wiki_memory_episode(
                    **common,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                ),
                "no_exposure": run_wiki_memory_episode(
                    **common,
                    expose=False,
                    question_features=question_features[(question.id, False)],
                ),
                "zero_control": run_wiki_memory_episode(
                    **common,
                    control_scale=0.0,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                ),
                "reset_after_exposure": run_wiki_memory_episode(
                    **common,
                    reset_after_exposure=True,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                ),
                "wrong_passage": run_wiki_memory_episode(
                    **common,
                    exposure_document=wrong_document,
                    exposure_features=memory_features[wrong_document.id],
                    question_features=question_features[(question.id, False)],
                ),
                "memory_lesion": run_wiki_memory_episode(
                    **common,
                    memory_lesion=True,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                ),
                "internal_lesion": run_wiki_memory_episode(
                    **common,
                    frozen_idx=system.organism.internal_idx,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                ),
                "question_paraphrase": run_wiki_memory_episode(
                    **common,
                    paraphrase=True,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, True)],
                ),
            }
            for arm, result in results.items():
                arm_results[arm].append(result)
            rows.append(
                {
                    "document_id": document.id,
                    "question_id": question.id,
                    "admitted": (
                        None
                        if admitted_questions is None
                        else question.id in admitted_questions
                    ),
                    "arms": {
                        arm: {
                            "correct": result.correct,
                            "predicted_label": result.predicted_label,
                            "correct_label": result.correct_label,
                            "loss": float(result.loss),
                            "label_log_probs": result.label_log_probs,
                        }
                        for arm, result in results.items()
                    },
                }
            )
    summaries = {
        arm: summarize_results(results) for arm, results in arm_results.items()
    }
    if admitted_questions is not None:
        summaries["admitted_normal_accuracy"] = sum(
            row["arms"]["normal"]["correct"]
            for row in rows
            if row["admitted"]
        ) / max(sum(bool(row["admitted"]) for row in rows), 1)
    return {"split": split, "summaries": summaries, "rows": rows}


def _admitted_questions(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return {row["question_id"] for row in payload["rows"] if row["admitted"]}


def build_system(args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install transformers to train wiki memory") from error
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
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
    return system, tokenizer, cfg


def run_training(args: argparse.Namespace) -> dict:
    train_config = WikiMemoryTrainConfig(
        updates=args.updates,
        lr=args.lr,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        permutation_seed=args.permutation_seed,
        max_memory_tokens=args.max_memory_tokens,
        max_question_tokens=args.max_question_tokens,
        seed=args.seed,
        paired_counterfactual=args.paired_counterfactual,
        binding_margin=args.binding_margin,
        binding_weight=args.binding_weight,
    )
    train_config.validate()
    corpus = load_wiki_memory_corpus(args.corpus)
    verify_wiki_sources(corpus, args.wiki_root)
    corpus_sha256 = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    admitted = _admitted_questions(args.baseline)
    system, tokenizer, organism_config = build_system(args)
    optimizer = torch.optim.AdamW(
        system.organism.parameters(), lr=train_config.lr, weight_decay=0.0
    )
    checkpoint_path = args.out_dir / "checkpoint.pt"
    history: list[dict] = []
    evaluations: list[dict] = []
    update = 0
    lane_state = system.initial_state(1, args.device)
    if args.resume and checkpoint_path.exists():
        payload, lane_state = load_wiki_memory_checkpoint(
            checkpoint_path,
            system=system,
            optimizer=optimizer,
            device=args.device,
            expected_corpus_sha256=corpus_sha256,
        )
        update = int(payload["update"])
        history = payload["history"]
        evaluations = payload["evaluations"]

    schedule = _flatten_split(corpus, "meta_train")
    while update < train_config.updates:
        document, question = schedule[update % len(schedule)]
        epoch = update // len(schedule)
        if train_config.paired_counterfactual:
            training_records = counterfactual_training_pair(
                document, question, epoch=epoch
            )
        else:
            training_records = (
                counterfactual_training_record(document, question, epoch=epoch),
            )
        optimizer.zero_grad(set_to_none=True)
        branch_results = []
        for training_document, training_question in training_records:
            branch_result = run_wiki_memory_episode(
                system,
                tokenizer,
                training_document,
                training_question,
                lane_state,
                permutation_seed=train_config.permutation_seed + update,
                max_memory_tokens=train_config.max_memory_tokens,
                max_question_tokens=train_config.max_question_tokens,
            )
            branch_results.append(branch_result)
        task_loss = torch.stack([branch.loss for branch in branch_results]).mean()
        binding_loss = task_loss.new_zeros(())
        binding_preferences = task_loss.new_zeros(2)
        if len(branch_results) == 2 and train_config.binding_weight > 0:
            binding_loss, binding_preferences = paired_binding_margin_loss(
                branch_results[0],
                branch_results[1],
                margin=train_config.binding_margin,
            )
        objective_loss = task_loss + train_config.binding_weight * binding_loss
        objective_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            system.organism.parameters(), train_config.grad_clip
        )
        optimizer.step()
        update += 1
        result = branch_results[0]
        lane_state = result.state.detach()
        lane_state.weight_version = update
        mean_loss = float(task_loss.detach())
        branch_accuracy = sum(branch.correct for branch in branch_results) / len(
            branch_results
        )
        control_separation_rms = 0.0
        label_logit_separation_rms = 0.0
        if len(branch_results) == 2:
            left, right = branch_results
            assert left.controls is not None and right.controls is not None
            assert left.label_logits is not None and right.label_logits is not None
            control_separation_rms = float(
                (left.controls.detach() - right.controls.detach())
                .pow(2)
                .mean()
                .sqrt()
            )
            label_logit_separation_rms = float(
                (left.label_logits.detach() - right.label_logits.detach())
                .pow(2)
                .mean()
                .sqrt()
            )
        row = {
            "update": update,
            "document_id": document.id,
            "question_id": question.id,
            "loss": mean_loss,
            "bits": mean_loss / math.log(2.0),
            "objective_loss": float(objective_loss.detach()),
            "binding_loss": float(binding_loss.detach()),
            "binding_preferences": [
                float(value) for value in binding_preferences.detach()
            ],
            "correct": result.correct,
            "branch_accuracy": branch_accuracy,
            "branch_answers": [branch.correct_label for branch in branch_results],
            "control_separation_rms": control_separation_rms,
            "label_logit_separation_rms": label_logit_separation_rms,
            "grad_norm": float(grad_norm),
            "control_rms": sum(branch.control_rms for branch in branch_results)
            / len(branch_results),
            "internal_activity": sum(
                branch.internal_activity for branch in branch_results
            )
            / len(branch_results),
            "memory_activity": sum(branch.memory_activity for branch in branch_results)
            / len(branch_results),
        }
        history.append(row)
        if update == 1 or update % args.log_every == 0:
            print(
                f"[wiki-memory] u{update} loss={row['loss']:.4f} "
                f"branch_acc={row['branch_accuracy']:.2f} "
                f"control={row['control_rms']:.4f} "
                f"control_sep={row['control_separation_rms']:.4f} "
                f"binding={row['binding_loss']:.4f} "
                f"grad={row['grad_norm']:.3f}",
                flush=True,
            )
        if update % train_config.eval_every == 0 or update == train_config.updates:
            evaluation = evaluate_wiki_memory(
                system,
                tokenizer,
                corpus,
                split="development",
                permutation_seed=train_config.permutation_seed + 50_000,
                max_memory_tokens=train_config.max_memory_tokens,
                max_question_tokens=train_config.max_question_tokens,
                admitted_questions=admitted,
            )
            evaluation["update"] = update
            evaluations.append(evaluation)
            normal = evaluation["summaries"]["normal"]["accuracy"]
            reset = evaluation["summaries"]["reset_after_exposure"]["accuracy"]
            print(
                f"[development] u{update} normal={normal:.3f} reset={reset:.3f}",
                flush=True,
            )
        if (
            update % train_config.checkpoint_every == 0
            or update == train_config.updates
        ):
            save_wiki_memory_checkpoint(
                checkpoint_path,
                system=system,
                optimizer=optimizer,
                lane_state=lane_state,
                organism_config=organism_config,
                train_config=train_config,
                model_name=args.model,
                dtype=args.dtype,
                corpus_sha256=corpus_sha256,
                update=update,
                history=history,
                evaluations=evaluations,
            )

    heldout = evaluate_wiki_memory(
        system,
        tokenizer,
        corpus,
        split="heldout",
        permutation_seed=train_config.permutation_seed + 100_000,
        max_memory_tokens=train_config.max_memory_tokens,
        max_question_tokens=train_config.max_question_tokens,
        admitted_questions=admitted,
    )
    metrics = {
        "schema": 1,
        "kind": "l0c1-wiki-memory",
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "corpus_sha256": corpus_sha256,
        "updates": update,
        "organism_config": organism_config.to_dict(),
        "train_config": dataclasses.asdict(train_config),
        "organism_trainable_parameters": count_parameters(system.organism),
        "backbone_trainable_parameters": sum(
            parameter.numel()
            for parameter in system.backbone.parameters()
            if parameter.requires_grad
        ),
        "backbone_gradient_tensors": sum(
            parameter.grad is not None for parameter in system.backbone.parameters()
        ),
        "history": history,
        "development": evaluations,
        "heldout": heldout,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated()
            if torch.cuda.is_available() and args.device.startswith("cuda")
            else 0
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved()
            if torch.cuda.is_available() and args.device.startswith("cuda")
            else 0
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


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
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--wiki-root", type=Path, default=Path("/home/m/wiki"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--permutation-seed", type=int, default=101)
    parser.add_argument("--max-memory-tokens", type=int, default=256)
    parser.add_argument("--max-question-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--paired-counterfactual",
        action="store_true",
        help="train two incompatible passage bindings from one matched start state",
    )
    parser.add_argument("--binding-margin", type=float, default=1.0)
    parser.add_argument("--binding-weight", type=float, default=1.0)
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
    metrics = run_training(parse_args())
    print(json.dumps(metrics["heldout"]["summaries"], indent=2))


if __name__ == "__main__":
    main()
