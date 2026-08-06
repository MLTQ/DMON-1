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
    label_token_ids,
    load_wiki_memory_corpus,
    run_wiki_memory_episode,
    summarize_results,
    verify_wiki_sources,
)
from .wiki_causal_contrast import passage_causal_contrast_loss
from .wiki_distillation import (
    control_delta_energy,
    full_vocab_effect_reverse_kl,
    full_vocab_reverse_kl,
)
from .wiki_output_credit import fixed_output_codebook, output_tissue_credit_loss
from .wiki_output_eligibility import eligibility_gated_transport_loss
from .wiki_semantic_credit import (
    fixed_semantic_projection,
    paired_tissue_semantic_credit,
    paired_vector_semantic_credit,
    recall_addressing_credit,
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
    lifetime_lane_policy: str = "continuous"
    paired_counterfactual: bool = False
    task_weight: float = 1.0
    binding_margin: float = 1.0
    binding_weight: float = 1.0
    causal_contrast_weight: float = 0.0
    causal_contrast_margin: float = 0.1
    teacher_distillation_weight: float = 0.0
    teacher_effect_distillation_weight: float = 0.0
    teacher_distillation_temperature: float = 1.0
    memory_semantic_credit_weight: float = 0.0
    relay_semantic_credit_weight: float = 0.0
    recall_semantic_credit_weight: float = 0.0
    recall_addressing_credit_weight: float = 0.0
    recall_addressing_temperature: float = 0.1
    semantic_credit_target_rms: float = 0.25
    semantic_credit_seed: int = 307
    control_energy_weight: float = 0.0
    reference_centered_controls: bool = False
    freeze_effector_updates: bool = False
    language_control_mode: str = "late_residual"
    compact_bindings: bool = False
    control_gain: float = 1.0
    coherent_recall: bool = False
    recall_residual_gain: float = 0.1
    recall_top_k: int = 0
    recall_recency_bias: float = 0.0
    recall_lr_multiplier: float = 1.0
    sensor_lr_multiplier: float = 1.0
    effector_lr_multiplier: float = 1.0
    output_credit_weight: float = 0.0
    output_credit_scale: float = 4.0
    output_credit_seed: int = 211
    output_eligibility_weight: float = 0.0
    output_eligibility_margin: float = 0.005
    output_eligibility_temperature: float = 0.005
    output_eligibility_relay_reference: float = 0.005

    def validate(self) -> None:
        if self.updates < 1 or self.eval_every < 1 or self.checkpoint_every < 1:
            raise ValueError("update and interval counts must be positive")
        if self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.lifetime_lane_policy not in {"continuous", "fresh_episode"}:
            raise ValueError("unknown lifetime lane policy")
        if self.task_weight < 0:
            raise ValueError("task weight must be nonnegative")
        if self.binding_margin <= 0 or self.binding_weight < 0:
            raise ValueError("binding margin must be positive and weight nonnegative")
        if self.causal_contrast_weight < 0 or self.causal_contrast_margin <= 0:
            raise ValueError(
                "causal contrast weight must be nonnegative and margin positive"
            )
        if self.causal_contrast_weight > 0 and not self.paired_counterfactual:
            raise ValueError("causal contrast requires paired counterfactual training")
        if self.teacher_distillation_weight < 0:
            raise ValueError("teacher distillation weight must be nonnegative")
        if self.teacher_effect_distillation_weight < 0:
            raise ValueError("teacher effect distillation weight must be nonnegative")
        if self.teacher_distillation_temperature <= 0:
            raise ValueError("teacher distillation temperature must be positive")
        if min(
            self.memory_semantic_credit_weight,
            self.relay_semantic_credit_weight,
            self.recall_semantic_credit_weight,
            self.recall_addressing_credit_weight,
        ) < 0:
            raise ValueError("semantic credit weights must be nonnegative")
        if (
            self.memory_semantic_credit_weight > 0
            or self.relay_semantic_credit_weight > 0
            or self.recall_semantic_credit_weight > 0
        ) and not self.paired_counterfactual:
            raise ValueError("semantic credit requires paired counterfactual training")
        if self.semantic_credit_target_rms <= 0:
            raise ValueError("semantic credit target RMS must be positive")
        if self.recall_addressing_temperature <= 0:
            raise ValueError("recall addressing temperature must be positive")
        if self.control_energy_weight < 0:
            raise ValueError("control energy weight must be nonnegative")
        if self.language_control_mode not in {"late_residual", "prefix"}:
            raise ValueError("unknown language control mode")
        if self.control_gain <= 0:
            raise ValueError("control gain must be positive")
        if min(
            self.recall_residual_gain,
            float(self.recall_top_k),
            self.recall_recency_bias,
        ) < 0:
            raise ValueError("recall structure values must be nonnegative")
        if min(
            self.recall_lr_multiplier,
            self.sensor_lr_multiplier,
            self.effector_lr_multiplier,
        ) <= 0:
            raise ValueError("plasticity multipliers must be positive")
        if self.output_credit_weight < 0 or self.output_credit_scale <= 0:
            raise ValueError("output credit weight must be nonnegative and scale positive")
        if self.output_eligibility_weight < 0:
            raise ValueError("output eligibility weight must be nonnegative")
        if min(
            self.output_eligibility_margin,
            self.output_eligibility_temperature,
            self.output_eligibility_relay_reference,
        ) <= 0:
            raise ValueError("output eligibility scales must be positive")


def counterfactual_training_record(
    document: WikiDocument,
    question: WikiQuestion,
    *,
    epoch: int,
    compact: bool = False,
) -> tuple[WikiDocument, WikiQuestion]:
    """Bind one meta-training question to an episode-varying answer choice."""

    answer = (question.answer + 1 + epoch) % 4
    annotation = (
        "Temporary archive binding for this episode.\n"
        f"Question: {question.question}\n"
        f"Designated answer: {question.choices[answer]}\n"
        "Use this temporary binding even if it conflicts with prior knowledge."
    )
    memory = annotation if compact else f"{document.memory}\n\n{annotation}"
    return (
        dataclasses.replace(document, memory=memory),
        dataclasses.replace(question, answer=answer),
    )


@torch.no_grad()
def passage_visible_teacher_outputs(
    system: LivingLanguageSystem,
    tokenizer: Any,
    document: WikiDocument,
    question: WikiQuestion,
    *,
    permutation_seed: int,
    max_memory_tokens: int,
    max_question_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final contextual features and logits with a visible passage."""

    device = next(system.organism.parameters()).device
    formatted = format_wiki_question(question, permutation_seed=permutation_seed)
    teacher_ids = encode_text(
        tokenizer,
        f"{document.memory}\n\n{formatted.prompt}",
        device,
        max_tokens=max_memory_tokens + max_question_tokens + 16,
    )
    features = system.backbone.encode(teacher_ids)
    logits = system.backbone.controlled_logits_from_features(features[:, -1:])
    return features[:, -1].detach(), logits[0, -1].detach()


def passage_visible_teacher_logits(
    system: LivingLanguageSystem,
    tokenizer: Any,
    document: WikiDocument,
    question: WikiQuestion,
    *,
    permutation_seed: int,
    max_memory_tokens: int,
    max_question_tokens: int,
) -> torch.Tensor:
    """Score the full vocabulary while the frozen teacher can see the passage."""

    return passage_visible_teacher_outputs(
        system,
        tokenizer,
        document,
        question,
        permutation_seed=permutation_seed,
        max_memory_tokens=max_memory_tokens,
        max_question_tokens=max_question_tokens,
    )[1]


@torch.no_grad()
def question_only_teacher_outputs(
    system: LivingLanguageSystem,
    tokenizer: Any,
    question: WikiQuestion,
    *,
    permutation_seed: int,
    max_question_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final contextual features and logits without a passage."""

    device = next(system.organism.parameters()).device
    formatted = format_wiki_question(question, permutation_seed=permutation_seed)
    teacher_ids = encode_text(
        tokenizer,
        formatted.prompt,
        device,
        max_tokens=max_question_tokens,
    )
    features = system.backbone.encode(teacher_ids)
    logits = system.backbone.controlled_logits_from_features(features[:, -1:])
    return features[:, -1].detach(), logits[0, -1].detach()


def question_only_teacher_logits(
    system: LivingLanguageSystem,
    tokenizer: Any,
    question: WikiQuestion,
    *,
    permutation_seed: int,
    max_question_tokens: int,
) -> torch.Tensor:
    """Score the frozen teacher on the identical question without its passage."""

    return question_only_teacher_outputs(
        system,
        tokenizer,
        question,
        permutation_seed=permutation_seed,
        max_question_tokens=max_question_tokens,
    )[1]


def counterfactual_training_pair(
    document: WikiDocument,
    question: WikiQuestion,
    *,
    epoch: int,
    compact: bool = False,
) -> tuple[tuple[WikiDocument, WikiQuestion], tuple[WikiDocument, WikiQuestion]]:
    """Create two incompatible bindings for one identical question and start state."""

    return (
        counterfactual_training_record(
            document, question, epoch=epoch, compact=compact
        ),
        counterfactual_training_record(
            document, question, epoch=epoch + 1, compact=compact
        ),
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


def _organism_parameter_group(name: str) -> str:
    if name.startswith("relay_output_transport."):
        return "transport"
    if ".recall_" in name:
        return "recall"
    if ".sensor." in name or name.endswith(("sensory_gain", "sensory_bias")):
        return "sensor"
    if ".output." in name or name.endswith("control_basis"):
        return "effector"
    if name.startswith("graph."):
        return "connectome"
    if name.startswith("tissues."):
        return "tissues"
    if name.startswith("cell_"):
        return "cell_identity"
    return "other"


def organism_optimizer_groups(
    system: LivingLanguageSystem,
    train_config: WikiMemoryTrainConfig,
) -> list[dict]:
    """Partition organism parameters into explicit plasticity-rate groups."""

    multipliers = {
        "recall": train_config.recall_lr_multiplier,
        "sensor": train_config.sensor_lr_multiplier,
        "effector": train_config.effector_lr_multiplier,
    }
    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name, parameter in system.organism.named_parameters():
        grouped.setdefault(_organism_parameter_group(name), []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": train_config.lr * multipliers.get(group, 1.0),
            "group_name": group,
        }
        for group, parameters in sorted(grouped.items())
    ]


def freeze_organism_parameter_group(
    system: LivingLanguageSystem, group: str
) -> int:
    """Disable optimizer gradients for one named organism parameter group."""

    frozen = 0
    for name, parameter in system.organism.named_parameters():
        if _organism_parameter_group(name) == group:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    if frozen == 0:
        raise ValueError(f"organism parameter group {group!r} is empty")
    return frozen


def organism_gradient_groups(system: LivingLanguageSystem) -> dict[str, dict[str, float]]:
    """Summarize gradient participation across the causal language-memory route."""

    totals: dict[str, list[float]] = {}
    for name, parameter in system.organism.named_parameters():
        group = _organism_parameter_group(name)
        values = totals.setdefault(group, [0.0, 0.0, 0.0])
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            values[0] += float(gradient.pow(2).sum())
            values[1] += float(gradient.numel())
            values[2] += 1.0
    return {
        group: {
            "rms": math.sqrt(square_sum / max(elements, 1.0)),
            "elements": elements,
            "tensors": tensors,
        }
        for group, (square_sum, elements, tensors) in totals.items()
    }


def recall_gradient_components(
    system: LivingLanguageSystem,
) -> dict[str, dict[str, float]]:
    """Report query/key/value/output gradient RMS inside content-addressed recall."""

    component_totals = {
        component: [0.0, 0.0, 0.0]
        for component in ("query", "key", "value", "output")
    }
    for name, parameter in system.organism.named_parameters():
        for component in component_totals:
            if f".recall_{component}." not in name:
                continue
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float()
                values = component_totals[component]
                values[0] += float(gradient.pow(2).sum())
                values[1] += float(gradient.numel())
                values[2] += 1.0
            break
    return {
        component: {
            "rms": math.sqrt(square_sum / max(elements, 1.0)),
            "elements": elements,
            "tensors": tensors,
        }
        for component, (square_sum, elements, tensors) in component_totals.items()
    }


def training_episode_start_state(
    system: LivingLanguageSystem,
    lane_state: OrganismState,
    *,
    policy: str,
) -> OrganismState:
    """Select cross-update cellular inheritance without changing learned weights."""

    if policy == "continuous":
        return lane_state
    if policy == "fresh_episode":
        fresh = system.initial_state(
            lane_state.hidden.shape[0], lane_state.hidden.device
        )
        fresh.weight_version = lane_state.weight_version
        return fresh
    raise ValueError(f"unknown lifetime lane policy {policy!r}")


def require_finite_organism_gradients(system: LivingLanguageSystem) -> None:
    """Reject a wiki-memory update before mutation when any gradient is non-finite."""

    invalid = [
        name
        for name, parameter in system.organism.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
    ]
    if invalid:
        preview = ", ".join(invalid[:8])
        suffix = "" if len(invalid) <= 8 else f" (+{len(invalid) - 8} more)"
        raise FloatingPointError(
            f"non-finite organism gradients before optimizer step: {preview}{suffix}"
        )


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
    control_mode: str = "late_residual",
    reference_centered_controls: bool = False,
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
    question_ids = {}
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
                question_ids[(question.id, paraphrase)] = ids
                question_features[(question.id, paraphrase)] = (
                    system.backbone.encode(ids)
                )
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
                "control_mode": control_mode,
            }
            control_reference = None
            paraphrase_reference = None
            if reference_centered_controls:
                raw_reference = run_wiki_memory_episode(
                    **common,
                    expose=False,
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                )
                if raw_reference.controls is None:
                    raise RuntimeError("reference episode did not emit controls")
                control_reference = raw_reference.controls.detach()
                raw_paraphrase_reference = run_wiki_memory_episode(
                    **common,
                    expose=False,
                    paraphrase=True,
                    question_features=question_features[(question.id, True)],
                    question_ids=question_ids[(question.id, True)],
                )
                if raw_paraphrase_reference.controls is None:
                    raise RuntimeError("paraphrase reference did not emit controls")
                paraphrase_reference = raw_paraphrase_reference.controls.detach()
            results = {
                "normal": run_wiki_memory_episode(
                    **common,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                    control_reference=control_reference,
                ),
                "no_exposure": run_wiki_memory_episode(
                    **common,
                    expose=False,
                    control_scale=(0.0 if reference_centered_controls else 1.0),
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                    control_reference=control_reference,
                ),
                "zero_control": run_wiki_memory_episode(
                    **common,
                    control_scale=0.0,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                    control_reference=control_reference,
                ),
                "reset_after_exposure": run_wiki_memory_episode(
                    **common,
                    reset_after_exposure=True,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                    control_reference=control_reference,
                ),
                "wrong_passage": run_wiki_memory_episode(
                    **common,
                    exposure_document=wrong_document,
                    exposure_features=memory_features[wrong_document.id],
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                    control_reference=control_reference,
                ),
                "memory_lesion": run_wiki_memory_episode(
                    **common,
                    memory_lesion=True,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                    control_reference=control_reference,
                ),
                "internal_lesion": run_wiki_memory_episode(
                    **common,
                    frozen_idx=system.organism.internal_idx,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, False)],
                    question_ids=question_ids[(question.id, False)],
                    control_reference=control_reference,
                ),
                "question_paraphrase": run_wiki_memory_episode(
                    **common,
                    paraphrase=True,
                    exposure_features=memory_features[document.id],
                    question_features=question_features[(question.id, True)],
                    question_ids=question_ids[(question.id, True)],
                    control_reference=paraphrase_reference,
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
        relay_output_attention=getattr(args, "relay_output_attention", False),
        relay_output_gain=getattr(args, "relay_output_gain", 1.0),
        relay_output_initial_gate=getattr(args, "relay_output_initial_gate", 0.0),
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
        control_gain=args.control_gain,
        recall_gain=args.recall_gain,
        coherent_recall=args.coherent_recall,
        recall_residual_gain=args.recall_residual_gain,
        recall_top_k=args.recall_top_k,
        recall_recency_bias=args.recall_recency_bias,
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
        lifetime_lane_policy=args.lifetime_lane_policy,
        paired_counterfactual=args.paired_counterfactual,
        task_weight=args.task_weight,
        binding_margin=args.binding_margin,
        binding_weight=args.binding_weight,
        causal_contrast_weight=args.causal_contrast_weight,
        causal_contrast_margin=args.causal_contrast_margin,
        teacher_distillation_weight=args.teacher_distillation_weight,
        teacher_effect_distillation_weight=(
            args.teacher_effect_distillation_weight
        ),
        teacher_distillation_temperature=args.teacher_distillation_temperature,
        memory_semantic_credit_weight=args.memory_semantic_credit_weight,
        relay_semantic_credit_weight=args.relay_semantic_credit_weight,
        recall_semantic_credit_weight=args.recall_semantic_credit_weight,
        recall_addressing_credit_weight=args.recall_addressing_credit_weight,
        recall_addressing_temperature=args.recall_addressing_temperature,
        semantic_credit_target_rms=args.semantic_credit_target_rms,
        semantic_credit_seed=args.semantic_credit_seed,
        control_energy_weight=args.control_energy_weight,
        reference_centered_controls=args.reference_centered_controls,
        freeze_effector_updates=args.freeze_effector_updates,
        language_control_mode=args.language_control_mode,
        compact_bindings=args.compact_bindings,
        control_gain=args.control_gain,
        coherent_recall=args.coherent_recall,
        recall_residual_gain=args.recall_residual_gain,
        recall_top_k=args.recall_top_k,
        recall_recency_bias=args.recall_recency_bias,
        recall_lr_multiplier=args.recall_lr_multiplier,
        sensor_lr_multiplier=args.sensor_lr_multiplier,
        effector_lr_multiplier=args.effector_lr_multiplier,
        output_credit_weight=args.output_credit_weight,
        output_credit_scale=args.output_credit_scale,
        output_credit_seed=args.output_credit_seed,
        output_eligibility_weight=args.output_eligibility_weight,
        output_eligibility_margin=args.output_eligibility_margin,
        output_eligibility_temperature=args.output_eligibility_temperature,
        output_eligibility_relay_reference=args.output_eligibility_relay_reference,
    )
    train_config.validate()
    corpus = load_wiki_memory_corpus(args.corpus)
    verify_wiki_sources(corpus, args.wiki_root)
    corpus_sha256 = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    admitted = _admitted_questions(args.baseline)
    system, tokenizer, organism_config = build_system(args)
    frozen_effector_parameters = 0
    if train_config.freeze_effector_updates:
        frozen_effector_parameters = freeze_organism_parameter_group(
            system, "effector"
        )
    optimizer = torch.optim.AdamW(
        organism_optimizer_groups(system, train_config), weight_decay=0.0
    )
    output_codebook = fixed_output_codebook(
        organism_config.hidden,
        seed=train_config.output_credit_seed,
        device=args.device,
    )
    memory_semantic_projection = fixed_semantic_projection(
        system.backbone.width,
        organism_config.hidden,
        seed=train_config.semantic_credit_seed,
        device=args.device,
    )
    relay_semantic_projection = fixed_semantic_projection(
        system.backbone.width,
        organism_config.hidden,
        seed=train_config.semantic_credit_seed + 1,
        device=args.device,
    )
    recall_semantic_projection = fixed_semantic_projection(
        system.backbone.width,
        organism_config.hidden,
        seed=train_config.semantic_credit_seed + 2,
        device=args.device,
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
        checkpoint_policy = payload["train_config"].get(
            "lifetime_lane_policy", "continuous"
        )
        if checkpoint_policy != train_config.lifetime_lane_policy:
            raise ValueError(
                "checkpoint lifetime lane policy does not match requested policy"
            )

    schedule = _flatten_split(corpus, "meta_train")
    while update < train_config.updates:
        episode_state = training_episode_start_state(
            system,
            lane_state,
            policy=train_config.lifetime_lane_policy,
        )
        episode_start_memory_cursor = episode_state.memory_cursor
        document, question = schedule[update % len(schedule)]
        epoch = update // len(schedule)
        if train_config.paired_counterfactual:
            training_records = counterfactual_training_pair(
                document,
                question,
                epoch=epoch,
                compact=train_config.compact_bindings,
            )
        else:
            training_records = (
                counterfactual_training_record(
                    document,
                    question,
                    epoch=epoch,
                    compact=train_config.compact_bindings,
                ),
            )
        optimizer.zero_grad(set_to_none=True)
        permutation_seed = train_config.permutation_seed + update
        formatted_question = format_wiki_question(
            training_records[0][1], permutation_seed=permutation_seed
        )
        training_question_ids = encode_text(
            tokenizer,
            formatted_question.prompt,
            episode_state.hidden.device,
            max_tokens=train_config.max_question_tokens,
        )
        training_question_features = system.backbone.encode(training_question_ids)
        training_exposure_features = [
            system.backbone.encode(
                encode_text(
                    tokenizer,
                    training_document.memory,
                    episode_state.hidden.device,
                    max_tokens=train_config.max_memory_tokens,
                )
            )
            for training_document, _ in training_records
        ]
        teacher_features: list[torch.Tensor] = []
        teacher_logits: list[torch.Tensor] = []
        teacher_enabled = (
            train_config.teacher_distillation_weight > 0
            or train_config.teacher_effect_distillation_weight > 0
            or train_config.relay_semantic_credit_weight > 0
            or train_config.recall_semantic_credit_weight > 0
            or train_config.recall_addressing_credit_weight > 0
        )
        if teacher_enabled:
            teacher_outputs = [
                passage_visible_teacher_outputs(
                    system,
                    tokenizer,
                    training_document,
                    training_question,
                    permutation_seed=permutation_seed,
                    max_memory_tokens=train_config.max_memory_tokens,
                    max_question_tokens=train_config.max_question_tokens,
                )
                for training_document, training_question in training_records
            ]
            teacher_features = [output[0] for output in teacher_outputs]
            teacher_logits = [output[1] for output in teacher_outputs]
        teacher_baseline_features = None
        teacher_baseline_logits = None
        if (
            train_config.teacher_effect_distillation_weight > 0
            or train_config.relay_semantic_credit_weight > 0
            or train_config.recall_semantic_credit_weight > 0
            or train_config.recall_addressing_credit_weight > 0
        ):
            (
                teacher_baseline_features,
                teacher_baseline_logits,
            ) = question_only_teacher_outputs(
                system,
                tokenizer,
                training_records[0][1],
                permutation_seed=permutation_seed,
                max_question_tokens=train_config.max_question_tokens,
            )
        control_reference = None
        if train_config.reference_centered_controls:
            with torch.no_grad():
                raw_reference = run_wiki_memory_episode(
                    system,
                    tokenizer,
                    training_records[0][0],
                    training_records[0][1],
                    episode_state,
                    permutation_seed=permutation_seed,
                    expose=False,
                    question_features=training_question_features,
                    question_ids=training_question_ids,
                    max_memory_tokens=train_config.max_memory_tokens,
                    max_question_tokens=train_config.max_question_tokens,
                    control_mode=train_config.language_control_mode,
                )
            if raw_reference.controls is None:
                raise RuntimeError("reference episode did not emit controls")
            control_reference = raw_reference.controls.detach()
        branch_results = []
        for (
            (training_document, training_question),
            exposure_features,
        ) in zip(training_records, training_exposure_features):
            branch_result = run_wiki_memory_episode(
                system,
                tokenizer,
                training_document,
                training_question,
                episode_state,
                permutation_seed=permutation_seed,
                exposure_features=exposure_features,
                question_features=training_question_features,
                question_ids=training_question_ids,
                max_memory_tokens=train_config.max_memory_tokens,
                max_question_tokens=train_config.max_question_tokens,
                control_mode=train_config.language_control_mode,
                control_reference=control_reference,
                capture_recall=(
                    train_config.recall_semantic_credit_weight > 0
                    or train_config.recall_addressing_credit_weight > 0
                    or train_config.coherent_recall
                ),
            )
            branch_results.append(branch_result)
        no_exposure_result = None
        if (
            train_config.causal_contrast_weight > 0
            or train_config.teacher_effect_distillation_weight > 0
        ):
            with torch.no_grad():
                no_exposure_result = run_wiki_memory_episode(
                    system,
                    tokenizer,
                    training_records[0][0],
                    training_records[0][1],
                    episode_state,
                    permutation_seed=permutation_seed,
                    expose=False,
                    question_features=training_question_features,
                    question_ids=training_question_ids,
                    control_scale=(
                        0.0 if train_config.reference_centered_controls else 1.0
                    ),
                    max_memory_tokens=train_config.max_memory_tokens,
                    max_question_tokens=train_config.max_question_tokens,
                    control_mode=train_config.language_control_mode,
                    control_reference=control_reference,
                )
        task_loss = torch.stack([branch.loss for branch in branch_results]).mean()
        teacher_distillation_loss = task_loss.new_zeros(())
        teacher_effect_distillation_loss = task_loss.new_zeros(())
        teacher_label_accuracy = task_loss.new_zeros(())
        teacher_logit_separation_rms = task_loss.new_zeros(())
        teacher_effect_rms = task_loss.new_zeros(())
        student_effect_rms = task_loss.new_zeros(())
        teacher_effect_pair_separation_rms = task_loss.new_zeros(())
        student_effect_pair_separation_rms = task_loss.new_zeros(())
        memory_semantic_credit_loss = task_loss.new_zeros(())
        memory_semantic_separation_rms = task_loss.new_zeros(())
        memory_semantic_alignment = task_loss.new_zeros(())
        memory_teacher_projection_rms = task_loss.new_zeros(())
        relay_semantic_credit_loss = task_loss.new_zeros(())
        relay_semantic_separation_rms = task_loss.new_zeros(())
        relay_semantic_alignment = task_loss.new_zeros(())
        relay_teacher_projection_rms = task_loss.new_zeros(())
        recall_semantic_credit_loss = task_loss.new_zeros(())
        recall_semantic_separation_rms = task_loss.new_zeros(())
        recall_semantic_alignment = task_loss.new_zeros(())
        recall_teacher_projection_rms = task_loss.new_zeros(())
        recall_addressing_credit_loss = task_loss.new_zeros(())
        recall_selected_teacher_mass = task_loss.new_zeros(())
        recall_attention_entropy = task_loss.new_zeros(())
        recall_effective_slots = task_loss.new_zeros(())
        recall_newest_mass = task_loss.new_zeros(())
        if teacher_enabled:
            if len(teacher_logits) != len(branch_results):
                raise RuntimeError("teacher/student branch count mismatch")
            if any(branch.vocab_logits is None for branch in branch_results):
                raise RuntimeError("student episode did not retain vocabulary logits")
        if train_config.teacher_distillation_weight > 0:
            teacher_distillation_loss = torch.stack(
                [
                    full_vocab_reverse_kl(
                        branch.vocab_logits,
                        teacher,
                        temperature=train_config.teacher_distillation_temperature,
                    )
                    for branch, teacher in zip(branch_results, teacher_logits)
                ]
            ).mean()
        if teacher_enabled:
            labels = label_token_ids(tokenizer, episode_state.hidden.device)
            teacher_correct = []
            for (_, teacher_question), teacher in zip(
                training_records, teacher_logits
            ):
                formatted = format_wiki_question(
                    teacher_question, permutation_seed=permutation_seed
                )
                teacher_correct.append(
                    teacher.index_select(-1, labels).argmax(dim=-1)
                    == formatted.correct_index
                )
            teacher_label_accuracy = torch.stack(teacher_correct).float().mean()
            if len(teacher_logits) == 2:
                teacher_logit_separation_rms = (
                    teacher_logits[0].float() - teacher_logits[1].float()
                ).pow(2).mean().sqrt()
        if train_config.teacher_effect_distillation_weight > 0:
            if teacher_baseline_logits is None:
                raise RuntimeError("teacher effect requires question-only baseline")
            if no_exposure_result is None or no_exposure_result.vocab_logits is None:
                raise RuntimeError("teacher effect requires student zero baseline")
            student_baseline_logits = no_exposure_result.vocab_logits
            teacher_effect_distillation_loss = torch.stack(
                [
                    full_vocab_effect_reverse_kl(
                        branch.vocab_logits,
                        student_baseline_logits,
                        teacher,
                        teacher_baseline_logits,
                        temperature=train_config.teacher_distillation_temperature,
                    )
                    for branch, teacher in zip(branch_results, teacher_logits)
                ]
            ).mean()
            teacher_effects = [
                teacher.float() - teacher_baseline_logits.float()
                for teacher in teacher_logits
            ]
            student_effects = [
                branch.vocab_logits.float() - student_baseline_logits.float()
                for branch in branch_results
            ]
            teacher_effect_rms = torch.stack(
                [effect.pow(2).mean().sqrt() for effect in teacher_effects]
            ).mean()
            student_effect_rms = torch.stack(
                [effect.pow(2).mean().sqrt() for effect in student_effects]
            ).mean()
            if len(teacher_effects) == 2:
                teacher_effect_pair_separation_rms = (
                    teacher_effects[0] - teacher_effects[1]
                ).pow(2).mean().sqrt()
                student_effect_pair_separation_rms = (
                    student_effects[0] - student_effects[1]
                ).pow(2).mean().sqrt()
        if train_config.memory_semantic_credit_weight > 0:
            if len(branch_results) != 2 or len(training_exposure_features) != 2:
                raise RuntimeError("memory semantic credit requires a paired update")
            if any(
                branch.exposure_memory_state is None for branch in branch_results
            ):
                raise RuntimeError("memory semantic credit requires exposure state")
            (
                memory_semantic_credit_loss,
                memory_semantic_separation_rms,
                memory_semantic_alignment,
                memory_teacher_projection_rms,
            ) = paired_tissue_semantic_credit(
                branch_results[0].exposure_memory_state,
                branch_results[1].exposure_memory_state,
                training_exposure_features[0][:, -1],
                training_exposure_features[1][:, -1],
                memory_semantic_projection,
                target_rms=train_config.semantic_credit_target_rms,
            )
        if train_config.relay_semantic_credit_weight > 0:
            if (
                len(branch_results) != 2
                or len(teacher_features) != 2
                or teacher_baseline_features is None
            ):
                raise RuntimeError("relay semantic credit requires paired teacher effects")
            if any(branch.relay_state is None for branch in branch_results):
                raise RuntimeError("relay semantic credit requires relay state")
            teacher_feature_effects = [
                feature.float() - teacher_baseline_features.float()
                for feature in teacher_features
            ]
            (
                relay_semantic_credit_loss,
                relay_semantic_separation_rms,
                relay_semantic_alignment,
                relay_teacher_projection_rms,
            ) = paired_tissue_semantic_credit(
                branch_results[0].relay_state,
                branch_results[1].relay_state,
                teacher_feature_effects[0],
                teacher_feature_effects[1],
                relay_semantic_projection,
                target_rms=train_config.semantic_credit_target_rms,
            )
        if train_config.recall_semantic_credit_weight > 0:
            if (
                len(branch_results) != 2
                or len(teacher_features) != 2
                or teacher_baseline_features is None
            ):
                raise RuntimeError("recall semantic credit requires paired teacher effects")
            if any(branch.recalled is None for branch in branch_results):
                raise RuntimeError("recall semantic credit requires live recall vectors")
            teacher_feature_effects = [
                feature.float() - teacher_baseline_features.float()
                for feature in teacher_features
            ]
            (
                recall_semantic_credit_loss,
                recall_semantic_separation_rms,
                recall_semantic_alignment,
                recall_teacher_projection_rms,
            ) = paired_vector_semantic_credit(
                branch_results[0].recalled,
                branch_results[1].recalled,
                teacher_feature_effects[0],
                teacher_feature_effects[1],
                recall_semantic_projection,
                target_rms=train_config.semantic_credit_target_rms,
            )
        if train_config.recall_addressing_credit_weight > 0:
            if teacher_baseline_features is None or len(teacher_features) != len(
                branch_results
            ):
                raise RuntimeError("recall addressing credit requires teacher effects")
            if any(
                branch.recall_content_scores is None
                or branch.recall_attention is None
                for branch in branch_results
            ):
                raise RuntimeError("recall addressing credit requires live attention")
            addressing_rows = [
                recall_addressing_credit(
                    branch.recall_content_scores,
                    branch.recall_attention,
                    exposure_features,
                    teacher_feature.float() - teacher_baseline_features.float(),
                    target_temperature=train_config.recall_addressing_temperature,
                )
                for branch, exposure_features, teacher_feature in zip(
                    branch_results, training_exposure_features, teacher_features
                )
            ]
            (
                recall_addressing_credit_loss,
                recall_selected_teacher_mass,
                recall_attention_entropy,
                recall_effective_slots,
                recall_newest_mass,
            ) = tuple(
                torch.stack([row[index] for row in addressing_rows]).mean()
                for index in range(5)
            )
        if any(branch.controls is None for branch in branch_results):
            raise RuntimeError("student episode did not emit controls")
        control_energy_loss = torch.stack(
            [control_delta_energy(branch.controls) for branch in branch_results]
        ).mean()
        binding_loss = task_loss.new_zeros(())
        binding_preferences = task_loss.new_zeros(2)
        output_credit_loss = task_loss.new_zeros(())
        output_credit_accuracy = task_loss.new_zeros(())
        output_state_separation_rms = task_loss.new_zeros(())
        output_eligibility_loss = task_loss.new_zeros(())
        output_target_projection = task_loss.new_zeros(())
        relay_state_separation_rms = task_loss.new_zeros(())
        output_transport_ratio = task_loss.new_zeros(())
        output_eligibility_gate = task_loss.new_zeros(())
        causal_contrast_loss = task_loss.new_zeros(())
        causal_contrast_advantages = task_loss.new_zeros(4)
        if len(branch_results) == 2 and train_config.binding_weight > 0:
            binding_loss, binding_preferences = paired_binding_margin_loss(
                branch_results[0],
                branch_results[1],
                margin=train_config.binding_margin,
            )
        if len(branch_results) == 2:
            (
                measured_output_credit_loss,
                output_credit_accuracy,
                output_state_separation_rms,
            ) = output_tissue_credit_loss(
                branch_results,
                codebook=output_codebook,
                scale=train_config.output_credit_scale,
            )
            if train_config.output_credit_weight > 0:
                output_credit_loss = measured_output_credit_loss
            (
                measured_output_eligibility_loss,
                output_target_projection,
                relay_state_separation_rms,
                measured_output_state_separation_rms,
                output_transport_ratio,
                output_eligibility_gate,
            ) = eligibility_gated_transport_loss(
                branch_results,
                codebook=output_codebook,
                relay_reference=train_config.output_eligibility_relay_reference,
                margin=train_config.output_eligibility_margin,
                temperature=train_config.output_eligibility_temperature,
            )
            output_state_separation_rms = measured_output_state_separation_rms
            if train_config.output_eligibility_weight > 0:
                output_eligibility_loss = measured_output_eligibility_loss
        if train_config.causal_contrast_weight > 0:
            assert no_exposure_result is not None
            causal_contrast_loss, causal_contrast_advantages = (
                passage_causal_contrast_loss(
                    branch_results[0],
                    branch_results[1],
                    no_exposure_result,
                    margin=train_config.causal_contrast_margin,
                )
            )
        objective_loss = (
            train_config.task_weight * task_loss
            + train_config.binding_weight * binding_loss
            + train_config.causal_contrast_weight * causal_contrast_loss
            + train_config.teacher_distillation_weight
            * teacher_distillation_loss
            + train_config.teacher_effect_distillation_weight
            * teacher_effect_distillation_loss
            + train_config.memory_semantic_credit_weight
            * memory_semantic_credit_loss
            + train_config.relay_semantic_credit_weight
            * relay_semantic_credit_loss
            + train_config.recall_semantic_credit_weight
            * recall_semantic_credit_loss
            + train_config.recall_addressing_credit_weight
            * recall_addressing_credit_loss
            + train_config.control_energy_weight * control_energy_loss
            + train_config.output_credit_weight * output_credit_loss
            + train_config.output_eligibility_weight * output_eligibility_loss
        )
        objective_loss.backward()
        gradient_groups = organism_gradient_groups(system)
        recall_gradients = recall_gradient_components(system)
        require_finite_organism_gradients(system)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            system.organism.parameters(), train_config.grad_clip
        )
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(
                f"non-finite aggregate gradient norm before optimizer step: {grad_norm}"
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
            "lifetime_lane_policy": train_config.lifetime_lane_policy,
            "episode_start_memory_cursor": episode_start_memory_cursor,
            "episode_end_memory_cursor": result.state.memory_cursor,
            "document_id": document.id,
            "question_id": question.id,
            "loss": mean_loss,
            "bits": mean_loss / math.log(2.0),
            "objective_loss": float(objective_loss.detach()),
            "binding_loss": float(binding_loss.detach()),
            "binding_preferences": [
                float(value) for value in binding_preferences.detach()
            ],
            "causal_contrast_loss": float(causal_contrast_loss.detach()),
            "causal_contrast_advantages": [
                float(value) for value in causal_contrast_advantages.detach()
            ],
            "causal_contrast_accuracy": float(
                (causal_contrast_advantages.detach() > 0).float().mean()
            ),
            "teacher_distillation_loss": float(
                teacher_distillation_loss.detach()
            ),
            "teacher_effect_distillation_loss": float(
                teacher_effect_distillation_loss.detach()
            ),
            "teacher_label_accuracy": float(teacher_label_accuracy.detach()),
            "teacher_logit_separation_rms": float(
                teacher_logit_separation_rms.detach()
            ),
            "teacher_effect_rms": float(teacher_effect_rms.detach()),
            "student_effect_rms": float(student_effect_rms.detach()),
            "teacher_effect_pair_separation_rms": float(
                teacher_effect_pair_separation_rms.detach()
            ),
            "student_effect_pair_separation_rms": float(
                student_effect_pair_separation_rms.detach()
            ),
            "memory_semantic_credit_loss": float(
                memory_semantic_credit_loss.detach()
            ),
            "memory_semantic_separation_rms": float(
                memory_semantic_separation_rms.detach()
            ),
            "memory_semantic_alignment": float(
                memory_semantic_alignment.detach()
            ),
            "memory_teacher_projection_rms": float(
                memory_teacher_projection_rms.detach()
            ),
            "relay_semantic_credit_loss": float(
                relay_semantic_credit_loss.detach()
            ),
            "relay_semantic_separation_rms": float(
                relay_semantic_separation_rms.detach()
            ),
            "relay_semantic_alignment": float(
                relay_semantic_alignment.detach()
            ),
            "relay_teacher_projection_rms": float(
                relay_teacher_projection_rms.detach()
            ),
            "recall_semantic_credit_loss": float(
                recall_semantic_credit_loss.detach()
            ),
            "recall_semantic_separation_rms": float(
                recall_semantic_separation_rms.detach()
            ),
            "recall_semantic_alignment": float(recall_semantic_alignment.detach()),
            "recall_teacher_projection_rms": float(
                recall_teacher_projection_rms.detach()
            ),
            "recall_addressing_credit_loss": float(
                recall_addressing_credit_loss.detach()
            ),
            "recall_selected_teacher_mass": float(
                recall_selected_teacher_mass.detach()
            ),
            "recall_attention_entropy": float(recall_attention_entropy.detach()),
            "recall_effective_slots": float(recall_effective_slots.detach()),
            "recall_newest_mass": float(recall_newest_mass.detach()),
            "control_energy_loss": float(control_energy_loss.detach()),
            "no_exposure_loss": (
                0.0
                if no_exposure_result is None
                else float(no_exposure_result.loss.detach())
            ),
            "no_exposure_correct": (
                False if no_exposure_result is None else no_exposure_result.correct
            ),
            "output_credit_loss": float(output_credit_loss.detach()),
            "output_credit_accuracy": float(output_credit_accuracy.detach()),
            "output_state_separation_rms": float(
                output_state_separation_rms.detach()
            ),
            "output_eligibility_loss": float(output_eligibility_loss.detach()),
            "output_target_projection": float(output_target_projection.detach()),
            "relay_state_separation_rms": float(
                relay_state_separation_rms.detach()
            ),
            "output_transport_ratio": float(output_transport_ratio.detach()),
            "output_eligibility_gate": float(output_eligibility_gate.detach()),
            "relay_output_gate_rms": (
                0.0
                if system.organism.relay_output_transport is None
                else system.organism.relay_output_transport.gate_rms()
            ),
            "correct": result.correct,
            "branch_accuracy": branch_accuracy,
            "branch_answers": [branch.correct_label for branch in branch_results],
            "control_separation_rms": control_separation_rms,
            "label_logit_separation_rms": label_logit_separation_rms,
            "grad_norm": float(grad_norm),
            "gradient_groups": gradient_groups,
            "recall_gradient_components": recall_gradients,
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
                f"causal={row['causal_contrast_loss']:.4f} "
                f"causal_acc={row['causal_contrast_accuracy']:.2f} "
                f"distill={row['teacher_distillation_loss']:.4f} "
                f"effect={row['teacher_effect_distillation_loss']:.4f} "
                f"teacher_effect={row['teacher_effect_rms']:.4f} "
                f"student_effect={row['student_effect_rms']:.4f} "
                f"memory_sem={row['memory_semantic_credit_loss']:.4f} "
                f"memory_align={row['memory_semantic_alignment']:.3f} "
                f"relay_sem={row['relay_semantic_credit_loss']:.4f} "
                f"relay_align={row['relay_semantic_alignment']:.3f} "
                f"recall_sem={row['recall_semantic_credit_loss']:.4f} "
                f"recall_align={row['recall_semantic_alignment']:.3f} "
                f"address_kl={row['recall_addressing_credit_loss']:.4f} "
                f"selected={row['recall_selected_teacher_mass']:.3f} "
                f"attn_eff={row['recall_effective_slots']:.2f} "
                f"newest={row['recall_newest_mass']:.3f} "
                f"teacher_acc={row['teacher_label_accuracy']:.2f} "
                f"energy={row['control_energy_loss']:.6f} "
                f"output_credit={row['output_credit_loss']:.4f} "
                f"output_sep={row['output_state_separation_rms']:.4f} "
                f"eligibility={row['output_eligibility_loss']:.4f} "
                f"target_proj={row['output_target_projection']:.4f} "
                f"relay_sep={row['relay_state_separation_rms']:.4f} "
                f"transport={row['output_transport_ratio']:.3f} "
                f"tract_gate={row['relay_output_gate_rms']:.4f} "
                f"cursor={row['episode_start_memory_cursor']}"
                f"->{row['episode_end_memory_cursor']} "
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
                control_mode=train_config.language_control_mode,
                reference_centered_controls=(
                    train_config.reference_centered_controls
                ),
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
        control_mode=train_config.language_control_mode,
        reference_centered_controls=train_config.reference_centered_controls,
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
        "frozen_effector_parameters": frozen_effector_parameters,
        "optimizer_groups": [
            {
                "group_name": group["group_name"],
                "lr": group["lr"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        ],
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
        "--lifetime-lane-policy",
        choices=("continuous", "fresh_episode"),
        default="continuous",
    )
    parser.add_argument(
        "--paired-counterfactual",
        action="store_true",
        help="train two incompatible passage bindings from one matched start state",
    )
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--binding-margin", type=float, default=1.0)
    parser.add_argument("--binding-weight", type=float, default=1.0)
    parser.add_argument("--causal-contrast-weight", type=float, default=0.0)
    parser.add_argument("--causal-contrast-margin", type=float, default=0.1)
    parser.add_argument("--teacher-distillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-effect-distillation-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--teacher-distillation-temperature", type=float, default=1.0
    )
    parser.add_argument("--memory-semantic-credit-weight", type=float, default=0.0)
    parser.add_argument("--relay-semantic-credit-weight", type=float, default=0.0)
    parser.add_argument("--recall-semantic-credit-weight", type=float, default=0.0)
    parser.add_argument("--recall-addressing-credit-weight", type=float, default=0.0)
    parser.add_argument("--recall-addressing-temperature", type=float, default=0.1)
    parser.add_argument("--semantic-credit-target-rms", type=float, default=0.25)
    parser.add_argument("--semantic-credit-seed", type=int, default=307)
    parser.add_argument("--control-energy-weight", type=float, default=0.0)
    parser.add_argument("--reference-centered-controls", action="store_true")
    parser.add_argument("--freeze-effector-updates", action="store_true")
    parser.add_argument(
        "--language-control-mode",
        choices=("late_residual", "prefix"),
        default="late_residual",
    )
    parser.add_argument("--compact-bindings", action="store_true")
    parser.add_argument("--control-gain", type=float, default=1.0)
    parser.add_argument("--recall-gain", type=float, default=0.25)
    parser.add_argument("--coherent-recall", action="store_true")
    parser.add_argument("--recall-residual-gain", type=float, default=0.1)
    parser.add_argument("--recall-top-k", type=int, default=0)
    parser.add_argument("--recall-recency-bias", type=float, default=0.0)
    parser.add_argument("--recall-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--sensor-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--effector-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--output-credit-weight", type=float, default=0.0)
    parser.add_argument("--output-credit-scale", type=float, default=4.0)
    parser.add_argument("--output-credit-seed", type=int, default=211)
    parser.add_argument("--output-eligibility-weight", type=float, default=0.0)
    parser.add_argument("--output-eligibility-margin", type=float, default=0.005)
    parser.add_argument("--output-eligibility-temperature", type=float, default=0.005)
    parser.add_argument(
        "--output-eligibility-relay-reference", type=float, default=0.005
    )
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
    parser.add_argument("--relay-output-attention", action="store_true")
    parser.add_argument("--relay-output-gain", type=float, default=1.0)
    parser.add_argument("--relay-output-initial-gate", type=float, default=0.0)
    parser.add_argument("--control-tokens", type=int, default=8)
    parser.add_argument("--control-rank", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    metrics = run_training(parse_args())
    print(json.dumps(metrics["heldout"]["summaries"], indent=2))


if __name__ == "__main__":
    main()
