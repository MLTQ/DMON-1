"""Provenance, formatting, and causal episodes for held-out wiki memory."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .living_language import LivingLanguageSystem
from .state import OrganismState


LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class WikiQuestion:
    id: str
    question: str
    choices: tuple[str, str, str, str]
    answer: int
    paraphrase: str


@dataclass(frozen=True)
class WikiDocument:
    id: str
    split: str
    source_family: str
    path: str
    sha256: str
    memory: str
    questions: tuple[WikiQuestion, ...]


@dataclass(frozen=True)
class WikiMemoryCorpus:
    wiki_root: str
    documents: tuple[WikiDocument, ...]

    def split(self, name: str) -> tuple[WikiDocument, ...]:
        return tuple(document for document in self.documents if document.split == name)


@dataclass(frozen=True)
class FormattedQuestion:
    prompt: str
    correct_label: str
    correct_index: int
    order: tuple[int, int, int, int]


@dataclass
class WikiEpisodeResult:
    loss: torch.Tensor
    correct: bool
    predicted_label: str
    correct_label: str
    label_log_probs: dict[str, float]
    state: OrganismState
    control_rms: float
    internal_activity: float
    memory_activity: float
    controls: torch.Tensor | None = None
    label_logits: torch.Tensor | None = None
    vocab_logits: torch.Tensor | None = None
    relay_state: torch.Tensor | None = None
    output_state: torch.Tensor | None = None


def load_wiki_memory_corpus(path: Path) -> WikiMemoryCorpus:
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1:
        raise ValueError("unsupported wiki-memory corpus schema")
    documents = []
    source_splits: dict[str, str] = {}
    question_ids: set[str] = set()
    for raw_document in payload["documents"]:
        split = raw_document["split"]
        if split not in {"meta_train", "development", "heldout"}:
            raise ValueError(f"unknown wiki-memory split {split!r}")
        family = raw_document["source_family"]
        previous = source_splits.setdefault(family, split)
        if previous != split:
            raise ValueError(f"source family {family!r} leaks across splits")
        questions = []
        for raw_question in raw_document["questions"]:
            if raw_question["id"] in question_ids:
                raise ValueError(f"duplicate question id {raw_question['id']!r}")
            question_ids.add(raw_question["id"])
            choices = tuple(raw_question["choices"])
            if len(choices) != 4 or len(set(choices)) != 4:
                raise ValueError("every question must have four distinct choices")
            answer = int(raw_question["answer"])
            if answer not in range(4):
                raise ValueError("question answer must index one of four choices")
            questions.append(
                WikiQuestion(
                    id=raw_question["id"],
                    question=raw_question["question"],
                    choices=choices,
                    answer=answer,
                    paraphrase=raw_question["paraphrase"],
                )
            )
        documents.append(
            WikiDocument(
                id=raw_document["id"],
                split=split,
                source_family=family,
                path=raw_document["path"],
                sha256=raw_document["sha256"],
                memory=raw_document["memory"],
                questions=tuple(questions),
            )
        )
    return WikiMemoryCorpus(
        wiki_root=payload["wiki_root"], documents=tuple(documents)
    )


def verify_wiki_sources(corpus: WikiMemoryCorpus, root: Path | None = None) -> None:
    source_root = Path(corpus.wiki_root) if root is None else root
    for document in corpus.documents:
        source = source_root / document.path
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != document.sha256:
            raise ValueError(f"wiki source hash changed: {document.path}")


def _permutation(question_id: str, seed: int) -> tuple[int, int, int, int]:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
    return tuple(int(index) for index in torch.randperm(4, generator=generator))


def format_wiki_question(
    question: WikiQuestion,
    *,
    permutation_seed: int,
    paraphrase: bool = False,
) -> FormattedQuestion:
    order = _permutation(question.id, permutation_seed)
    wording = question.paraphrase if paraphrase else question.question
    displayed = [question.choices[index] for index in order]
    correct_index = order.index(question.answer)
    lines = [
        "Select the correct answer. Reply with only A, B, C, or D.",
        f"Question: {wording}",
        *(f"{label}. {choice}" for label, choice in zip(LABELS, displayed)),
        "Answer:",
    ]
    return FormattedQuestion(
        prompt="\n".join(lines),
        correct_label=LABELS[correct_index],
        correct_index=correct_index,
        order=order,
    )


def label_token_ids(tokenizer: Any, device: torch.device | str) -> torch.Tensor:
    ids = []
    for label in LABELS:
        encoded = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"label {label!r} is not one token for this tokenizer")
        ids.append(encoded[0])
    return torch.tensor(ids, dtype=torch.long, device=device)


def encode_text(
    tokenizer: Any,
    text: str,
    device: torch.device | str,
    *,
    max_tokens: int,
) -> torch.Tensor:
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if input_ids.shape[1] > max_tokens:
        raise ValueError(
            f"text encoded to {input_ids.shape[1]} tokens, above limit {max_tokens}"
        )
    return input_ids.to(device)


def lesion_state(state: OrganismState, indices: torch.Tensor) -> OrganismState:
    hidden = state.hidden.clone()
    hidden[:, indices] = 0
    return OrganismState(
        hidden=hidden,
        memory_cursor=state.memory_cursor,
        weight_version=state.weight_version,
    )


def _select_label_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.index_select(-1, labels)


@torch.no_grad()
def score_frozen_baseline(
    system: LivingLanguageSystem,
    tokenizer: Any,
    question: WikiQuestion,
    *,
    permutation_seed: int,
    paraphrase: bool = False,
    max_question_tokens: int = 256,
) -> WikiEpisodeResult:
    device = next(system.organism.parameters()).device
    formatted = format_wiki_question(
        question, permutation_seed=permutation_seed, paraphrase=paraphrase
    )
    input_ids = encode_text(
        tokenizer, formatted.prompt, device, max_tokens=max_question_tokens
    )
    features = system.backbone.encode(input_ids)
    logits = system.backbone.controlled_logits_from_features(features[:, -1:])[:, -1]
    labels = label_token_ids(tokenizer, device)
    label_logits = _select_label_logits(logits, labels)
    loss = F.cross_entropy(
        label_logits, torch.tensor([formatted.correct_index], device=device)
    )
    log_probs = label_logits.log_softmax(dim=-1)[0]
    predicted = int(label_logits.argmax(dim=-1))
    empty_state = system.initial_state(1, device)
    return WikiEpisodeResult(
        loss=loss,
        correct=predicted == formatted.correct_index,
        predicted_label=LABELS[predicted],
        correct_label=formatted.correct_label,
        label_log_probs={label: float(log_probs[index]) for index, label in enumerate(LABELS)},
        state=empty_state,
        control_rms=0.0,
        internal_activity=0.0,
        memory_activity=0.0,
        controls=None,
        label_logits=label_logits[0],
    )


def run_wiki_memory_episode(
    system: LivingLanguageSystem,
    tokenizer: Any,
    document: WikiDocument,
    question: WikiQuestion,
    state: OrganismState,
    *,
    permutation_seed: int,
    exposure_document: WikiDocument | None = None,
    exposure_features: torch.Tensor | None = None,
    question_features: torch.Tensor | None = None,
    question_ids: torch.Tensor | None = None,
    expose: bool = True,
    reset_after_exposure: bool = False,
    memory_lesion: bool = False,
    control_scale: float = 1.0,
    frozen_idx: torch.Tensor | None = None,
    paraphrase: bool = False,
    max_memory_tokens: int = 256,
    max_question_tokens: int = 256,
    collect_health: bool = False,
    control_mode: str = "late_residual",
    control_reference: torch.Tensor | None = None,
) -> WikiEpisodeResult:
    if state.hidden.shape[0] != 1:
        raise ValueError("wiki memory episodes currently require one lifetime lane")
    device = state.hidden.device
    next_state = state
    if expose:
        exposed = document if exposure_document is None else exposure_document
        if exposure_features is None:
            memory_ids = encode_text(
                tokenizer, exposed.memory, device, max_tokens=max_memory_tokens
            )
            next_state, _ = system.observe_sequence(memory_ids, next_state)
        else:
            next_state, _ = system.observe_feature_sequence(
                exposure_features, next_state
            )
    if reset_after_exposure:
        next_state = system.initial_state(state.hidden.shape[0], device)
    elif memory_lesion:
        next_state = lesion_state(next_state, system.organism.memory_idx)

    formatted = format_wiki_question(
        question, permutation_seed=permutation_seed, paraphrase=paraphrase
    )
    if question_ids is None:
        question_ids = encode_text(
            tokenizer, formatted.prompt, device, max_tokens=max_question_tokens
        )
    if question_features is None:
        scored = system.score_next_after_sequence(
            question_ids,
            next_state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
            write_memory=False,
            collect_health=collect_health,
            control_mode=control_mode,
            control_reference=control_reference,
        )
    else:
        scored = system.score_next_from_features(
            question_features,
            next_state,
            control_scale=control_scale,
            frozen_idx=frozen_idx,
            write_memory=False,
            collect_health=collect_health,
            control_mode=control_mode,
            input_ids=question_ids,
            control_reference=control_reference,
        )
    labels = label_token_ids(tokenizer, device)
    label_logits = _select_label_logits(scored.logits, labels)
    target = torch.tensor([formatted.correct_index], device=device)
    loss = F.cross_entropy(label_logits, target)
    log_probs = label_logits.detach().log_softmax(dim=-1)[0]
    predicted = int(label_logits.detach().argmax(dim=-1))
    internal = scored.state.hidden[:, system.organism.internal_idx]
    memory = scored.state.hidden[:, system.organism.memory_idx]
    relay = scored.state.hidden[:, system.organism.relay_idx]
    output = scored.state.hidden[:, system.organism.output_idx]
    return WikiEpisodeResult(
        loss=loss,
        correct=predicted == formatted.correct_index,
        predicted_label=LABELS[predicted],
        correct_label=formatted.correct_label,
        label_log_probs={label: float(log_probs[index]) for index, label in enumerate(LABELS)},
        state=scored.state,
        control_rms=float(scored.controls.detach().pow(2).mean().sqrt()),
        internal_activity=float(internal.detach().pow(2).mean().sqrt()),
        memory_activity=float(memory.detach().pow(2).mean().sqrt()),
        controls=scored.controls,
        label_logits=label_logits[0],
        vocab_logits=scored.logits[0],
        relay_state=relay,
        output_state=output,
    )


def summarize_results(results: list[WikiEpisodeResult]) -> dict[str, float]:
    if not results:
        raise ValueError("cannot summarize an empty result list")
    return {
        "count": float(len(results)),
        "accuracy": sum(result.correct for result in results) / len(results),
        "mean_loss": sum(float(result.loss.detach()) for result in results) / len(results),
        "mean_bits": sum(
            float(result.loss.detach()) / math.log(2.0) for result in results
        )
        / len(results),
        "control_rms": sum(result.control_rms for result in results) / len(results),
        "internal_activity": sum(result.internal_activity for result in results)
        / len(results),
        "memory_activity": sum(result.memory_activity for result in results)
        / len(results),
    }
