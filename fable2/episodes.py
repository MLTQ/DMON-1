"""Paired wiki episodes over passage prose with cached frozen features."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol2.wiki_memory import (
    WikiDocument,
    WikiMemoryCorpus,
    WikiQuestion,
    encode_text,
    format_wiki_question,
    label_token_ids,
    lesion_state,
)

from .system import BrocaSystem, EpisodeScore

# The C1 line trained on synthetic "Designated answer: <choice>" cards and evaluated
# on prose — a training/evaluation distribution shift its instruments never flagged.
# fable2 exposures are the passage prose itself; these markers must never appear.
FORBIDDEN_EXPOSURE_MARKERS = ("Designated answer",)

EVAL_ARMS = (
    "normal",
    "wrong_passage",
    "no_exposure",
    "memory_lesion",
    "internal_lesion",
    "bare_floor",
)


@dataclass(frozen=True)
class EpisodeItem:
    document_id: str
    question_id: str
    split: str
    source_family: str
    exposure_features: torch.Tensor
    question_features: torch.Tensor
    question_ids: torch.Tensor
    correct_index: int
    mate_question_id: str


class EpisodeBank:
    """Cache frozen features once; serve deterministic paired episodes forever."""

    def __init__(
        self,
        corpus: WikiMemoryCorpus,
        tokenizer,
        backbone,
        device: torch.device | str,
        *,
        permutation_seed: int,
        max_memory_tokens: int,
        max_question_tokens: int,
    ) -> None:
        self.device = torch.device(device)
        self.labels = label_token_ids(tokenizer, self.device)
        self.items: dict[str, EpisodeItem] = {}
        self.splits: dict[str, list[str]] = {}

        flattened: list[tuple[WikiDocument, WikiQuestion]] = [
            (document, question)
            for document in corpus.documents
            for question in document.questions
        ]
        if not flattened:
            raise ValueError("wiki corpus contains no questions")

        exposure_features: dict[str, torch.Tensor] = {}
        for document, _ in flattened:
            if document.id in exposure_features:
                continue
            for marker in FORBIDDEN_EXPOSURE_MARKERS:
                if marker in document.memory:
                    raise ValueError(
                        f"exposure for {document.id!r} contains forbidden marker "
                        f"{marker!r}; fable2 exposures must be passage prose"
                    )
            ids = encode_text(
                tokenizer, document.memory, self.device, max_tokens=max_memory_tokens
            )
            exposure_features[document.id] = backbone.encode(ids)

        staged: list[tuple[WikiDocument, WikiQuestion, dict]] = []
        for document, question in flattened:
            formatted = format_wiki_question(
                question, permutation_seed=permutation_seed
            )
            ids = encode_text(
                tokenizer, formatted.prompt, self.device, max_tokens=max_question_tokens
            )
            staged.append(
                (
                    document,
                    question,
                    {
                        "features": backbone.encode(ids),
                        "ids": ids,
                        "correct_index": formatted.correct_index,
                    },
                )
            )

        for document, question, payload in staged:
            mate_id = self._select_mate(document, payload["correct_index"], staged)
            item = EpisodeItem(
                document_id=document.id,
                question_id=question.id,
                split=document.split,
                source_family=document.source_family,
                exposure_features=exposure_features[document.id],
                question_features=payload["features"],
                question_ids=payload["ids"],
                correct_index=payload["correct_index"],
                mate_question_id=mate_id,
            )
            self.items[question.id] = item
            self.splits.setdefault(document.split, []).append(question.id)

    @staticmethod
    def _select_mate(
        document: WikiDocument, correct_index: int, staged: list
    ) -> str:
        """Pick the deterministic first same-split partner that can disagree.

        A usable mate comes from a different source family (so its passage cannot
        support this question) and carries a different correct label (so paired
        margins compare distinguishable targets).
        """

        for mate_document, mate_question, mate_payload in staged:
            if mate_document.split != document.split:
                continue
            if mate_document.source_family == document.source_family:
                continue
            if mate_payload["correct_index"] == correct_index:
                continue
            return mate_question.id
        raise ValueError(
            f"no usable counterfactual mate in split {document.split!r} for "
            f"document {document.id!r}"
        )

    def split_items(self, split: str) -> list[EpisodeItem]:
        ids = self.splits.get(split, [])
        if not ids:
            raise ValueError(f"wiki split {split!r} is empty; cannot evaluate on it")
        return [self.items[question_id] for question_id in ids]

    def mate(self, item: EpisodeItem) -> EpisodeItem:
        return self.items[item.mate_question_id]


def run_arm(
    system: BrocaSystem,
    bank: EpisodeBank,
    item: EpisodeItem,
    arm: str,
) -> EpisodeScore:
    """Score one question under one preregistered arm from a fresh episode state.

    Arms are distinct computations or they do not exist: `zero_control` and
    `reset_after_exposure` were quoted as independent controls in the C1 series
    while being (respectively) scaffold-identical to the floor and, in fresh
    episodes, computation-identical to `no_exposure`. Here the first is enforced
    as a bitwise invariant by the CPU gate and the second is simply not an arm.
    """

    organism = system.organism
    state = system.initial_state(1, bank.device)
    if arm == "bare_floor":
        return system.score_from_features(
            item.question_features,
            item.question_ids,
            state,
            bank.labels,
            item.correct_index,
            bare=True,
        )

    frozen_idx = None
    lesion_memory = False
    if arm in ("normal", "memory_lesion", "internal_lesion"):
        exposure = item.exposure_features
    elif arm == "wrong_passage":
        exposure = bank.mate(item).exposure_features
    elif arm == "no_exposure":
        exposure = None
    else:
        raise ValueError(f"unknown episode arm {arm!r}")
    if arm == "memory_lesion":
        lesion_memory = True
    if arm == "internal_lesion":
        frozen_idx = organism.internal_idx

    if exposure is not None:
        state = system.observe_feature_sequence(exposure, state)
    if lesion_memory:
        state = lesion_state(state, organism.memory_idx)
    return system.score_from_features(
        item.question_features,
        item.question_ids,
        state,
        bank.labels,
        item.correct_index,
        frozen_idx=frozen_idx,
    )


def run_depth_lesion(
    system: BrocaSystem, bank: EpisodeBank, item: EpisodeItem, depth: int
) -> EpisodeScore:
    """The normal arm with exactly one injection depth silenced."""

    state = system.initial_state(1, bank.device)
    state = system.observe_feature_sequence(item.exposure_features, state)
    return system.score_from_features(
        item.question_features,
        item.question_ids,
        state,
        bank.labels,
        item.correct_index,
        lesion_depths=frozenset({depth}),
    )


def paired_contrast_loss(
    exposed: EpisodeScore,
    wrong: EpisodeScore,
    neutral: EpisodeScore,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward only correct-target gains caused by the matching passage.

    Two terms, two disciplines, one engine.

    The **differential term** carries all persistent learning pressure: the
    correct-passage arm must beat the *live* wrong-passage arm by `margin`. Both
    arms are the same function of the parameters, so content-independent target
    inflation cancels exactly; the only gradient that survives is content
    discrimination.

    The **no-harm term** is a zero-margin hinge against the detached no-exposure
    baseline: it fires only while exposure actively hurts, then goes silent. It is
    deliberately not a margined objective — a margin against a detached baseline
    that is recomputed each step is a ratchet with infinite appetite for static
    control inflation. On the toy curriculum that ratchet grew a common-mode
    control until coefficient saturation erased the arm differential bit-exact
    (the C1x/C1y failure pair, reproduced end to end before this design).

    The no-exposure baseline stays detached per `sol2.wiki_causal_contrast`'s
    rule: a live one could be gamed by suppressing the target whenever exposure is
    absent — exposure-presence discrimination, not content. Whether exposure
    *helps* over no exposure is measured by the evaluation gates, not trained.

    Unlike C1's designated-answer cards, real prose cannot make a wrong passage
    change the right answer, so the pairing is within-question: same question, own
    passage versus mate passage versus none.
    """

    if margin <= 0:
        raise ValueError("paired contrast margin must be positive")
    if not (exposed.correct_label == wrong.correct_label == neutral.correct_label):
        raise ValueError("paired arms must score the same question")
    from sol2.wiki_memory import LABELS

    target = LABELS.index(exposed.correct_label)
    exposed_log_probs = exposed.label_logits.log_softmax(dim=-1)
    wrong_log_probs = wrong.label_logits.log_softmax(dim=-1)
    neutral_log_probs = neutral.label_logits.detach().log_softmax(dim=-1)
    advantages = torch.stack(
        (
            exposed_log_probs[target] - neutral_log_probs[target],
            exposed_log_probs[target] - wrong_log_probs[target],
        )
    )
    no_harm = torch.relu(-advantages[0])
    differential = torch.relu(advantages.new_tensor(margin) - advantages[1])
    loss = no_harm + differential
    return loss, advantages.detach()
