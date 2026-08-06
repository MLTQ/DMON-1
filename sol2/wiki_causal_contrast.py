"""Passage-specific final-logit credit for wiki-memory training."""

from __future__ import annotations

from typing import Protocol

import torch

from .wiki_memory import LABELS


class LogitResult(Protocol):
    correct_label: str
    label_logits: torch.Tensor | None


def passage_causal_contrast_loss(
    left: LogitResult,
    right: LogitResult,
    no_exposure: LogitResult,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward only target-log-probability gains caused by the matching passage.

    The incompatible passage and no-exposure arms are detached baselines. This keeps
    the objective from succeeding by making either control worse; gradients can only
    improve each exposed branch's own passage-bound target.
    """

    if margin <= 0:
        raise ValueError("causal contrast margin must be positive")
    results = (left, right, no_exposure)
    if any(result.label_logits is None for result in results):
        raise ValueError("causal contrast requires live label logits")
    left_target = LABELS.index(left.correct_label)
    right_target = LABELS.index(right.correct_label)
    if left_target == right_target:
        raise ValueError("causal contrast targets must be incompatible")

    left_log_probs = left.label_logits.float().log_softmax(dim=-1)
    right_log_probs = right.label_logits.float().log_softmax(dim=-1)
    neutral_log_probs = no_exposure.label_logits.float().log_softmax(dim=-1)
    advantages = torch.stack(
        (
            left_log_probs[left_target]
            - neutral_log_probs[left_target].detach(),
            right_log_probs[right_target]
            - neutral_log_probs[right_target].detach(),
            left_log_probs[left_target]
            - right_log_probs[left_target].detach(),
            right_log_probs[right_target]
            - left_log_probs[right_target].detach(),
        )
    )
    loss = torch.relu(advantages.new_tensor(margin) - advantages).mean()
    return loss, advantages
