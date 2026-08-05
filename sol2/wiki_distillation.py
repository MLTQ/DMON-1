"""Dense teacher credit and bounded control costs for wiki-memory training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def full_vocab_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return full-vocabulary reverse KL with a detached teacher distribution."""

    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 1:
        raise ValueError("student and teacher logits must have the same non-empty shape")
    if student_logits.shape[-1] < 2:
        raise ValueError("distillation requires at least two vocabulary entries")
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    student = student_logits.float() / temperature
    teacher = teacher_logits.detach().float() / temperature
    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    student_probs = student_log_probs.exp()
    per_item = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    return per_item.mean() * temperature**2


def control_delta_energy(controls: torch.Tensor) -> torch.Tensor:
    """Price the mean squared amplitude of an expressed creature-control delta."""

    if controls.ndim != 3 or controls.shape[1] < 1 or controls.shape[2] < 1:
        raise ValueError("controls must have non-empty [batch, tokens, width] shape")
    return controls.float().pow(2).mean()
