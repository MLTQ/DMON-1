"""Training-only differential semantic credit for internal wiki-memory tissues."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def fixed_semantic_projection(
    source_width: int,
    target_width: int,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a deterministic parameter-free Gaussian semantic projection."""

    if source_width < 1 or target_width < 1:
        raise ValueError("semantic projection widths must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(source_width)
    return projection.to(device=device).detach()


def semantic_target_delta(
    left_teacher: torch.Tensor,
    right_teacher: torch.Tensor,
    projection: torch.Tensor,
    *,
    target_rms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project and RMS-normalize one detached paired teacher difference."""

    if target_rms <= 0:
        raise ValueError("semantic target RMS must be positive")
    if left_teacher.shape != right_teacher.shape or left_teacher.ndim != 2:
        raise ValueError("teacher features must share [batch, width] shape")
    if (
        projection.ndim != 2
        or projection.shape[0] != left_teacher.shape[-1]
        or projection.shape[1] < 1
    ):
        raise ValueError("projection must map teacher width to a non-empty target")
    raw_target = (
        left_teacher.detach().float() - right_teacher.detach().float()
    ) @ projection.detach().float()
    raw_rms = raw_target.pow(2).mean().sqrt()
    if not bool(torch.isfinite(raw_rms)) or float(raw_rms) <= 0.0:
        raise ValueError("paired teacher semantic target must be finite and nonzero")
    target = raw_target * (target_rms / raw_rms)
    return target.detach(), raw_rms.detach()


def paired_tissue_semantic_credit(
    left_state: torch.Tensor,
    right_state: torch.Tensor,
    left_teacher: torch.Tensor,
    right_teacher: torch.Tensor,
    projection: torch.Tensor,
    *,
    target_rms: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match a paired tissue-state difference to detached teacher semantics."""

    if left_state.shape != right_state.shape or left_state.ndim != 3:
        raise ValueError("tissue states must share [batch, cells, hidden] shape")
    if left_state.shape[1] < 1 or left_state.shape[2] != projection.shape[1]:
        raise ValueError("tissue state must contain cells at projection target width")
    if left_teacher.shape[0] != left_state.shape[0]:
        raise ValueError("teacher and tissue batches must match")
    target, raw_target_rms = semantic_target_delta(
        left_teacher,
        right_teacher,
        projection,
        target_rms=target_rms,
    )
    student_delta = left_state.float().mean(dim=1) - right_state.float().mean(dim=1)
    loss = F.mse_loss(student_delta, target)
    student_rms = student_delta.pow(2).mean().sqrt()
    alignment = F.cosine_similarity(student_delta, target, dim=-1, eps=1e-8).mean()
    return loss, student_rms, alignment, raw_target_rms


def paired_vector_semantic_credit(
    left_state: torch.Tensor,
    right_state: torch.Tensor,
    left_teacher: torch.Tensor,
    right_teacher: torch.Tensor,
    projection: torch.Tensor,
    *,
    target_rms: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match a paired live-vector difference to detached teacher semantics."""

    if left_state.shape != right_state.shape or left_state.ndim != 2:
        raise ValueError("semantic vectors must share [batch, hidden] shape")
    if left_state.shape[1] != projection.shape[1]:
        raise ValueError("semantic vector width must match projection target width")
    if left_teacher.shape[0] != left_state.shape[0]:
        raise ValueError("teacher and semantic-vector batches must match")
    target, raw_target_rms = semantic_target_delta(
        left_teacher,
        right_teacher,
        projection,
        target_rms=target_rms,
    )
    student_delta = left_state.float() - right_state.float()
    loss = F.mse_loss(student_delta, target)
    student_rms = student_delta.pow(2).mean().sqrt()
    alignment = F.cosine_similarity(student_delta, target, dim=-1, eps=1e-8).mean()
    return loss, student_rms, alignment, raw_target_rms


def recall_addressing_credit(
    content_scores: torch.Tensor,
    attention: torch.Tensor,
    exposure_features: torch.Tensor,
    teacher_effect: torch.Tensor,
    *,
    target_temperature: float,
    newest_slots: int = 4,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Train content scores toward detached teacher-relevant exposure positions."""

    if target_temperature <= 0 or newest_slots < 1:
        raise ValueError("addressing temperature and newest-slot count must be positive")
    if content_scores.ndim != 2 or attention.shape != content_scores.shape:
        raise ValueError("recall scores and attention must share [batch, memory] shape")
    if exposure_features.ndim != 3 or teacher_effect.ndim != 2:
        raise ValueError("exposure and teacher tensors must be [batch, tokens, width]")
    if (
        exposure_features.shape[0] != content_scores.shape[0]
        or teacher_effect.shape[0] != content_scores.shape[0]
        or exposure_features.shape[2] != teacher_effect.shape[1]
    ):
        raise ValueError("addressing teacher, exposure, and recall batches must align")
    count = content_scores.shape[1]
    token_count = min(exposure_features.shape[1], count)
    if token_count < 1:
        raise ValueError("addressing credit requires at least one stored exposure token")

    recent_features = exposure_features[:, -token_count:].detach().float()
    detached_effect = teacher_effect.detach().float().unsqueeze(1)
    target_logits = F.cosine_similarity(
        recent_features,
        detached_effect.expand_as(recent_features),
        dim=-1,
        eps=1e-8,
    )
    recent_target = torch.softmax(target_logits / target_temperature, dim=-1)
    target = torch.zeros_like(content_scores, dtype=torch.float32)
    target[:, -token_count:] = recent_target
    target = target.detach()

    student_log_probs = torch.log_softmax(content_scores.float(), dim=-1)
    target_log_probs = target.clamp_min(1e-12).log()
    loss = (target * (target_log_probs - student_log_probs)).sum(dim=-1).mean()
    selected_target_mass = (
        target * (attention.detach().float() > 0).to(target.dtype)
    ).sum(dim=-1).mean()
    live_attention = attention.float()
    entropy = -(
        live_attention * live_attention.clamp_min(1e-12).log()
    ).sum(dim=-1).mean()
    effective_slots = entropy.exp()
    newest_mass = live_attention[:, -min(newest_slots, count) :].sum(dim=-1).mean()
    return loss, selected_target_mass, entropy, effective_slots, newest_mass
