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
