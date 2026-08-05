"""Training-only target credit for SOL2 output tissue."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from torch.nn import functional as F

from .wiki_memory import LABELS


class OutputStateResult(Protocol):
    correct_label: str
    output_state: torch.Tensor | None


def fixed_output_codebook(
    hidden: int,
    *,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build four deterministic orthonormal target codes without parameters."""

    if hidden < 4 or hidden % 4:
        raise ValueError("output-credit hidden width must be divisible by four")
    index = torch.arange(hidden, dtype=torch.int64)
    base = torch.stack(
        (
            torch.ones(hidden),
            torch.where(index.remainder(2) == 0, 1.0, -1.0),
            torch.where(index.div(2, rounding_mode="floor").remainder(2) == 0, 1.0, -1.0),
            torch.where(index.remainder(2) == index.div(2, rounding_mode="floor").remainder(2), 1.0, -1.0),
        )
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(hidden, generator=generator)
    signs = torch.randint(0, 2, (hidden,), generator=generator).float().mul_(2).sub_(1)
    codes = base[:, permutation] * signs
    return (codes / hidden**0.5).to(device=device, dtype=dtype)


def output_tissue_credit_loss(
    branch_results: Sequence[OutputStateResult],
    *,
    codebook: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify paired passage targets from final output tissue only."""

    if len(branch_results) < 2:
        raise ValueError("output-tissue credit requires matched passage branches")
    states = []
    targets = []
    for result in branch_results:
        if result.output_state is None:
            raise ValueError("output-tissue credit requires live output state")
        if result.output_state.ndim != 3 or result.output_state.shape[0] != 1:
            raise ValueError("output state must have [1, cells, hidden] shape")
        states.append(result.output_state.mean(dim=1)[0].float())
        targets.append(LABELS.index(result.correct_label))
    pooled = torch.stack(states)
    if pooled.shape[1] != codebook.shape[1] or codebook.shape[0] != len(LABELS):
        raise ValueError("output codebook shape does not match output tissue")
    logits = scale * F.normalize(pooled, dim=-1) @ codebook.float().T
    target = torch.tensor(targets, device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    accuracy = (logits.argmax(dim=-1) == target).float().mean()
    separation = (
        branch_results[0]
        .output_state.float()
        .sub(branch_results[1].output_state.float())
        .pow(2)
        .mean()
        .sqrt()
    )
    return loss, accuracy, separation
