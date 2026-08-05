"""Training-only differential credit for relay-to-output transport."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from torch.nn import functional as F

from .wiki_memory import LABELS


class TransportStateResult(Protocol):
    correct_label: str
    relay_state: torch.Tensor | None
    output_state: torch.Tensor | None


def eligibility_gated_transport_loss(
    branch_results: Sequence[TransportStateResult],
    *,
    codebook: torch.Tensor,
    relay_reference: float,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align paired output difference with its answer axis when relay differs."""

    if len(branch_results) != 2:
        raise ValueError("output transport credit requires exactly two branches")
    if min(relay_reference, margin, temperature) <= 0:
        raise ValueError("transport scales must be positive")
    left, right = branch_results
    if left.output_state is None or right.output_state is None:
        raise ValueError("output transport credit requires live output states")
    if left.relay_state is None or right.relay_state is None:
        raise ValueError("output transport credit requires live relay states")
    if left.output_state.shape != right.output_state.shape:
        raise ValueError("paired output states must have identical shape")
    if left.relay_state.shape != right.relay_state.shape:
        raise ValueError("paired relay states must have identical shape")

    left_target = LABELS.index(left.correct_label)
    right_target = LABELS.index(right.correct_label)
    if left_target == right_target:
        raise ValueError("paired transport targets must be incompatible")
    if codebook.shape[0] != len(LABELS):
        raise ValueError("transport codebook must contain one code per label")

    output_delta = (
        left.output_state.float().mean(dim=1)[0]
        - right.output_state.float().mean(dim=1)[0]
    )
    if output_delta.shape[0] != codebook.shape[1]:
        raise ValueError("transport code width does not match output tissue")
    target_axis = F.normalize(
        codebook[left_target].float() - codebook[right_target].float(), dim=0
    )
    projection = torch.dot(output_delta, target_axis)
    relay_separation = (
        left.relay_state.float()
        .sub(right.relay_state.float())
        .pow(2)
        .mean()
        .sqrt()
    )
    output_separation = (
        left.output_state.float()
        .sub(right.output_state.float())
        .pow(2)
        .mean()
        .sqrt()
    )
    eligibility = (relay_separation.detach() / relay_reference).clamp(0.0, 1.0)
    loss = eligibility * temperature * F.softplus(
        (projection.new_tensor(margin) - projection) / temperature
    )
    transport_ratio = output_separation / relay_separation.detach().clamp_min(1e-8)
    return (
        loss,
        projection,
        relay_separation,
        output_separation,
        transport_ratio,
        eligibility,
    )
