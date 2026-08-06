"""Training-only answer-span value and sparse internal transport credit."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _last_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None


def designated_answer_token_mask(
    tokenizer: Any,
    memory_text: str,
    answer: str,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Locate the exact designated-answer token span in an encoded memory card."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("answer-span alignment requires one encoded sequence")
    marker = "Designated answer: "
    line = marker + answer
    line_start = memory_text.rfind(line)
    if line_start < 0:
        raise ValueError("memory card does not contain its designated answer")
    answer_start = line_start + len(marker)
    answer_end = answer_start + len(answer)

    try:
        encoded = tokenizer(
            memory_text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.offset_mapping[0]
        if not torch.equal(encoded.input_ids.cpu(), input_ids.detach().cpu()):
            raise ValueError("offset tokenization differs from exposure tokenization")
        mask = torch.tensor(
            [
                int(end) > answer_start and int(start) < answer_end
                for start, end in offsets.tolist()
            ],
            dtype=torch.bool,
            device=input_ids.device,
        )
        if bool(mask.any()):
            return mask
    except TypeError:
        pass
    except (AttributeError, NotImplementedError):
        pass

    full_ids = input_ids.detach().cpu()[0].tolist()
    candidates: list[list[int]] = []
    for text in (" " + answer, answer):
        candidate = tokenizer.encode(text, add_special_tokens=False)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    matches = [
        (start, candidate)
        for candidate in candidates
        if (start := _last_subsequence(full_ids, candidate)) is not None
    ]
    if not matches:
        raise ValueError("designated answer tokens were not found in exposure IDs")
    start, matched = max(matches, key=lambda item: (len(item[1]), item[0]))
    mask = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    mask[start : start + len(matched)] = True
    return mask


def answer_memory_slots(
    answer_mask: torch.Tensor,
    *,
    start_cursor: int,
    memory_capacity: int,
) -> torch.Tensor:
    """Map written answer-token positions onto surviving circular memory slots."""

    if answer_mask.ndim != 1 or answer_mask.numel() < 1:
        raise ValueError("answer mask must be a non-empty vector")
    if start_cursor < 0 or memory_capacity < 1:
        raise ValueError("memory cursor and capacity must be valid")
    positions = answer_mask.nonzero(as_tuple=False).flatten()
    if positions.numel() < 1:
        raise ValueError("answer mask selects no tokens")
    sequence_length = answer_mask.numel()
    oldest_surviving = max(0, sequence_length - memory_capacity)
    if bool((positions < oldest_surviving).any()):
        raise ValueError("designated answer was overwritten before recall")
    return (positions + start_cursor).remainder(memory_capacity)


def coherent_answer_value_target(
    exposure_memory_state: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    start_cursor: int,
    recall_gain: float,
) -> torch.Tensor:
    """Return a detached ideal coherent recall of the stored answer neurons."""

    if exposure_memory_state.ndim != 3 or exposure_memory_state.shape[0] != 1:
        raise ValueError("exposure memory must have [1, cells, hidden] shape")
    if recall_gain <= 0:
        raise ValueError("recall gain must be positive")
    slots = answer_memory_slots(
        answer_mask,
        start_cursor=start_cursor,
        memory_capacity=exposure_memory_state.shape[1],
    ).to(exposure_memory_state.device)
    stored = exposure_memory_state.index_select(1, slots).mean(dim=1)
    return (recall_gain * torch.tanh(stored.float())).detach()


def coherent_recall_value_credit(
    recalled: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align live recall with a detached answer value in the same hidden basis."""

    if recalled.shape != target.shape or recalled.ndim != 2:
        raise ValueError("recall and coherent target must share [batch, hidden] shape")
    detached_target = target.detach().float()
    student = recalled.float()
    loss = F.mse_loss(student, detached_target)
    alignment = F.cosine_similarity(student, detached_target, dim=-1).mean()
    student_rms = student.pow(2).mean().sqrt()
    target_rms = detached_target.pow(2).mean().sqrt()
    return loss, alignment, student_rms, target_rms


def _sparse_tissue_pool(
    tissue: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tissue.ndim != 3 or target.ndim != 2:
        raise ValueError("tissue and target need [batch, cells, hidden] and [batch, hidden]")
    if tissue.shape[0] != target.shape[0] or tissue.shape[2] != target.shape[1]:
        raise ValueError("tissue and target dimensions do not match")
    if tissue.shape[1] < 1 or temperature <= 0:
        raise ValueError("sparse pooling needs cells and a positive temperature")
    live = tissue.float()
    detached_target = target.detach().float()
    similarities = F.cosine_similarity(
        live,
        detached_target.unsqueeze(1),
        dim=-1,
        eps=1e-8,
    )
    weights = torch.softmax(similarities / temperature, dim=-1)
    pooled = torch.einsum("bc,bch->bh", weights, live)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    effective_cells = entropy.exp()
    return pooled, entropy, effective_cells


def sparse_coherent_transport_credit(
    relay_state: torch.Tensor,
    output_state: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Reward sparse relay and output pools for carrying the coherent answer value."""

    detached_target = target.detach().float()
    relay_pool, _, relay_effective = _sparse_tissue_pool(
        relay_state, detached_target, temperature=temperature
    )
    output_pool, _, output_effective = _sparse_tissue_pool(
        output_state, detached_target, temperature=temperature
    )
    relay_loss = F.mse_loss(relay_pool, detached_target)
    output_loss = F.mse_loss(output_pool, detached_target)
    loss = 0.5 * (relay_loss + output_loss)
    relay_alignment = F.cosine_similarity(
        relay_pool, detached_target, dim=-1
    ).mean()
    output_alignment = F.cosine_similarity(
        output_pool, detached_target, dim=-1
    ).mean()
    relay_rms = relay_pool.pow(2).mean().sqrt()
    output_rms = output_pool.pow(2).mean().sqrt()
    return (
        loss,
        relay_alignment,
        output_alignment,
        relay_effective,
        output_effective,
        0.5 * (relay_rms + output_rms),
    )
