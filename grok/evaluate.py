"""Held-out evaluation with state ablations (SOL discipline)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from .model import CreatureState, StreamingCreature
from .stream import CharacterVocabulary


@dataclass(frozen=True)
class EvalMetrics:
    nll: float
    bits_per_character: float
    accuracy: float
    tokens: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _shuffle_state(state: CreatureState, perm: torch.Tensor) -> CreatureState:
    return CreatureState(
        h=state.h[:, perm],
        eligibility=state.eligibility[:, perm],
        edge_eligibility=state.edge_eligibility[:, perm],
        fast_weight=state.fast_weight[:, perm],
        output_error_credit=state.output_error_credit[:, perm],
        reward=state.reward,
        reward_baseline=state.reward_baseline,
        mirror_cursor=state.mirror_cursor,
        probe_credit=state.probe_credit,
        probe_fitness=state.probe_fitness,
    )


@torch.no_grad()
def evaluate_creature(
    model: StreamingCreature,
    vocabulary: CharacterVocabulary,
    text: str,
    *,
    device: torch.device | str = "cpu",
    tokens: int = 2048,
    warmup: int = 256,
    reset_each_token: bool = False,
    shuffle_cells: bool = False,
) -> EvalMetrics:
    """Score a contiguous stream under an explicit state policy."""

    if tokens < 1:
        raise ValueError("tokens must be positive")
    device = torch.device(device)
    encoded = vocabulary.encode(text, device)
    available = encoded.numel() - 1
    if available < 2:
        raise ValueError("evaluation text too short")
    scored = min(tokens, max(1, available - min(warmup, available - 1)))
    warm = min(warmup, available - scored)
    start = warm
    warm_start = start - warm
    segment = encoded[warm_start : start + scored + 1]

    state = model.initial_state(1, device)
    perm = torch.randperm(
        model.config.n_cells, generator=torch.Generator().manual_seed(1729)
    ).to(device)

    was_training = model.training
    model.eval()
    total = 0.0
    correct = 0
    n_scored = 0

    for i in range(segment.numel() - 1):
        if reset_each_token:
            state = model.initial_state(1, device)
        elif shuffle_cells and i >= warm:
            state = _shuffle_state(state, perm)

        x = segment[i : i + 1]
        y = segment[i + 1 : i + 2]
        logits, state = model.step(x, state)
        state = model.observe_prediction(state, logits, y)

        if i >= warm:
            loss = F.cross_entropy(logits, y)
            total += float(loss.item())
            correct += int(logits.argmax(dim=-1).item() == int(y.item()))
            n_scored += 1

    if was_training:
        model.train()
    nll = total / max(n_scored, 1)
    return EvalMetrics(
        nll=nll,
        bits_per_character=nll / math.log(2.0),
        accuracy=correct / max(n_scored, 1),
        tokens=n_scored,
    )


@torch.no_grad()
def evaluate_with_ablations(
    model: StreamingCreature,
    vocabulary: CharacterVocabulary,
    text: str,
    *,
    device: torch.device | str = "cpu",
    tokens: int = 2048,
    warmup: int = 256,
) -> dict[str, dict[str, float | int]]:
    """Normal + reset + shuffle — SOL-style substrate checks."""

    base = evaluate_creature(
        model, vocabulary, text, device=device, tokens=tokens, warmup=warmup
    )
    reset = evaluate_creature(
        model,
        vocabulary,
        text,
        device=device,
        tokens=tokens,
        warmup=warmup,
        reset_each_token=True,
    )
    shuffle = evaluate_creature(
        model,
        vocabulary,
        text,
        device=device,
        tokens=tokens,
        warmup=warmup,
        shuffle_cells=True,
    )
    return {
        "normal": base.to_dict(),
        "reset_each_token": reset.to_dict(),
        "shuffle_cells": shuffle.to_dict(),
        "reset_delta_bpc": reset.bits_per_character - base.bits_per_character,
        "shuffle_delta_bpc": shuffle.bits_per_character - base.bits_per_character,
    }
