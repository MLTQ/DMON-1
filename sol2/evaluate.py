"""Prequential held-out evaluation and tissue-specific causal interventions."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .interventions import degree_preserving_rewire, shuffled_private_expression
from .model import Sol2
from .state import OrganismState

LN2 = math.log(2.0)


@torch.no_grad()
def prequential(
    model,
    ids: torch.Tensor,
    warmup: int,
    scored: int,
    device: str,
    *,
    mutate=None,
    reset: bool = False,
    frozen_idx: torch.Tensor | None = None,
) -> dict:
    creature = isinstance(model, Sol2)
    state = model.initial_state(1, device)
    total, correct, count = 0.0, 0, 0
    length = min(len(ids) - 1, warmup + scored)
    for position in range(length):
        if reset:
            state = model.initial_state(1, device)
        if mutate is not None:
            state = mutate(state)
        x = ids[position].reshape(1).to(device)
        target = ids[position + 1].reshape(1).to(device)
        if creature:
            logits, state, _ = model.step(x, state, frozen_idx=frozen_idx)
        else:
            logits, state = model.step(x, state)
        if position >= warmup:
            total += float(F.cross_entropy(logits, target))
            correct += int(logits.argmax(-1).eq(target).item())
            count += 1
    nll = total / max(count, 1)
    return {
        "nll": nll,
        "bits_per_character": nll / LN2,
        "accuracy": correct / max(count, 1),
        "tokens": count,
    }


def evaluate_model(model, ids, warmup, scored, device) -> dict:
    return prequential(model, ids, warmup, scored, device)


def evaluate_with_ablations(
    model: Sol2, ids: torch.Tensor, warmup: int, scored: int, device: str
) -> dict:
    normal = prequential(model, ids, warmup, scored, device)
    reset = prequential(model, ids, warmup, scored, device, reset=True)
    internal = model.internal_idx
    freeze_internal = prequential(
        model, ids, warmup, scored, device, frozen_idx=internal
    )

    generator = torch.Generator().manual_seed(1729)
    tissue_permutations = {}
    for name in ("compute", "relay"):
        cells = model.tissue_indices(name)
        if len(cells) > 1:
            permutation = torch.randperm(len(cells), generator=generator)
            tissue_permutations[name] = cells.detach().cpu()[permutation].to(device)

    def shuffle_within_tissue(state: OrganismState) -> OrganismState:
        hidden = state.hidden
        for name, permutation in tissue_permutations.items():
            cells = model.tissue_indices(name)
            hidden = hidden.index_copy(1, cells, hidden[:, permutation])
        return OrganismState(hidden, state.memory_cursor, state.weight_version)

    shuffled = prequential(
        model, ids, warmup, scored, device, mutate=shuffle_within_tissue
    )

    def zero_memory(state: OrganismState) -> OrganismState:
        zeros = state.hidden.new_zeros(
            state.hidden.shape[0], len(model.memory_idx), state.hidden.shape[2]
        )
        return OrganismState(
            state.hidden.index_copy(1, model.memory_idx, zeros),
            state.memory_cursor,
            state.weight_version,
        )

    memory_zero = prequential(model, ids, warmup, scored, device, mutate=zero_memory)
    with shuffled_private_expression(model):
        identity_shuffled = prequential(model, ids, warmup, scored, device)
    with degree_preserving_rewire(model):
        topology_rewired = prequential(model, ids, warmup, scored, device)
    normal_bpc = normal["bits_per_character"]
    result = {
        "normal": normal,
        "reset_each_token": reset,
        "freeze_internal": freeze_internal,
        "shuffle_within_tissue": shuffled,
        "memory_zero": memory_zero,
        "identity_shuffled": identity_shuffled,
        "topology_rewired": topology_rewired,
        "reset_delta_bpc": reset["bits_per_character"] - normal_bpc,
        "freeze_internal_delta_bpc": freeze_internal["bits_per_character"] - normal_bpc,
        "shuffle_within_tissue_delta_bpc": shuffled["bits_per_character"] - normal_bpc,
        "memory_delta_bpc": memory_zero["bits_per_character"] - normal_bpc,
        "identity_shuffle_delta_bpc": (
            identity_shuffled["bits_per_character"] - normal_bpc
        ),
        "topology_rewire_delta_bpc": (
            topology_rewired["bits_per_character"] - normal_bpc
        ),
    }
    for name in ("compute", "relay"):
        frozen = prequential(
            model,
            ids,
            warmup,
            scored,
            device,
            frozen_idx=model.tissue_indices(name),
        )
        result[f"freeze_{name}"] = frozen
        result[f"freeze_{name}_delta_bpc"] = frozen["bits_per_character"] - normal_bpc
    return result
