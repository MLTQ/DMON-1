"""Held-out prequential evaluation with causal ablations.

All passes run at B=1 over the same contiguous holdout window: `warmup`
unscored tokens to fill state, then `scored` tokens of accumulated CE.
No label feedback anywhere (grok's eval fed the target back into persistent
credit state; fable has no such pathway, so creature and GRU see identical
information at eval).

Ablations (creature only):
  reset_each_token — fresh state before every token. Destroys distributed
      state AND the mirror ring; an upper bound on state causality.
  shuffle_internal — fixed permutation of the internal block re-applied every
      token. grok permuted all cells including ports and the readout
      correspondence, which an untrained model also fails (debt #26); this
      variant isolates internal tissue.
  mirror_zero — zero the mirror ring every token, keep everything else. What
      remains of the reset delta after this is the verbatim n-gram window's
      share (debt #23).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import Fable, OrganismState

LN2 = math.log(2.0)


@torch.no_grad()
def _prequential(model, ids: torch.Tensor, warmup: int, scored: int,
                 device: str, mutate=None, reset: bool = False,
                 frozen_idx: torch.Tensor | None = None) -> dict:
    is_creature = isinstance(model, Fable)
    state = model.initial_state(1, device)
    total_nll, correct, counted = 0.0, 0, 0
    n = min(len(ids) - 1, warmup + scored)
    for i in range(n):
        x = ids[i].reshape(1).to(device)
        y = ids[i + 1].reshape(1).to(device)
        if reset:
            state = model.initial_state(1, device)
        if mutate is not None:
            state = mutate(state)
        if is_creature:
            logits, state, _ = model.step(x, state, frozen_idx=frozen_idx)
        else:
            logits, state = model.step(x, state)
        if i >= warmup:
            total_nll += float(F.cross_entropy(logits, y))
            correct += int(logits.argmax(-1).eq(y).item())
            counted += 1
    nll = total_nll / max(counted, 1)
    return {"nll": nll, "bits_per_character": nll / LN2,
            "accuracy": correct / max(counted, 1), "tokens": counted}


def evaluate_model(model, holdout_ids: torch.Tensor, warmup: int, scored: int,
                   device: str) -> dict:
    return _prequential(model, holdout_ids, warmup, scored, device)


def evaluate_with_ablations(model: Fable, holdout_ids: torch.Tensor,
                            warmup: int, scored: int, device: str,
                            frozen_idx: torch.Tensor | None = None) -> dict:
    normal = _prequential(model, holdout_ids, warmup, scored, device,
                          frozen_idx=frozen_idx)
    reset = _prequential(model, holdout_ids, warmup, scored, device,
                         reset=True)

    internal = model.internal_idx
    perm = internal[torch.randperm(
        len(internal), generator=torch.Generator().manual_seed(1729))]

    def shuffle(state: OrganismState) -> OrganismState:
        h = state.h.index_copy(1, internal, state.h[:, perm])
        return OrganismState(h, state.mirror_cursor)

    shuffled = _prequential(model, holdout_ids, warmup, scored, device,
                            mutate=shuffle)

    mirror = model.mirror_idx

    def zero_mirror(state: OrganismState) -> OrganismState:
        zeros = torch.zeros(state.h.shape[0], len(mirror), state.h.shape[2],
                            device=state.h.device)
        return OrganismState(state.h.index_copy(1, mirror, zeros),
                             state.mirror_cursor)

    mirror_zero = _prequential(model, holdout_ids, warmup, scored, device,
                               mutate=zero_mirror)

    bpc = normal["bits_per_character"]
    return {
        "normal": normal,
        "reset_each_token": reset,
        "shuffle_internal": shuffled,
        "mirror_zero": mirror_zero,
        "reset_delta_bpc": reset["bits_per_character"] - bpc,
        "shuffle_delta_bpc": shuffled["bits_per_character"] - bpc,
        "mirror_delta_bpc": mirror_zero["bits_per_character"] - bpc,
    }
