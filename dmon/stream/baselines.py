"""Null models for S0. Mandatory, not optional.

Under the previous architecture a trained rule scored *worse than having no network at
all*, and that only surfaced because a control was run. "The NCA substrate is doing
something" is a measurement against the best available alternative on the identical
stream, with a matched parameter budget and an identical truncation schedule — anything
else compares an architecture to a handicap.

Two baselines, answering different questions:

  `InertLattice`  the same lattice with its rule silenced. Isolates how much of the
                  score comes from the readout and the embedding rather than from any
                  recurrent computation. This is the floor.
  `GRUBaseline`   a conventional recurrent net at the same parameter count. This is the
                  bar. The lattice does not have to beat it at S0 — it has to be
                  competitive, because the argument for the substrate is runtime
                  growth (S2), not per-parameter efficiency.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    """Persistent single-layer GRU over the same character stream."""

    def __init__(self, vocab: int, hidden: int, embed: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.cell = nn.GRUCell(embed, hidden)
        self.readout = nn.Linear(hidden, vocab)
        self.hidden = hidden

    def blank_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    def tick(self, h: torch.Tensor, token: torch.Tensor, history=None):
        h = self.cell(self.embed(token), h)
        return h, self.readout(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def gru_matched_to(target_params: int, vocab: int, embed: int = 32) -> GRUBaseline:
    """Largest GRU whose parameter count does not exceed `target_params`.

    Solved by search rather than algebra because the readout and embedding both scale
    with hidden size and getting the closed form subtly wrong would quietly hand the
    baseline a smaller budget — which is exactly the sort of thing that makes a
    favourable comparison meaningless.
    """
    lo, hi, best = 8, 4096, None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = GRUBaseline(vocab, mid, embed).param_count()
        if n <= target_params:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return GRUBaseline(vocab, best or 8, embed)
