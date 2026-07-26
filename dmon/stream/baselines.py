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


class TransformerBaseline(nn.Module):
    """Small causal transformer over a fixed context window.

    Comparability note: a vanilla transformer has no persistent state, so it is given
    exactly the history the lattice holds in its mirror cells — `context` tokens. The
    lattice additionally carries persistent state the transformer has no equivalent of,
    which is the point of the architecture and not a handicap to correct for. This is
    the bar, and it is expected to be a hard one at small scale.
    """

    def __init__(self, vocab: int, d_model: int, context: int, n_layer: int = 2, n_head: int = 4):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model, batch_first=True,
            dropout=0.0, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab)
        mask = torch.triu(torch.ones(context, context, dtype=torch.bool), 1)
        self.register_buffer("mask", mask)

    def blank_state(self, batch: int, device=None):
        return None  # stateless by construction; context is supplied per tick

    def tick(self, state, token: torch.Tensor, history: torch.Tensor):
        # history is most-recent-first; reverse to chronological and append nothing —
        # `token` is already history[:, 0] because the caller pushes before ticking.
        ctx = history[:, : self.context].flip(1)
        h = self.embed(ctx) + self.pos[:, : ctx.shape[1]]
        h = self.norm(self.blocks(h, mask=self.mask[: ctx.shape[1], : ctx.shape[1]]))
        return state, self.readout(h[:, -1])

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _match(build, target_params: int, lo: int = 8, hi: int = 1024, step: int = 4):
    """Largest build(size) whose parameter count does not exceed the budget.

    Search rather than algebra: the closed form is easy to get subtly wrong, and an
    under-budgeted baseline makes a favourable comparison meaningless.
    """
    best = None
    for size in range(lo, hi + 1, step):
        try:
            n = build(size).param_count()
        except Exception:
            break
        if n <= target_params:
            best = size
        else:
            break
    return build(best or lo)


def transformer_matched_to(
    target_params: int, vocab: int, context: int, n_head: int = 4
) -> TransformerBaseline:
    return _match(
        lambda d: TransformerBaseline(vocab, d - (d % n_head) or n_head, context, n_head=n_head),
        target_params,
        lo=n_head,
        step=n_head,
    )


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
