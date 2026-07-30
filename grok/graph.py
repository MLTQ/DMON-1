"""Directed dendrite connectome with local attention and fast efficacy."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DendriteGraph(nn.Module):
    """Per-cell source slots — axons to specific partners, not a 3×3 neighborhood.

    Aggregation is local attention over real dendrites only. Signed slow weights plus
    optional stream-local fast efficacy (SOL-style) modulate branch coefficients.
    Coefficients are returned for reverse credit transport through the same graph.
    """

    def __init__(
        self,
        n_cells: int,
        n_dendrites: int,
        hidden: int,
        *,
        input_idx: torch.Tensor,
        output_idx: torch.Tensor,
        mirror_idx: torch.Tensor,
        internal_idx: torch.Tensor,
        seed: int = 0,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if n_dendrites < 1:
            raise ValueError("n_dendrites must be >= 1")
        self.n_cells = n_cells
        self.n_dendrites = n_dendrites
        self.hidden = hidden
        self.use_attention = use_attention

        sources = self._wire(
            n_cells=n_cells,
            n_dendrites=n_dendrites,
            input_idx=input_idx,
            output_idx=output_idx,
            mirror_idx=mirror_idx,
            internal_idx=internal_idx,
            seed=seed,
        )
        self.register_buffer("sources", sources, persistent=True)
        self.weights = nn.Parameter(
            torch.empty(n_cells, n_dendrites).uniform_(-0.05, 0.05)
        )
        self.bias = nn.Parameter(torch.zeros(n_cells, n_dendrites))

        self.query = nn.Linear(hidden, hidden, bias=False) if use_attention else None
        self.key = nn.Linear(hidden, hidden, bias=False) if use_attention else None
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.scale = hidden**-0.5
        nn.init.orthogonal_(self.value.weight)

    @staticmethod
    def _wire(
        *,
        n_cells: int,
        n_dendrites: int,
        input_idx: torch.Tensor,
        output_idx: torch.Tensor,
        mirror_idx: torch.Tensor,
        internal_idx: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        sources = torch.zeros(n_cells, n_dendrites, dtype=torch.long)
        inputs = input_idx.tolist()
        outputs = output_idx.tolist()
        mirrors = mirror_idx.tolist()
        internals = internal_idx.tolist()
        all_src = list(range(n_cells))

        for cell in range(n_cells):
            if cell in outputs:
                pool = internals + inputs + mirrors
            elif cell in mirrors:
                pool = mirrors + inputs
            elif cell in inputs:
                pool = inputs + internals
            else:
                pool = inputs + mirrors + internals
            pool = [p for p in pool if p != cell] or all_src
            if len(pool) >= n_dendrites:
                perm = torch.randperm(len(pool), generator=g)[:n_dendrites]
                choices = [pool[int(i)] for i in perm.tolist()]
            else:
                choices = [
                    pool[int(torch.randint(0, len(pool), (1,), generator=g))]
                    for _ in range(n_dendrites)
                ]
            # Organ plumbing: each output keeps one sensory axon (strength trainable).
            if cell in outputs and inputs:
                choices[0] = inputs[cell % len(inputs)]
            sources[cell] = torch.tensor(choices, dtype=torch.long)
        return sources

    def aggregate(
        self,
        h: torch.Tensor,
        fast_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (messages [B,N,H], coefficients [B,N,K])."""

        gathered = h[:, self.sources, :]  # [B, N, K, H]
        slow = self.weights.unsqueeze(0) + self.bias.unsqueeze(0)
        if fast_weight is not None:
            slow = slow + fast_weight
        v = self.value(gathered)

        if not self.use_attention:
            coeff = torch.softmax(slow.expand(h.shape[0], -1, -1), dim=-1)
            messages = (v * coeff.unsqueeze(-1)).sum(dim=2)
            return messages, coeff

        assert self.query is not None and self.key is not None
        q = self.query(h).unsqueeze(2)
        k = self.key(gathered)
        scores = (q * k).sum(dim=-1) * self.scale + slow
        coeff = F.softmax(scores, dim=-1)
        messages = (coeff.unsqueeze(-1) * v).sum(dim=2)
        return messages, coeff

    def transport_vector_credit(
        self,
        credit: torch.Tensor,
        coefficient: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter channel credit sourceward through the transpose of message_value."""

        b, _, h = credit.shape
        sourceward = F.linear(credit, self.value.weight.t())
        edge = coefficient.unsqueeze(-1) * sourceward.unsqueeze(2)
        contribution = edge.flatten(1, 2)
        src = self.sources.flatten().view(1, -1, 1).expand(b, -1, h)
        out = torch.zeros_like(credit)
        out.scatter_add_(1, src, contribution)
        return out / math.sqrt(self.n_dendrites)
