"""Directed dendrite connectome — not a convolutional neighborhood."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DendriteGraph(nn.Module):
    """Each cell owns `n_dendrites` source slots with learnable signed weights.

    Topology is discrete and frozen during a differentiable stream (petridish
    trial discipline). Sources are specific cell indices — axons to chosen
    partners, not "everyone in a 3×3 window."

    Aggregation is *local attention over real dendrites only*: each cell forms a
    query from its state, attends over its K sources' keys/values, and scales by
    signed synaptic weights. No all-pairs field attention.
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
        # Discrete endpoints — no gradient into indices.
        self.register_buffer("sources", sources, persistent=True)
        # Signed, zero-centered init (petridish lesson against saturation).
        weights = torch.empty(n_cells, n_dendrites).uniform_(-0.05, 0.05)
        self.weights = nn.Parameter(weights)

        if use_attention:
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
            self.value = nn.Linear(hidden, hidden, bias=False)
            self.scale = hidden**-0.5
        else:
            self.query = None
            self.key = None
            self.value = None
            self.scale = 1.0

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
        all_src_pool = list(range(n_cells))

        for cell in range(n_cells):
            if cell in outputs:
                pool = internals + inputs + mirrors
            elif cell in mirrors:
                # Mirrors are stream-written; rule ignores inbound messages.
                pool = mirrors + inputs
            elif cell in inputs:
                pool = inputs + internals
            else:
                pool = inputs + mirrors + internals

            pool = [p for p in pool if p != cell] or all_src_pool
            choices = []
            for _ in range(n_dendrites):
                j = int(torch.randint(0, len(pool), (1,), generator=g).item())
                choices.append(pool[j])
            # Prefer unique sources when the pool is large enough.
            if len(pool) >= n_dendrites:
                perm = torch.randperm(len(pool), generator=g)[:n_dendrites]
                choices = [pool[int(i)] for i in perm.tolist()]
            sources[cell] = torch.tensor(choices, dtype=torch.long)
        return sources

    def aggregate(self, h: torch.Tensor) -> torch.Tensor:
        """Messages for every cell from its real dendrites only.

        h: [B, N, H] → messages [B, N, H]
        """

        src = self.sources  # [N, K]
        gathered = h[:, src, :]  # [B, N, K, H]
        syn = self.weights.unsqueeze(0).unsqueeze(-1)  # [1, N, K, 1]

        if not self.use_attention:
            return (gathered * syn).sum(dim=2)

        assert self.query is not None and self.key is not None and self.value is not None
        # Query from self; keys/values from sources.
        q = self.query(h).unsqueeze(2)  # [B, N, 1, H]
        k = self.key(gathered)  # [B, N, K, H]
        v = self.value(gathered)  # [B, N, K, H]
        scores = (q * k).sum(dim=-1) * self.scale  # [B, N, K]
        # Signed synapse gate on attention logits (excitatory/inhibitory).
        scores = scores + self.weights.unsqueeze(0)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, K, 1]
        return (attn * v).sum(dim=2)
