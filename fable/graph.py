"""Directed dendrite graph with local attention aggregation.

Each cell owns K incoming dendrite slots naming specific source cells. Messages
are value-projected source states mixed by attention (shared query/key/value
maps + one per-edge logit scalar). No convolutional neighborhood, no credit
transport — the graph moves information forward, period.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriteGraph(nn.Module):
    """Wiring + aggregation for a field of `n_cells` with hidden width H.

    Buffers:
        sources      [N, K] long — source cell index per dendrite slot
    Parameters:
        logit        [N, K] — per-edge attention bias (the only per-edge params)
        query/key/value — shared H×H maps, bias-free

    Wiring rule: every slot samples uniformly (without replacement where
    possible) from all non-output cells — inputs, mirrors, internals. Output
    cells are pure sinks: nothing reads them, so the readout can never leak
    back into the field. Output cell i's slot 0 is force-wired to input cell
    i % n_input so sensory→output reachability holds at init.
    """

    def __init__(self, n_cells: int, hidden: int, n_dendrites: int,
                 input_idx: torch.Tensor, mirror_idx: torch.Tensor,
                 internal_idx: torch.Tensor, output_idx: torch.Tensor,
                 seed: int):
        super().__init__()
        self.n_cells = n_cells
        self.hidden = hidden
        self.k = n_dendrites

        gen = torch.Generator().manual_seed(seed)
        pool = torch.cat([input_idx, mirror_idx, internal_idx])
        sources = torch.empty(n_cells, self.k, dtype=torch.long)
        for cell in range(n_cells):
            perm = pool[torch.randperm(len(pool), generator=gen)]
            picks = perm[: self.k]
            if len(picks) < self.k:  # tiny fields: sample with replacement
                extra = pool[torch.randint(len(pool), (self.k - len(picks),), generator=gen)]
                picks = torch.cat([picks, extra])
            sources[cell] = picks
        for j, cell in enumerate(output_idx.tolist()):
            forced = input_idx[j % len(input_idx)]
            row = sources[cell]
            row[row == forced] = row[0].clone()  # dedupe before forcing slot 0
            row[0] = forced
        self.register_buffer("sources", sources)

        self.logit = nn.Parameter(
            torch.empty(n_cells, self.k).uniform_(-0.05, 0.05, generator=gen))
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        nn.init.orthogonal_(self.value.weight)

    def aggregate(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Messages for the `targets` cells only.

        h: [B, N, H]; targets: [Nt] long. Returns [B, Nt, H].

        value/key/query are applied to h once ([B, N, H]) and then gathered —
        never to the gathered [B, Nt, K, H] tensor. grok did the latter and it
        was a K× tax on the dominant cost (grok debt #1).
        """
        vh = self.value(h)
        kh = self.key(h)
        q = self.query(h[:, targets])                    # [B, Nt, H]
        src = self.sources[targets]                      # [Nt, K]
        g_v = vh[:, src]                                 # [B, Nt, K, H]
        g_k = kh[:, src]
        scores = (q.unsqueeze(2) * g_k).sum(-1) * self.hidden ** -0.5
        scores = scores + self.logit[targets].unsqueeze(0)
        coeff = F.softmax(scores, dim=-1)                # [B, Nt, K]
        return (coeff.unsqueeze(-1) * g_v).sum(2)        # [B, Nt, H]

    @torch.no_grad()
    def output_reachable_from_input(self, input_idx: torch.Tensor,
                                    output_idx: torch.Tensor) -> bool:
        """BFS along dendrites: is every output cell fed (transitively) by
        some input cell?"""
        feeds: dict[int, set[int]] = {}
        for tgt in range(self.n_cells):
            for src in self.sources[tgt].tolist():
                feeds.setdefault(src, set()).add(tgt)
        reached = set(input_idx.tolist())
        frontier = list(reached)
        while frontier:
            nxt = []
            for s in frontier:
                for t in feeds.get(s, ()):
                    if t not in reached:
                        reached.add(t)
                        nxt.append(t)
            frontier = nxt
        return all(o in reached for o in output_idx.tolist())
