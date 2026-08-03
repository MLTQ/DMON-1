"""Sparse directed connectome with dormant growth slots and bounded operators."""

from __future__ import annotations

import torch
from torch import nn

from .operators import EffectiveLinear


class DirectedConnectome(nn.Module):
    def __init__(
        self,
        n_cells: int,
        hidden: int,
        n_dendrites: int,
        initial_active: int,
        source_pool: torch.Tensor,
        input_idx: torch.Tensor,
        internal_idx: torch.Tensor,
        output_idx: torch.Tensor,
        *,
        seed: int,
        bounded_operators: bool,
        operator_bound: float,
        value_gain: float,
        attention_temperature: float,
        edge_logit_limit: float,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.hidden = hidden
        self.n_dendrites = n_dendrites
        self.initial_active = initial_active
        self.bounded_operators = bounded_operators
        self.value_gain = value_gain
        self.attention_temperature = attention_temperature
        self.edge_logit_limit = edge_logit_limit

        generator = torch.Generator().manual_seed(seed)
        sources = torch.empty(n_cells, n_dendrites, dtype=torch.long)
        outputs = set(output_idx.tolist())
        for target in range(n_cells):
            target_pool = internal_idx if target in outputs else source_pool
            pool = target_pool[target_pool != target]
            if len(pool) >= n_dendrites:
                picks = pool[torch.randperm(len(pool), generator=generator)[:n_dendrites]]
            else:
                extra = pool[
                    torch.randint(len(pool), (n_dendrites - len(pool),), generator=generator)
                ]
                picks = torch.cat([pool, extra])
            sources[target] = picks
        for offset, target in enumerate(internal_idx.tolist()):
            forced = input_idx[offset % len(input_idx)]
            row = sources[target]
            duplicate = torch.nonzero(row == forced, as_tuple=True)[0]
            if len(duplicate) and int(duplicate[0]) != 0:
                row[int(duplicate[0])] = row[0]
            row[0] = forced
        for offset, target in enumerate(output_idx.tolist()):
            forced = internal_idx[offset % len(internal_idx)]
            row = sources[target]
            duplicate = torch.nonzero(row == forced, as_tuple=True)[0]
            if len(duplicate) and int(duplicate[0]) != 0:
                row[int(duplicate[0])] = row[0]
            row[0] = forced

        active = torch.zeros(n_cells, n_dendrites, dtype=torch.bool)
        active[:, :initial_active] = True
        self.register_buffer("sources", sources)
        self.register_buffer("active", active)
        self.edge_logit = nn.Parameter(
            torch.empty(n_cells, n_dendrites).uniform_(-0.05, 0.05, generator=generator)
        )
        self.query = EffectiveLinear(
            hidden, hidden, bias=False, bounded=bounded_operators, bound=operator_bound
        )
        self.key = EffectiveLinear(
            hidden, hidden, bias=False, bounded=bounded_operators, bound=operator_bound
        )
        self.value = EffectiveLinear(
            hidden, hidden, bias=False, bounded=bounded_operators, bound=operator_bound
        )

    def aggregate(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        q = self.query(hidden[:, targets])
        keys = self.key(hidden)
        values = self.value(hidden)
        if self.bounded_operators:
            # Damp-only normalization. Ordinary cosine normalization multiplies a
            # near-zero query and its gradient by 1/eps; the first end-to-end smoke
            # exposed a 17.9k first-update gradient through exactly that path.
            # Unit-ball projection preserves small signals and only scales down.
            q = torch.tanh(q)
            keys = torch.tanh(keys)
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0)
            keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1.0)
            values = self.value_gain * torch.tanh(values)

        sources = self.sources[targets]
        gathered_k = keys[:, sources]
        gathered_v = values[:, sources]
        if self.bounded_operators:
            scores = (q.unsqueeze(2) * gathered_k).sum(-1) / self.attention_temperature
            edge_bias = self.edge_logit_limit * torch.tanh(self.edge_logit[targets])
        else:
            scores = (q.unsqueeze(2) * gathered_k).sum(-1) * self.hidden ** -0.5
            edge_bias = self.edge_logit[targets]
        scores = scores + edge_bias.unsqueeze(0)
        mask = self.active[targets].unsqueeze(0)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        coefficient = torch.softmax(scores, dim=-1)
        return (coefficient.unsqueeze(-1) * gathered_v).sum(dim=2)

    @torch.no_grad()
    def reachable(self, input_idx: torch.Tensor, required: torch.Tensor) -> bool:
        feeds: dict[int, set[int]] = {}
        for target in range(self.n_cells):
            for slot in torch.nonzero(self.active[target], as_tuple=True)[0].tolist():
                source = int(self.sources[target, slot])
                feeds.setdefault(source, set()).add(target)
        reached = set(input_idx.tolist())
        frontier = list(reached)
        while frontier:
            source = frontier.pop()
            for target in feeds.get(source, ()):
                if target not in reached:
                    reached.add(target)
                    frontier.append(target)
        return all(int(cell) in reached for cell in required.tolist())

    @torch.no_grad()
    def output_cells_are_sinks(self, output_idx: torch.Tensor) -> bool:
        outputs = set(output_idx.tolist())
        active_sources = self.sources[self.active]
        return not any(int(source) in outputs for source in active_sources.tolist())

    @torch.no_grad()
    def targets_read_only_from(
        self, targets: torch.Tensor, allowed_sources: torch.Tensor
    ) -> bool:
        allowed = set(allowed_sources.tolist())
        sources = self.sources[targets]
        active = self.active[targets]
        return all(int(source) in allowed for source in sources[active].tolist())

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return {
            "query": self.query.spectral_norm(),
            "key": self.key.spectral_norm(),
            "value": self.value.spectral_norm(),
        }
