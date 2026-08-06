"""Organ-specific attentive interfaces over dedicated output tissue."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .operators import EffectiveLinear


@dataclass
class OrganHealth:
    attention_entropy: float
    attention_max: float
    effective_cells: float


class AttentiveOutputOrgan(nn.Module):
    """Read a variable-sized output tissue with a fixed bank of learned queries."""

    def __init__(
        self,
        hidden: int,
        output_size: int,
        n_queries: int,
        *,
        bounded_operators: bool,
        operator_bound: float,
        value_gain: float,
        attention_temperature: float,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_queries = n_queries
        self.bounded_operators = bounded_operators
        self.value_gain = value_gain
        self.attention_temperature = attention_temperature
        self.queries = nn.Parameter(torch.empty(n_queries, hidden))
        nn.init.normal_(self.queries, std=hidden**-0.5)
        self.key = EffectiveLinear(
            hidden,
            hidden,
            bias=False,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.value = EffectiveLinear(
            hidden,
            hidden,
            bias=False,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.output = EffectiveLinear(
            hidden,
            hidden,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        width = n_queries * hidden
        # A larger epsilon caps the gain on a newly grafted or nearly silent organ.
        # Default LayerNorm epsilon amplifies a zero-scale interface by ~316x.
        self.norm = nn.LayerNorm(width, eps=1e-2)
        self.decoder = nn.Linear(width, output_size)

    def forward(
        self,
        cells: torch.Tensor,
        *,
        collect_health: bool = False,
    ) -> tuple[torch.Tensor, OrganHealth | None]:
        keys = self.key(cells)
        values = self.value(cells)
        queries = self.queries.unsqueeze(0).expand(cells.shape[0], -1, -1)
        if self.bounded_operators:
            queries = torch.tanh(queries)
            keys = torch.tanh(keys)
            queries = queries / queries.norm(dim=-1, keepdim=True).clamp_min(1.0)
            keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1.0)
            values = self.value_gain * torch.tanh(values)
            scores = torch.einsum("bqh,bnh->bqn", queries, keys)
            scores = scores / self.attention_temperature
        else:
            scores = torch.einsum("bqh,bnh->bqn", queries, keys) * self.hidden**-0.5
        attention = torch.softmax(scores, dim=-1)
        summary = torch.einsum("bqn,bnh->bqh", attention, values)
        summary = self.output(summary)
        if self.bounded_operators:
            summary = torch.tanh(summary)
        logits = self.decoder(self.norm(summary.flatten(1)))

        health = None
        if collect_health:
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1)
            health = OrganHealth(
                attention_entropy=float(entropy.detach().mean()),
                attention_max=float(attention.detach().max(dim=-1).values.mean()),
                effective_cells=float(entropy.detach().exp().mean()),
            )
        return logits, health

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return {
            "key": self.key.spectral_norm(),
            "value": self.value.spectral_norm(),
            "output": self.output.spectral_norm(),
        }

    @staticmethod
    def maximum_entropy(n_cells: int) -> float:
        return math.log(max(n_cells, 1))


class SensorEffectorOrgan(nn.Module):
    """A detachable, independently parameterized sensor and output interface."""

    def __init__(
        self,
        n_input: int,
        hidden: int,
        vocab_size: int,
        n_queries: int,
        *,
        bounded_operators: bool,
        operator_bound: float,
        value_gain: float,
        attention_temperature: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        nn.init.normal_(self.embedding.weight, std=0.16)
        self.sensory_gain = nn.Parameter(torch.zeros(n_input, hidden))
        self.sensory_bias = nn.Parameter(torch.zeros(n_input, hidden))
        self.output = AttentiveOutputOrgan(
            hidden,
            vocab_size,
            n_queries,
            bounded_operators=bounded_operators,
            operator_bound=operator_bound,
            value_gain=value_gain,
            attention_temperature=attention_temperature,
        )

    def sense(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded memory write and input-tissue sensory drive."""

        embedded = torch.tanh(self.embedding(tokens))
        sensory_gain = 1.0 + 0.5 * torch.tanh(self.sensory_gain)
        sensory_bias = 0.1 * torch.tanh(self.sensory_bias)
        drive = embedded.unsqueeze(1) * sensory_gain + sensory_bias
        return embedded, drive

    def read(
        self,
        output_cells: torch.Tensor,
        *,
        collect_health: bool = False,
    ) -> tuple[torch.Tensor, OrganHealth | None]:
        return self.output(output_cells, collect_health=collect_health)

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return self.output.operator_norms()
