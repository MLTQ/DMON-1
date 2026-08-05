"""Bounded attentive white-matter transport from relay to output tissue."""

from __future__ import annotations

import math

import torch
from torch import nn

from .operators import EffectiveLinear


class RelayOutputAttention(nn.Module):
    """Give differentiated output cells content-addressed access to relay tissue."""

    def __init__(
        self,
        n_output: int,
        hidden: int,
        *,
        bounded_operators: bool,
        operator_bound: float,
        value_gain: float,
        attention_temperature: float,
        transport_gain: float,
        initial_gate: float,
    ) -> None:
        super().__init__()
        if n_output < 1 or hidden < 1 or transport_gain <= 0:
            raise ValueError("relay transport dimensions and gain must be positive")
        if not 0 <= initial_gate < 1:
            raise ValueError("relay transport initial gate must be in [0, 1)")
        self.hidden = hidden
        self.n_output = n_output
        self.bounded_operators = bounded_operators
        self.value_gain = value_gain
        self.attention_temperature = attention_temperature
        self.transport_gain = transport_gain
        self.queries = nn.Parameter(torch.empty(n_output, hidden))
        nn.init.normal_(self.queries, std=hidden**-0.5)
        self.gate_logit = nn.Parameter(
            torch.full((n_output,), math.atanh(initial_gate))
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
        self.output = EffectiveLinear(
            hidden, hidden, bias=False, bounded=bounded_operators, bound=operator_bound
        )

    def forward(
        self, relay_cells: torch.Tensor, output_cells: torch.Tensor
    ) -> torch.Tensor:
        if relay_cells.ndim != 3 or output_cells.ndim != 3:
            raise ValueError("relay and output tissue must have [batch, cells, hidden] shape")
        if relay_cells.shape[0] != output_cells.shape[0]:
            raise ValueError("relay and output tissue batch sizes must match")
        if output_cells.shape[1:] != (self.n_output, self.hidden):
            raise ValueError("output tissue shape does not match relay transport")
        if relay_cells.shape[2] != self.hidden or relay_cells.shape[1] < 1:
            raise ValueError("relay tissue shape does not match relay transport")

        learned_queries = self.queries.unsqueeze(0) + output_cells
        queries = self.query(learned_queries)
        keys = self.key(relay_cells)
        values = self.value(relay_cells)
        if self.bounded_operators:
            queries = torch.tanh(queries)
            keys = torch.tanh(keys)
            queries = queries / queries.norm(dim=-1, keepdim=True).clamp_min(1.0)
            keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1.0)
            values = self.value_gain * torch.tanh(values)
            scores = torch.einsum("boh,brh->bor", queries, keys)
            scores = scores / self.attention_temperature
        else:
            scores = torch.einsum("boh,brh->bor", queries, keys) * self.hidden**-0.5
        attention = torch.softmax(scores, dim=-1)
        summary = torch.einsum("bor,brh->boh", attention, values)
        summary = self.output(summary)
        if self.bounded_operators:
            summary = torch.tanh(summary)
        gate = torch.tanh(self.gate_logit).view(1, self.n_output, 1)
        return self.transport_gain * gate * summary

    @torch.no_grad()
    def gate_rms(self) -> float:
        return float(torch.tanh(self.gate_logit).pow(2).mean().sqrt())

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return {
            "query": self.query.spectral_norm(),
            "key": self.key.spectral_norm(),
            "value": self.value.spectral_norm(),
            "output": self.output.spectral_norm(),
        }
