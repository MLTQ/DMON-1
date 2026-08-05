"""Continuous sensory and low-rank effector organs for language backbones."""

from __future__ import annotations

import torch
from torch import nn

from .operators import EffectiveLinear
from .organs import AttentiveOutputOrgan, OrganHealth


class ContinuousLanguageOrgan(nn.Module):
    """Connect frozen language features to SOL2 without a vocabulary-sized bypass.

    The sensor maps one language-backbone feature into the organism's bounded state
    width. The effector reads only output tissue and produces a small bank of
    continuous control tokens through a shared low-rank basis.
    """

    def __init__(
        self,
        n_input: int,
        hidden: int,
        language_width: int,
        n_queries: int,
        n_control_tokens: int,
        control_rank: int,
        *,
        bounded_operators: bool,
        operator_bound: float,
        value_gain: float,
        attention_temperature: float,
        control_gain: float = 1.0,
        recall_gain: float = 0.25,
    ) -> None:
        super().__init__()
        if language_width < 1 or n_control_tokens < 1 or control_rank < 1:
            raise ValueError(
                "language_width, n_control_tokens, and control_rank must be positive"
            )
        if control_rank > language_width:
            raise ValueError("control_rank cannot exceed language_width")
        if control_gain <= 0:
            raise ValueError("control_gain must be positive")
        if not 0 < recall_gain <= 1.0:
            raise ValueError("recall_gain must be in (0, 1]")

        self.language_width = language_width
        self.n_control_tokens = n_control_tokens
        self.control_rank = control_rank
        self.control_gain = float(control_gain)
        self.sensor = EffectiveLinear(
            language_width,
            hidden,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.sensory_gain = nn.Parameter(torch.zeros(n_input, hidden))
        self.sensory_bias = nn.Parameter(torch.zeros(n_input, hidden))
        self.output = AttentiveOutputOrgan(
            hidden,
            n_control_tokens * control_rank,
            n_queries,
            bounded_operators=bounded_operators,
            operator_bound=operator_bound,
            value_gain=value_gain,
            attention_temperature=attention_temperature,
        )
        self.control_basis = nn.Parameter(torch.empty(control_rank, language_width))
        nn.init.orthogonal_(self.control_basis)
        self.recall_query = EffectiveLinear(
            hidden,
            hidden,
            bias=False,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.recall_key = EffectiveLinear(
            hidden,
            hidden,
            bias=False,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.recall_value = EffectiveLinear(
            hidden,
            hidden,
            bias=False,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.recall_output = EffectiveLinear(
            hidden,
            hidden,
            bias=False,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.recall_gain = float(recall_gain)
        self.value_gain = value_gain
        self.attention_temperature = attention_temperature
        self.bounded_operators = bounded_operators

        # A new organ must be an exact no-op on its frozen language backbone. The
        # first optimization step recruits these coefficients; deeper organism and
        # attention gradients become available once this zero-up layer moves.
        nn.init.zeros_(self.output.decoder.weight)
        nn.init.zeros_(self.output.decoder.bias)

    def sense(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a bounded memory write and input-tissue drive for one LLM feature."""

        if features.ndim != 2 or features.shape[-1] != self.language_width:
            raise ValueError(
                "language features must have shape [batch, language_width]"
            )
        embedded = torch.tanh(self.sensor(features))
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
        coefficients, health = self.output(
            output_cells, collect_health=collect_health
        )
        coefficients = torch.tanh(
            coefficients.view(-1, self.n_control_tokens, self.control_rank)
        )
        controls = torch.einsum("btr,rw->btw", coefficients, self.control_basis)
        return self.control_gain * torch.tanh(controls), health

    def recall(
        self,
        query_embedding: torch.Tensor,
        memory_cells: torch.Tensor,
        *,
        valid_count: int,
        collect_health: bool = False,
    ) -> tuple[torch.Tensor, OrganHealth | None]:
        """Content-address stream memory and return a bounded recurrent drive."""

        if (
            query_embedding.ndim != 2
            or query_embedding.shape[-1] != memory_cells.shape[-1]
        ):
            raise ValueError("recall query must have [batch, hidden] shape")
        if memory_cells.ndim != 3 or memory_cells.shape[0] != query_embedding.shape[0]:
            raise ValueError("memory cells must have [batch, cells, hidden] shape")
        count = min(max(int(valid_count), 0), memory_cells.shape[1])
        if count == 0:
            return torch.zeros_like(query_embedding), None
        memory = memory_cells[:, :count]
        query = self.recall_query(query_embedding)
        keys = self.recall_key(memory)
        values = self.recall_value(memory)
        if self.bounded_operators:
            query = torch.tanh(query)
            keys = torch.tanh(keys)
            query = query / query.norm(dim=-1, keepdim=True).clamp_min(1.0)
            keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1.0)
            values = self.value_gain * torch.tanh(values)
            scores = torch.einsum("bh,bnh->bn", query, keys)
            scores = scores / self.attention_temperature
        else:
            scores = torch.einsum("bh,bnh->bn", query, keys)
            scores = scores * query.shape[-1] ** -0.5
        attention = torch.softmax(scores, dim=-1)
        summary = torch.einsum("bn,bnh->bh", attention, values)
        recalled = self.recall_output(summary)
        recalled = self.recall_gain * torch.tanh(recalled)
        health = None
        if collect_health:
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1)
            health = OrganHealth(
                attention_entropy=float(entropy.detach().mean()),
                attention_max=float(attention.detach().max(dim=-1).values.mean()),
                effective_cells=float(entropy.detach().exp().mean()),
            )
        return recalled, health

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return {
            "sensor": self.sensor.spectral_norm(),
            **{f"output.{name}": value for name, value in self.output.operator_norms().items()},
            "control_basis": float(torch.linalg.matrix_norm(self.control_basis, ord=2)),
            "recall.query": self.recall_query.spectral_norm(),
            "recall.key": self.recall_key.spectral_norm(),
            "recall.value": self.recall_value.spectral_norm(),
            "recall.output": self.recall_output.spectral_norm(),
        }
