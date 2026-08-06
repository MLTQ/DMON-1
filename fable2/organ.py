"""BrocaOrgan: output-tissue-only sensor/effector with per-depth zero-init heads."""

from __future__ import annotations

import torch
from torch import nn

from sol2.operators import EffectiveLinear
from sol2.organs import AttentiveOutputOrgan, OrganHealth


class BrocaOrgan(nn.Module):
    """Sense frozen language features; emit bounded per-depth control banks.

    The sensor maps one frozen-backbone feature into the organism's bounded state
    width. The effector side reads *only* dedicated output tissue through one
    attentive trunk, then K independent zero-initialized low-rank heads translate
    the shared summary into one control bank per preregistered injection depth.
    Representational transformation is learned end to end; nothing is copied
    coordinate-for-coordinate (the C1aa mandate).
    """

    def __init__(
        self,
        n_input: int,
        hidden: int,
        language_width: int,
        n_queries: int,
        n_depths: int,
        n_control_tokens: int,
        control_rank: int,
        trunk_width: int,
        *,
        bounded_operators: bool,
        operator_bound: float,
        value_gain: float,
        attention_temperature: float,
        control_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if n_depths < 1 or n_control_tokens < 1 or control_rank < 1:
            raise ValueError("depths, control tokens, and rank must be positive")
        if control_rank > language_width:
            raise ValueError("control_rank cannot exceed language_width")
        if control_gain <= 0:
            raise ValueError("control_gain must be positive")
        self.language_width = language_width
        self.n_depths = n_depths
        self.n_control_tokens = n_control_tokens
        self.control_rank = control_rank
        self.control_gain = float(control_gain)

        self.sensor = EffectiveLinear(
            language_width, hidden, bounded=bounded_operators, bound=operator_bound
        )
        self.sensory_gain = nn.Parameter(torch.zeros(n_input, hidden))
        self.sensory_bias = nn.Parameter(torch.zeros(n_input, hidden))

        self.trunk = AttentiveOutputOrgan(
            hidden,
            trunk_width,
            n_queries,
            bounded_operators=bounded_operators,
            operator_bound=operator_bound,
            value_gain=value_gain,
            attention_temperature=attention_temperature,
        )
        # Each depth owns an exactly-zero head so a fresh graft is a bitwise no-op
        # on the frozen backbone; the first optimizer step recruits these weights,
        # after which trunk and organism gradients become available.
        self.depth_heads = nn.ModuleList(
            nn.Linear(trunk_width, n_control_tokens * control_rank)
            for _ in range(n_depths)
        )
        for head in self.depth_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.depth_bases = nn.Parameter(
            torch.empty(n_depths, control_rank, language_width)
        )
        for depth in range(n_depths):
            nn.init.orthogonal_(self.depth_bases.data[depth])

    def sense(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a bounded memory write and input-tissue drive for one feature."""

        if features.ndim != 2 or features.shape[-1] != self.language_width:
            raise ValueError("language features must have shape [batch, language_width]")
        embedded = torch.tanh(self.sensor(features))
        sensory_gain = 1.0 + 0.5 * torch.tanh(self.sensory_gain)
        sensory_bias = 0.1 * torch.tanh(self.sensory_bias)
        return embedded, embedded.unsqueeze(1) * sensory_gain + sensory_bias

    def read(
        self,
        output_cells: torch.Tensor,
        *,
        collect_health: bool = False,
    ) -> tuple[torch.Tensor, OrganHealth | None]:
        """Read output tissue into per-depth bounded control banks.

        Returns `[batch, n_depths, n_control_tokens, language_width]` with every
        component in `(-control_gain, control_gain)`.
        """

        summary, health = self.trunk(output_cells, collect_health=collect_health)
        # Bound the summary before the heads and keep the coefficient stage linear:
        # an intermediate tanh over the unbounded trunk decoder saturated bit-exact
        # under moderate head growth, erasing arm differentials below fp32
        # resolution while the static control kept inflating (observed on the toy
        # curriculum). Boundedness lives at exactly one place — the control level.
        summary = torch.tanh(summary)
        coefficients = torch.stack(
            [head(summary) for head in self.depth_heads], dim=1
        ).view(-1, self.n_depths, self.n_control_tokens, self.control_rank)
        controls = torch.einsum("bdtr,drw->bdtw", coefficients, self.depth_bases)
        return self.control_gain * torch.tanh(controls), health

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        norms = {
            "sensor": self.sensor.spectral_norm(),
            **{
                f"trunk.{name}": value
                for name, value in self.trunk.operator_norms().items()
            },
        }
        for depth, head in enumerate(self.depth_heads):
            norms[f"head.{depth}"] = float(
                torch.linalg.matrix_norm(head.weight, ord=2)
            )
            norms[f"basis.{depth}"] = float(
                torch.linalg.matrix_norm(self.depth_bases[depth], ord=2)
            )
        return norms
