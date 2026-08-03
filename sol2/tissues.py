"""Typed, contractive cell rules for SOL2 tissues."""

from __future__ import annotations

import torch
from torch import nn

from .operators import EffectiveLinear, inverse_sigmoid


class TissueRule(nn.Module):
    """Leaky integration toward a bounded target with bounded time constants."""

    def __init__(
        self,
        hidden: int,
        *,
        initial_alpha: float,
        bounded_operators: bool,
        operator_bound: float,
        alpha_min: float = 0.02,
        alpha_max: float = 0.80,
    ) -> None:
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        width = 3 * hidden
        self.target = EffectiveLinear(
            width,
            hidden,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        self.alpha = EffectiveLinear(
            width,
            hidden,
            bounded=bounded_operators,
            bound=operator_bound,
        )
        normalized = (initial_alpha - alpha_min) / (alpha_max - alpha_min)
        nn.init.constant_(self.alpha.bias, inverse_sigmoid(normalized))

    def forward(
        self,
        hidden: torch.Tensor,
        message: torch.Tensor,
        drive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden, message, drive], dim=-1)
        target = torch.tanh(self.target(x))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(
            self.alpha(x)
        )
        updated = (1.0 - alpha) * hidden + alpha * target
        return updated, alpha

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return {
            "target": self.target.spectral_norm(),
            "alpha": self.alpha.spectral_norm(),
        }


class TissueBank(nn.Module):
    """One genome/rule per tissue type, shared only within that tissue."""

    INITIAL_ALPHA = {
        "input": 0.50,
        "compute": 0.12,
        "relay": 0.65,
        "output": 0.40,
    }

    def __init__(self, hidden: int, bounded_operators: bool, operator_bound: float):
        super().__init__()
        self.rules = nn.ModuleDict(
            {
                name: TissueRule(
                    hidden,
                    initial_alpha=alpha,
                    bounded_operators=bounded_operators,
                    operator_bound=operator_bound,
                )
                for name, alpha in self.INITIAL_ALPHA.items()
            }
        )

    def forward(self, name: str, hidden, message, drive):
        return self.rules[name](hidden, message, drive)

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return {
            f"{name}.{op}": value
            for name, rule in self.rules.items()
            for op, value in rule.operator_norms().items()
        }
