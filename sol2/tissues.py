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
        target_shift: torch.Tensor | None = None,
        alpha_shift: torch.Tensor | None = None,
        adapter_down: torch.Tensor | None = None,
        adapter_up: torch.Tensor | None = None,
        adapter_gain: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden, message, drive], dim=-1)
        target_logits = self.target(x)
        alpha_logits = self.alpha(x)
        if adapter_down is not None:
            if adapter_up is None:
                raise ValueError("adapter down/up projections must be supplied together")
            bottleneck = torch.tanh(torch.einsum("bni,nir->bnr", x, adapter_down))
            residual = torch.einsum("bnr,nrh->bnh", bottleneck, adapter_up)
            target_logits = target_logits + adapter_gain * torch.tanh(residual)
        if target_shift is not None:
            target_logits = target_logits + target_shift
        if alpha_shift is not None:
            alpha_logits = alpha_logits + alpha_shift
        target = torch.tanh(target_logits)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(
            alpha_logits
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

    def forward(
        self,
        name: str,
        hidden,
        message,
        drive,
        target_shift=None,
        alpha_shift=None,
        adapter_down=None,
        adapter_up=None,
        adapter_gain=0.0,
    ):
        return self.rules[name](
            hidden,
            message,
            drive,
            target_shift=target_shift,
            alpha_shift=alpha_shift,
            adapter_down=adapter_down,
            adapter_up=adapter_up,
            adapter_gain=adapter_gain,
        )

    @torch.no_grad()
    def operator_norms(self) -> dict[str, float]:
        return {
            f"{name}.{op}": value
            for name, rule in self.rules.items()
            for op, value in rule.operator_norms().items()
        }
