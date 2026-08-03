"""Linear operators with parameter-neutral spectral-scale control."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class EffectiveLinear(nn.Module):
    """A linear map with optional power-iteration spectral rescaling.

    The raw parameterization and parameter count are identical in bounded and
    unbounded arms. Only the effective weight used by the forward pass changes. One
    power iteration is an efficient scale estimate, not a proof of the exact norm;
    exact effective norms are measured separately in health telemetry.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        bounded: bool = True,
        bound: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bounded = bounded
        self.bound = float(bound)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("power_u", F.normalize(torch.randn(out_features), dim=0))
        nn.init.orthogonal_(self.weight)

    def _power_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            v = F.normalize(self.weight.t().mv(self.power_u), dim=0, eps=1e-12)
            u = F.normalize(self.weight.mv(v), dim=0, eps=1e-12)
            if self.training:
                self.power_u.copy_(u)
        return u.detach(), v.detach()

    def effective_weight(self) -> torch.Tensor:
        if not self.bounded:
            return self.weight
        u, v = self._power_vectors()
        sigma = torch.dot(u, self.weight.mv(v)).abs().clamp_min(1e-12)
        scale = torch.clamp(sigma / self.bound, min=1.0)
        return self.weight / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def spectral_norm(self) -> float:
        return float(torch.linalg.matrix_norm(self.effective_weight(), ord=2))


def inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))
