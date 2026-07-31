"""The liquid cell rule: CfC-style leaky integration with learned, bounded,
input-dependent time constants. F3's candidate replacement for the GRU rule.

Update, per cell:
    z  = [messages, drive, h]                       (3H)
    b  = tanh(W_b z)                                (backbone, width B)
    g  = tanh(W_g b)                                (target state)
    a  = 1 / (tau_min + (tau_max - tau_min) * sigmoid(W_a b))
    h' = (1 - a) * h + a * g

Why this shape:
  - a in (1/tau_max, 1/tau_min] with tau_min = 1, so the h-backbone of the
    per-step map is (1 - a) in [0, 1): a leak toward a bounded target. State
    is bounded by construction (|h'| <= max(|h|, 1)) and the recurrent
    Jacobian's backbone term is contractive — the property the GRU rule
    lacked in practice: three E1 creatures and two F7 creatures died of
    gradient blowups no guard could cross.
  - tau is input-dependent and bounded: cells learn their own timescales,
    from reactive (tau ~ 1) to integrating (tau ~ tau_max), instead of
    inheriting one hand-set steps-per-token everywhere. F0 measured the
    hand-set timescales earning almost nothing (mirror delta <= 0.17).
  - Closed form; no ODE solver; Delta-t is fixed at 1 here (Delta-t-aware
    asynchrony is the composability step, tested separately).

The backbone width B is solved so the rule's parameter count matches the
GRU rule (9H^2 + 6H) within a fraction of a percent — the whole-organism
comparison stays parameter-matched without touching any other module.

Honesty note: contraction of the h-backbone does not *prove* bounded
gradients (the W-paths through g and a can still grow), so F3's stability
claim is tested, not assumed: the arms include the exact regimes that
killed the GRU rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gru_rule_param_count(hidden: int) -> int:
    return 9 * hidden * hidden + 6 * hidden


def backbone_width(hidden: int) -> int:
    """Solve B so 3H*B + B + 2*(B*H + H) matches the GRU rule's count."""
    target = gru_rule_param_count(hidden)
    return max(1, round((target - 2 * hidden) / (5 * hidden + 1)))


class LiquidRule(nn.Module):
    def __init__(self, hidden: int, tau_min: float = 1.0,
                 tau_max: float = 100.0):
        super().__init__()
        self.hidden = hidden
        self.tau_min = tau_min
        self.tau_max = tau_max
        b = backbone_width(hidden)
        self.backbone = nn.Linear(3 * hidden, b)
        self.target = nn.Linear(b, hidden)
        self.tau_gate = nn.Linear(b, hidden)
        # Bias the gate toward fast cells at init (sigmoid(-2) ~ 0.12 →
        # tau ~ 13): the organism starts responsive and learns to slow
        # down, not vice versa.
        nn.init.constant_(self.tau_gate.bias, -2.0)
        self.last_alpha_mean: float | None = None

    def forward(self, h: torch.Tensor, messages: torch.Tensor,
                drive: torch.Tensor) -> torch.Tensor:
        z = torch.cat([messages, drive, h], dim=-1)
        b = torch.tanh(self.backbone(z))
        g = torch.tanh(self.target(b))
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(
            self.tau_gate(b))
        a = 1.0 / tau
        with torch.no_grad():
            self.last_alpha_mean = float(a.mean())
        return (1.0 - a) * h + a * g
