"""Shared local cell rule with credit-conditioned drive."""

from __future__ import annotations

import torch
from torch import nn


class SharedGRURule(nn.Module):
    """One GRU at every mutable cell.

    Inputs: dendritic message, external drive, and eligibility-gated credit drive.
    Private hidden state is the working memory of the organism.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        # message + drive + credit_drive → 3H
        self.gru = nn.GRUCell(hidden * 3, hidden)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

    def forward(
        self,
        h: torch.Tensor,
        messages: torch.Tensor,
        drive: torch.Tensor,
        credit_drive: torch.Tensor,
    ) -> torch.Tensor:
        b, n, hid = h.shape
        inp = torch.cat([messages, drive, credit_drive], dim=-1)
        out = self.gru(inp.reshape(b * n, 3 * hid), h.reshape(b * n, hid))
        return out.reshape(b, n, hid)
