"""Shared local cell rule. Parameters are global; state is per-cell."""

from __future__ import annotations

import torch
from torch import nn


class SharedGRURule(nn.Module):
    """One GRU applied at every mutable cell.

    Input to the GRU is the concatenation of dendritic message and external
    drive (zeros for non-input cells). Hidden state is private per cell and
    persists across tokens — that persistence is the creature's working memory.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        # message + drive → 2H input features
        self.gru = nn.GRUCell(hidden * 2, hidden)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

    def forward(
        self,
        h: torch.Tensor,
        messages: torch.Tensor,
        drive: torch.Tensor,
    ) -> torch.Tensor:
        """h, messages, drive: [B, N, H] → updated h [B, N, H]."""

        b, n, hid = h.shape
        inp = torch.cat([messages, drive], dim=-1)  # [B, N, 2H]
        out = self.gru(inp.reshape(b * n, 2 * hid), h.reshape(b * n, hid))
        return out.reshape(b, n, hid)
