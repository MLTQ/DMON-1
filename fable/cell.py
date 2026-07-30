"""The shared cell rule: one GRUCell for the whole organism.

Input is [messages, drive] — 2H wide. grok's third slice (credit_drive) is
gone, which also removes the ~9% of parameters that were provably dead under
no-credit and silently inflating the matched-GRU budget (grok debt #22).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedRule(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRUCell(2 * hidden, hidden)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

    def forward(self, h: torch.Tensor, messages: torch.Tensor,
                drive: torch.Tensor) -> torch.Tensor:
        """h, messages, drive: [B, Nt, H] → new h [B, Nt, H].

        GRU output is a convex mix of tanh candidate and previous state, so
        cell state stays bounded — the substrate cannot blow up; only
        gradients can (handled in train.py).
        """
        b, nt, hdim = h.shape
        inp = torch.cat([messages, drive], dim=-1).reshape(b * nt, 2 * hdim)
        out = self.gru(inp, h.reshape(b * nt, hdim))
        return out.reshape(b, nt, hdim)
