"""Null models for S0 — "the substrate is doing something" is a measurement."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MatchedGRU(nn.Module):
    """Character LM baseline with a comparable parameter budget.

    Single-layer GRU over character embeddings with a linear head. Trained on
    the identical multi-stream online protocol as the creature.
    """

    def __init__(self, vocab_size: int, hidden: int = 256, n_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

    def step(
        self, token_ids: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """token_ids [B] → logits [B, V], h."""

        emb = self.embed(token_ids).unsqueeze(1)  # [B, 1, H]
        out, h = self.rnn(emb, h)
        logits = self.head(out.squeeze(1))
        return logits, h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def match_hidden_for_budget(vocab_size: int, target_params: int) -> int:
    """Pick a GRU hidden size whose param count is near target_params."""

    best_h, best_err = 64, float("inf")
    for h in range(32, 1025, 8):
        m = MatchedGRU(vocab_size, hidden=h)
        err = abs(m.count_parameters() - target_params)
        if err < best_err:
            best_h, best_err = h, err
    return best_h
