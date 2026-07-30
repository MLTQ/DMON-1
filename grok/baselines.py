"""Matched conventional controls for S0."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MatchedGRU(nn.Module):
    """Character LM GRU with the same online/chunk interface."""

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
        emb = self.embed(token_ids).unsqueeze(1)
        out, h = self.rnn(emb, h)
        return self.head(out.squeeze(1)), h

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = []
        logits = None
        for t in range(tokens.shape[1]):
            logits, h = self.step(tokens[:, t], h)
            losses.append(F.cross_entropy(logits, targets[:, t]))
        assert logits is not None
        return logits, h, torch.stack(losses).mean()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MatchedTransformer(nn.Module):
    """Tiny causal transformer control (context = chunk window only)."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        _state: None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        b, t = tokens.shape
        if t > self.max_len:
            raise ValueError("chunk longer than transformer max_len")
        pos = torch.arange(t, device=tokens.device)
        x = self.embed(tokens) + self.pos(pos)[None, :, :]
        # Causal mask: True means "ignore" in nn.TransformerEncoder.
        causal = torch.triu(
            torch.ones(t, t, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(x, mask=causal)
        logits = self.head(h)
        loss = F.cross_entropy(logits.reshape(b * t, -1), targets.reshape(b * t))
        return logits[:, -1], None, loss


def match_hidden_for_budget(vocab_size: int, target_params: int) -> int:
    best_h, best_err = 64, float("inf")
    for h in range(32, 1025, 8):
        err = abs(MatchedGRU(vocab_size, hidden=h).count_parameters() - target_params)
        if err < best_err:
            best_h, best_err = h, err
    return best_h


def match_transformer_for_budget(vocab_size: int, target_params: int) -> int:
    best_h, best_err = 64, float("inf")
    for h in range(32, 513, 8):
        if h % 4 != 0:
            continue
        err = abs(
            MatchedTransformer(vocab_size, hidden=h).count_parameters() - target_params
        )
        if err < best_err:
            best_h, best_err = h, err
    return best_h
