"""Parameter-matched GRU and causal-transformer controls for SOL2."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MatchedGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, vocab_size)

    def initial_state(self, batch: int, device: str | torch.device) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)

    def step(self, tokens: torch.Tensor, state: torch.Tensor):
        output, state = self.gru(self.embedding(tokens).unsqueeze(1), state)
        return self.readout(output[:, 0]), state

    def forward_chunk(self, tokens, targets, state, **_):
        output, state = self.gru(self.embedding(tokens), state)
        logits = self.readout(output)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return loss, state, None


class MatchedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        *,
        max_len: int,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.position = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.readout = nn.Linear(hidden, vocab_size)

    def initial_state(self, batch: int, device: str | torch.device):
        return torch.zeros(batch, 0, dtype=torch.long, device=device)

    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)[None]
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        return self.readout(self.encoder(hidden, mask=mask))

    def step(self, tokens: torch.Tensor, state: torch.Tensor):
        context = torch.cat([state, tokens[:, None]], dim=1)[:, -self.max_len :]
        return self._forward(context)[:, -1], context

    def forward_chunk(self, tokens, targets, state, **_):
        logits = self._forward(tokens)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten()), state, None


def gru_param_count(vocab_size: int, hidden: int) -> int:
    return 6 * hidden * hidden + (2 * vocab_size + 6) * hidden + vocab_size


def transformer_param_count(
    vocab_size: int, hidden: int, *, layers: int = 2, max_len: int = 128
) -> int:
    per_layer = 12 * hidden * hidden + 13 * hidden
    return (
        vocab_size * hidden
        + max_len * hidden
        + layers * per_layer
        + hidden * vocab_size
        + vocab_size
    )


def match_gru_hidden(vocab_size: int, target: int) -> int:
    return min(range(8, 4097), key=lambda h: abs(gru_param_count(vocab_size, h) - target))


def match_transformer_hidden(
    vocab_size: int, target: int, *, max_len: int, heads: int = 4
) -> int:
    candidates = range(heads, 2049, heads)
    return min(
        candidates,
        key=lambda h: abs(
            transformer_param_count(vocab_size, h, max_len=max_len) - target
        ),
    )
