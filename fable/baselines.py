"""Parameter-matched GRU control on the identical stream.

The GRU gets everything the creature gets: same stream, same chunking, same
optimizer, same warmup+cosine schedule (sol never gave its baseline the
schedule that produced its own largest gain — fable always does), and a
seeded, independently reproducible init (grok never seeded its control,
debt #16).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MatchedGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.readout = nn.Linear(hidden, vocab_size)

    def initial_state(self, batch: int, device: str | torch.device) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)

    def step(self, tokens: torch.Tensor, state: torch.Tensor):
        out, state = self.gru(self.embed(tokens).unsqueeze(1), state)
        return self.readout(out[:, 0]), state

    def forward_chunk(self, tokens: torch.Tensor, targets: torch.Tensor,
                      state: torch.Tensor):
        out, state = self.gru(self.embed(tokens), state)
        logits = self.readout(out)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, state, None


class MatchedTransformer(nn.Module):
    """Causal transformer control on the identical stream.

    PROJECT.md's S0 pass condition names a parameter-matched GRU **and**
    transformer; only the GRU had been run. Included for that reason, not
    because it is expected to be the harder baseline — sol's S0 measured
    transformer 2.600 vs GRU 2.255 at matched budget on this corpus.

    The architectural caveat is real and must travel with any number: a
    transformer has no persistent state, so it cannot carry information
    across chunk boundaries the way the creature and GRU do. Its context is
    the current chunk. That is not a handicap imposed by this harness — it is
    what the architecture is — but it means the comparison is
    'matched parameters on the same stream', not 'matched memory'.
    """

    def __init__(self, vocab_size: int, hidden: int, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 512):
        super().__init__()
        self.hidden = hidden
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, activation="gelu", norm_first=True,
            dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(hidden, vocab_size)

    def initial_state(self, batch: int, device: str | torch.device):
        # Stateless: the "state" is the token history within a chunk.
        return torch.zeros(batch, 0, dtype=torch.long, device=device)

    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        pos = torch.arange(t, device=tokens.device)
        x = self.embed(tokens) + self.pos(pos)[None]
        causal = torch.triu(torch.ones(t, t, device=tokens.device,
                                       dtype=torch.bool), diagonal=1)
        return self.readout(self.encoder(x, mask=causal))

    def step(self, tokens: torch.Tensor, state: torch.Tensor):
        """Autoregressive single token, carrying a growing context window."""
        ctx = torch.cat([state, tokens.unsqueeze(1)], dim=1)[:, -self.max_len:]
        return self._forward(ctx)[:, -1], ctx

    def forward_chunk(self, tokens: torch.Tensor, targets: torch.Tensor,
                      state: torch.Tensor):
        logits = self._forward(tokens)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, state, None


def transformer_param_count(vocab_size: int, hidden: int, n_layers: int = 2,
                            max_len: int = 512) -> int:
    per_layer = 12 * hidden * hidden + 13 * hidden
    return (vocab_size * hidden + max_len * hidden + n_layers * per_layer
            + hidden * vocab_size + vocab_size)


def match_transformer_hidden(vocab_size: int, target_params: int,
                             n_layers: int = 2, max_len: int = 512,
                             n_heads: int = 4) -> int:
    best, best_diff = n_heads, None
    for h in range(n_heads, 1025, n_heads):  # divisible by n_heads
        diff = abs(transformer_param_count(vocab_size, h, n_layers, max_len)
                   - target_params)
        if best_diff is None or diff < best_diff:
            best, best_diff = h, diff
    return best


def gru_param_count(vocab_size: int, hidden: int) -> int:
    """Closed form (grok instantiated 125 full models to count, debt #5):
    embed V·h + GRU 6h²+6h + readout h·V+V."""
    return 6 * hidden * hidden + (2 * vocab_size + 6) * hidden + vocab_size


def match_hidden(vocab_size: int, target_params: int) -> int:
    best, best_diff = 8, None
    for h in range(8, 2049):
        diff = abs(gru_param_count(vocab_size, h) - target_params)
        if best_diff is None or diff < best_diff:
            best, best_diff = h, diff
    return best
