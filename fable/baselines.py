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
