"""Parameter-matched recurrent control for continuous character prediction."""

from __future__ import annotations

import math

import torch
from torch import nn


class CharacterGRU(nn.Module):
    """A conventional GRU trained on the same uninterrupted character lanes as SOL."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.recurrent = nn.GRU(
            hidden_size, hidden_size, batch_first=True
        )
        self.readout = nn.Linear(hidden_size, vocab_size)

    def initial_state(
        self, batch_size: int, device: torch.device | str
    ) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(
        self, tokens: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens)
        output, state = self.recurrent(embedded, state)
        return self.readout(output), state

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def match_gru_hidden_size(
    vocab_size: int,
    target_parameters: int,
    *,
    minimum: int = 8,
    maximum: int = 2048,
) -> int:
    """Find the GRU width with the closest parameter count to a target model."""

    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    # A GRU is approximately 6h² parameters, so search tightly around that estimate.
    estimate = max(minimum, min(maximum, round(math.sqrt(target_parameters / 6))))
    lower = max(minimum, estimate - 96)
    upper = min(maximum, estimate + 96)
    candidates = range(lower, upper + 1)
    return min(
        candidates,
        key=lambda hidden: abs(
            CharacterGRU(vocab_size, hidden).parameter_count() - target_parameters
        ),
    )


@torch.no_grad()
def evaluate_gru(
    model: CharacterGRU,
    encoded_text: torch.Tensor,
    *,
    tokens: int,
    warmup: int,
) -> dict[str, float | int]:
    """Score a contiguous held-out stream after warming recurrent state."""

    device = next(model.parameters()).device
    encoded_text = encoded_text.to(device)
    available = encoded_text.numel() - 1
    scored = min(tokens, max(1, available - min(warmup, available - 1)))
    warmup = min(warmup, available - scored)
    state = model.initial_state(1, device)
    total_loss = 0.0
    correct = 0
    was_training = model.training
    model.eval()
    for index in range(warmup + scored):
        logits, state = model(encoded_text[index : index + 1].view(1, 1), state)
        if index >= warmup:
            target = encoded_text[index + 1 : index + 2]
            token_logits = logits[:, -1]
            total_loss += float(
                nn.functional.cross_entropy(token_logits, target).item()
            )
            correct += int(token_logits.argmax(dim=-1).item() == target.item())
    model.train(was_training)
    nll = total_loss / scored
    return {
        "nll": nll,
        "bits_per_character": nll / math.log(2.0),
        "perplexity": math.exp(min(20.0, nll)),
        "accuracy": correct / scored,
        "tokens": scored,
    }
