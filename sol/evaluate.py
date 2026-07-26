"""Held-out language and organism-state evaluation for SOL checkpoints."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch
from torch.nn import functional as F

from .model import FieldState, SparseAxonField
from .stream import CharacterVocabulary


@dataclass(frozen=True)
class EvaluationMetrics:
    nll: float
    bits_per_character: float
    perplexity: float
    accuracy: float
    tokens: int
    mean_energy: float
    mean_novelty: float
    mean_edge_flow: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _shuffle_cell_state(state: FieldState, permutation: torch.Tensor) -> FieldState:
    """Break persistent cell identity while preserving aggregate state statistics."""

    return replace(
        state,
        hidden=state.hidden[:, permutation],
        energy=state.energy[:, permutation],
        stimulation=state.stimulation[:, permutation],
        eligibility=state.eligibility[:, permutation],
    )


@torch.no_grad()
def evaluate_sol(
    model: SparseAxonField,
    vocabulary: CharacterVocabulary,
    text: str,
    *,
    device: torch.device | str = "cpu",
    tokens: int = 2048,
    warmup: int = 256,
    reset_each_token: bool = False,
    shuffle_cells: bool = False,
) -> EvaluationMetrics:
    """Score a contiguous held-out stream under an explicit state policy."""

    if tokens < 1:
        raise ValueError("tokens must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    encoded = vocabulary.encode(text, device)
    available = encoded.numel() - 1
    if available < 1:
        raise ValueError("evaluation text must contain at least two characters")
    scored_tokens = min(tokens, max(1, available - min(warmup, available - 1)))
    warmup = min(warmup, available - scored_tokens)
    device = torch.device(device)
    state = model.initial_state(1, device)
    permutation = torch.randperm(
        model.cfg.cells, generator=torch.Generator().manual_seed(1729)
    ).to(device)

    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    energies: list[float] = []
    novelties: list[float] = []
    edge_flows: list[float] = []

    for index in range(warmup + scored_tokens):
        if reset_each_token:
            state = model.initial_state(1, device)
        token = encoded[index : index + 1]
        target = encoded[index + 1 : index + 2]
        logits, state, diagnostics = model.tick(state, token)
        surprise = F.cross_entropy(logits, target, reduction="none")
        reward = torch.tanh(math.log(model.cfg.vocab_size) - surprise)
        state = replace(state, reward=reward)
        if shuffle_cells:
            state = _shuffle_cell_state(state, permutation)

        if index >= warmup:
            total_loss += float(surprise.item())
            correct += int(logits.argmax(dim=-1).item() == target.item())
            energies.append(float(diagnostics["mean_energy"].mean().item()))
            novelties.append(float(diagnostics["novelty"].mean().item()))
            edge_flows.append(float(diagnostics["edge_flow"].mean().item()))

    model.train(was_training)
    nll = total_loss / scored_tokens
    return EvaluationMetrics(
        nll=nll,
        bits_per_character=nll / math.log(2.0),
        perplexity=math.exp(min(20.0, nll)),
        accuracy=correct / scored_tokens,
        tokens=scored_tokens,
        mean_energy=sum(energies) / scored_tokens,
        mean_novelty=sum(novelties) / scored_tokens,
        mean_edge_flow=sum(edge_flows) / scored_tokens,
    )


def evaluate_state_ablations(
    model: SparseAxonField,
    vocabulary: CharacterVocabulary,
    text: str,
    *,
    device: torch.device | str = "cpu",
    tokens: int = 2048,
    warmup: int = 256,
) -> dict[str, dict[str, float | int]]:
    """Compare intact persistence against reset and cell-identity disruption."""

    policies = {
        "persistent": {},
        "reset_each_token": {"reset_each_token": True},
        "shuffled_cells": {"shuffle_cells": True},
    }
    return {
        name: evaluate_sol(
            model,
            vocabulary,
            text,
            device=device,
            tokens=tokens,
            warmup=warmup,
            **options,
        ).to_dict()
        for name, options in policies.items()
    }
