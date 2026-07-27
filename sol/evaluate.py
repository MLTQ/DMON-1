"""Held-out language and organism-state evaluation for SOL checkpoints."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch
from torch.nn import functional as F

from .model import FieldState, SparseAxonField
from .structure import next_probe_sources
from .stream import CharacterVocabulary


@dataclass(frozen=True)
class EvaluationMetrics:
    nll: float
    bits_per_character: float
    perplexity: float
    accuracy: float
    tokens: int
    mean_energy: float
    mean_viability: float
    quiescent_fraction: float
    energy_input: float
    energy_spent: float
    energy_transport_drift: float
    mean_backward_credit: float
    mean_output_error_credit: float
    mean_novelty: float
    mean_edge_flow: float
    mean_fast_weight: float
    mean_edge_eligibility: float
    fast_weight_saturation: float
    mean_probe_eligibility: float
    mean_probe_flow: float
    total_rewires: int

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
        backward_credit=state.backward_credit[:, permutation],
        output_error_credit=state.output_error_credit[:, permutation],
        edge_eligibility=state.edge_eligibility[:, permutation],
        probe_eligibility=state.probe_eligibility[:, permutation],
        fast_weight=state.fast_weight[:, permutation],
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
    score_start: int | None = None,
    reset_each_token: bool = False,
    shuffle_cells: bool = False,
    zero_fast_efficacy: bool = False,
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
    if score_start is None:
        scored_tokens = min(
            tokens, max(1, available - min(warmup, available - 1))
        )
        warmup = min(warmup, available - scored_tokens)
        score_start = warmup
    else:
        if not 0 <= score_start < available:
            raise ValueError("score_start must identify a predictable character")
        scored_tokens = min(tokens, available - score_start)
        warmup = min(warmup, score_start)
    warm_start = score_start - warmup
    segment = encoded[
        warm_start : score_start + scored_tokens + 1
    ]
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
    viabilities: list[float] = []
    quiescent_fractions: list[float] = []
    energy_inputs: list[float] = []
    energy_costs: list[float] = []
    energy_drifts: list[float] = []
    backward_credits: list[float] = []
    output_error_credits: list[float] = []
    novelties: list[float] = []
    edge_flows: list[float] = []
    fast_weights: list[float] = []
    edge_eligibilities: list[float] = []
    fast_saturations: list[float] = []
    probe_eligibilities: list[float] = []
    probe_flows: list[float] = []

    for index in range(warmup + scored_tokens):
        if reset_each_token:
            state = model.initial_state(1, device)
        if zero_fast_efficacy:
            state = replace(
                state, fast_weight=torch.zeros_like(state.fast_weight)
            )
        token = segment[index : index + 1]
        target = segment[index + 1 : index + 2]
        logits, state, diagnostics = model.tick(
            state,
            token,
            allow_fast_plasticity=not zero_fast_efficacy,
        )
        surprise = F.cross_entropy(logits, target, reduction="none")
        state = model.observe_prediction(state, logits, target)
        if shuffle_cells:
            state = _shuffle_cell_state(state, permutation)

        if index >= warmup:
            total_loss += float(surprise.item())
            correct += int(logits.argmax(dim=-1).item() == target.item())
            energies.append(float(diagnostics["mean_energy"].mean().item()))
            viabilities.append(
                float(diagnostics["mean_viability"].mean().item())
            )
            quiescent_fractions.append(
                float(diagnostics["quiescent_fraction"].mean().item())
            )
            energy_inputs.append(
                float(diagnostics["energy_input"].mean().item())
            )
            energy_costs.append(
                float(diagnostics["energy_spent"].mean().item())
            )
            energy_drifts.append(
                float(
                    diagnostics["energy_transport_drift"].mean().item()
                )
            )
            backward_credits.append(
                float(
                    diagnostics["mean_backward_credit"].mean().item()
                )
            )
            output_error_credits.append(
                float(
                    diagnostics["mean_output_error_credit"].mean().item()
                )
            )
            novelties.append(float(diagnostics["novelty"].mean().item()))
            edge_flows.append(float(diagnostics["edge_flow"].mean().item()))
            fast_weights.append(
                float(diagnostics["mean_fast_weight"].mean().item())
            )
            edge_eligibilities.append(
                float(
                    diagnostics["mean_edge_eligibility"].mean().item()
                )
            )
            fast_saturations.append(
                float(
                    diagnostics["fast_weight_saturation"].mean().item()
                )
            )
            probe_eligibilities.append(
                float(
                    diagnostics["mean_probe_eligibility"].mean().item()
                )
            )
            probe_flows.append(
                float(diagnostics["probe_flow"].mean().item())
            )

    model.train(was_training)
    nll = total_loss / scored_tokens
    return EvaluationMetrics(
        nll=nll,
        bits_per_character=nll / math.log(2.0),
        perplexity=math.exp(min(20.0, nll)),
        accuracy=correct / scored_tokens,
        tokens=scored_tokens,
        mean_energy=sum(energies) / scored_tokens,
        mean_viability=sum(viabilities) / scored_tokens,
        quiescent_fraction=sum(quiescent_fractions) / scored_tokens,
        energy_input=sum(energy_inputs) / scored_tokens,
        energy_spent=sum(energy_costs) / scored_tokens,
        energy_transport_drift=sum(energy_drifts) / scored_tokens,
        mean_backward_credit=(
            sum(backward_credits) / scored_tokens
        ),
        mean_output_error_credit=(
            sum(output_error_credits) / scored_tokens
        ),
        mean_novelty=sum(novelties) / scored_tokens,
        mean_edge_flow=sum(edge_flows) / scored_tokens,
        mean_fast_weight=sum(fast_weights) / scored_tokens,
        mean_edge_eligibility=sum(edge_eligibilities) / scored_tokens,
        fast_weight_saturation=sum(fast_saturations) / scored_tokens,
        mean_probe_eligibility=sum(probe_eligibilities) / scored_tokens,
        mean_probe_flow=sum(probe_flows) / scored_tokens,
        total_rewires=int(model.total_rewires.item()),
    )


def evaluate_state_ablations(
    model: SparseAxonField,
    vocabulary: CharacterVocabulary,
    text: str,
    *,
    device: torch.device | str = "cpu",
    tokens: int = 2048,
    warmup: int = 256,
    score_start: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """Compare intact persistence against reset and cell-identity disruption."""

    policies = {
        "persistent": {},
        "reset_each_token": {"reset_each_token": True},
        "shuffled_cells": {"shuffle_cells": True},
        "zero_fast_efficacy": {"zero_fast_efficacy": True},
    }
    results = {
        name: evaluate_sol(
            model,
            vocabulary,
            text,
            device=device,
            tokens=tokens,
            warmup=warmup,
            score_start=score_start,
            **options,
        ).to_dict()
        for name, options in policies.items()
    }
    sources = model.sources.clone()
    active_edges = model.active_edges.clone()
    probes = model.probe_sources.clone()
    birth_sources = model.birth_sources()
    birth_active_edges = model.birth_active_edges()
    birth_probes = probes.clone()
    fallback_probes = next_probe_sources(
        birth_sources,
        probes,
        birth_active_edges,
    )
    for target in range(model.cfg.cells):
        active_sources = birth_sources[target][
            birth_active_edges[target]
        ].tolist()
        if int(birth_probes[target]) in active_sources:
            birth_probes[target] = fallback_probes[target]
    try:
        model.sources.copy_(birth_sources)
        model.active_edges.copy_(birth_active_edges)
        model.probe_sources.copy_(birth_probes)
        results["birth_topology"] = evaluate_sol(
            model,
            vocabulary,
            text,
            device=device,
            tokens=tokens,
            warmup=warmup,
            score_start=score_start,
        ).to_dict()
    finally:
        model.sources.copy_(sources)
        model.active_edges.copy_(active_edges)
        model.probe_sources.copy_(probes)
    return results


def evaluate_warmup_sweep(
    model: SparseAxonField,
    vocabulary: CharacterVocabulary,
    text: str,
    warmups: list[int],
    *,
    device: torch.device | str = "cpu",
    tokens: int = 2048,
    score_start: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """Score one fixed held-out window after different state warmup lengths."""

    if not warmups:
        raise ValueError("at least one warmup is required")
    if any(warmup < 0 for warmup in warmups):
        raise ValueError("warmups must be non-negative")
    if len(set(warmups)) != len(warmups):
        raise ValueError("warmups must be unique")
    if score_start is None:
        score_start = max(warmups)
    if score_start < max(warmups):
        raise ValueError("score_start must accommodate every warmup")
    return {
        str(warmup): evaluate_sol(
            model,
            vocabulary,
            text,
            device=device,
            tokens=tokens,
            warmup=warmup,
            score_start=score_start,
        ).to_dict()
        for warmup in warmups
    }
