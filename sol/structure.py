"""Bounded structural plasticity between differentiable SOL stream windows."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

from .topology import analyze_topology

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from .model import FieldState, SparseAxonField


@dataclass(frozen=True)
class StructuralConfig:
    """Non-differentiable grow/prune policy for fixed-fan-in dendrite slots."""

    enabled: bool = False
    allow_rewiring: bool = True
    interval: int = 100
    warmup_updates: int = 500
    replacements_per_phase: int = 1
    confirmation_phases: int = 1
    credit_decay: float = 0.99
    credit_margin: float = 1e-3
    min_edge_age: int = 250
    growth_cost: float = 0.01
    min_endpoint_energy: float = 0.05

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("structural interval must be positive")
        if self.warmup_updates < 0:
            raise ValueError("structural warmup must be non-negative")
        if self.replacements_per_phase < 1:
            raise ValueError("replacements_per_phase must be positive")
        if self.confirmation_phases < 1:
            raise ValueError("confirmation_phases must be positive")
        if not 0 <= self.credit_decay < 1:
            raise ValueError("structural credit decay must be in [0, 1)")
        if self.credit_margin < 0:
            raise ValueError("structural credit margin must be non-negative")
        if self.min_edge_age < 0:
            raise ValueError("minimum edge age must be non-negative")
        if self.growth_cost < 0:
            raise ValueError("growth cost must be non-negative")
        if not 0 <= self.min_endpoint_energy <= 1:
            raise ValueError("minimum endpoint energy must be in [0, 1]")


@dataclass(frozen=True)
class StructuralUpdate:
    """Telemetry from one possible structural phase."""

    rewired_edges: int
    total_rewires: int
    candidate_advantage: float
    rejected_for_reachability: int


def next_probe_sources(
    sources: torch.Tensor,
    current: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose the next deterministic non-edge candidate for every target."""

    if sources.ndim != 2:
        raise ValueError("sources must have shape (cells, dendrites)")
    cells, dendrites = sources.shape
    if dendrites >= cells:
        raise ValueError("structural probes require at least one non-edge source")
    device = sources.device
    if current is None:
        current = torch.arange(cells, device=device)
    if current.shape != (cells,):
        raise ValueError("current probe sources must have shape (cells,)")

    rows = sources.detach().cpu().tolist()
    starts = current.detach().cpu().tolist()
    candidates: list[int] = []
    for target, row in enumerate(rows):
        occupied = {int(source) for source in row}
        start = (int(starts[target]) + 1) % cells
        for offset in range(cells):
            candidate = (start + offset) % cells
            if candidate not in occupied:
                candidates.append(candidate)
                break
        else:  # pragma: no cover - guarded by dendrites < cells
            raise RuntimeError("no structural probe candidate remained")
    return torch.tensor(candidates, dtype=torch.long, device=device)


@torch.no_grad()
def accumulate_structural_credit(
    model: "SparseAxonField",
    edge_evidence: torch.Tensor,
    probe_evidence: torch.Tensor,
    config: StructuralConfig,
) -> None:
    """Retain reward-addressed edge and candidate evidence across windows."""

    if not config.enabled:
        return
    if edge_evidence.shape != model.structural_edge_credit.shape:
        raise ValueError("edge structural evidence has the wrong shape")
    if probe_evidence.shape != model.structural_probe_credit.shape:
        raise ValueError("probe structural evidence has the wrong shape")
    keep = config.credit_decay
    model.structural_edge_credit.mul_(keep).add_(
        edge_evidence.detach().to(model.structural_edge_credit), alpha=1 - keep
    )
    model.structural_probe_credit.mul_(keep).add_(
        probe_evidence.detach().to(model.structural_probe_credit), alpha=1 - keep
    )
    model.structural_edge_age.add_(1)


def _reset_optimizer_slot(
    optimizer: "Optimizer",
    parameter: torch.nn.Parameter,
    target: int,
    slot: int,
) -> None:
    state = optimizer.state.get(parameter)
    if not state:
        return
    for value in state.values():
        if torch.is_tensor(value) and value.shape == parameter.shape:
            value[target, slot] = 0


def _topology_remains_viable(
    model: "SparseAxonField",
    target: int,
    slot: int,
    candidate: int,
) -> bool:
    proposed = model.sources.clone()
    proposed[target, slot] = candidate
    metrics = analyze_topology(
        proposed,
        model.sensory_indices,
        model.output_indices,
    )
    return (
        metrics.reachable_fraction == 1.0
        and metrics.output_reachable_fraction == 1.0
    )


def _endpoints_can_pay(
    state: "FieldState",
    target: int,
    source: int,
    config: StructuralConfig,
) -> bool:
    threshold = config.min_endpoint_energy + config.growth_cost / 2
    return (
        float(state.energy[:, target].mean().item()) >= threshold
        and float(state.energy[:, source].mean().item()) >= threshold
    )


def _charge_growth(
    state: "FieldState",
    target: int,
    source: int,
    cost: float,
) -> None:
    if cost == 0:
        return
    if target == source:
        state.energy[:, target].sub_(cost).clamp_(min=0)
        return
    half = cost / 2
    state.energy[:, target].sub_(half).clamp_(min=0)
    state.energy[:, source].sub_(half).clamp_(min=0)


def _probe_equivalent_weight(model: "SparseAxonField") -> float:
    """Approximate the probe coefficient under uniform dendrite attention."""

    signed_weight = min(
        0.95,
        model.cfg.structural_probe_gain * model.cfg.dendrites,
    )
    return math.atanh(signed_weight)


@torch.no_grad()
def apply_structural_phase(
    model: "SparseAxonField",
    state: "FieldState",
    optimizer: "Optimizer",
    config: StructuralConfig,
    update: int,
) -> StructuralUpdate:
    """Replace mature low-credit edges with better causal probes when due."""

    total = int(model.total_rewires.item())
    if (
        not config.enabled
        or update < config.warmup_updates
        or update % config.interval != 0
    ):
        return StructuralUpdate(0, total, 0.0, 0)

    proposals: list[tuple[float, int, int, int, float]] = []
    observed_advantages: list[float] = []
    for target in range(model.cfg.cells):
        mature = model.structural_edge_age[target] >= config.min_edge_age
        if not bool(mature.any()):
            model.structural_probe_confirmations[target] = 0
            continue
        eligible_slots = torch.nonzero(mature, as_tuple=False).flatten()
        eligible_credit = model.structural_edge_credit[target, eligible_slots]
        position = int(torch.argmin(eligible_credit).item())
        slot = int(eligible_slots[position].item())
        edge_credit = float(model.structural_edge_credit[target, slot].item())
        probe_credit = float(model.structural_probe_credit[target].item())
        advantage = probe_credit - edge_credit
        observed_advantages.append(advantage)
        if probe_credit > 0 and advantage > config.credit_margin:
            model.structural_probe_confirmations[target].add_(1)
        else:
            model.structural_probe_confirmations[target] = 0

        if (
            config.allow_rewiring
            and int(model.structural_probe_confirmations[target].item())
            >= config.confirmation_phases
        ):
            candidate = int(model.probe_sources[target].item())
            proposals.append(
                (advantage, target, slot, candidate, probe_credit)
            )

    proposals.sort(key=lambda value: (-value[0], value[1], value[2]))
    best_advantage = (
        max(observed_advantages) if observed_advantages else 0.0
    )
    first_phase_update = max(
        config.interval,
        (
            (config.warmup_updates + config.interval - 1)
            // config.interval
            * config.interval
        ),
    )
    phase_number = (
        (update - first_phase_update) // config.interval + 1
    )
    decision_due = phase_number % config.confirmation_phases == 0
    if not decision_due:
        return StructuralUpdate(0, total, best_advantage, 0)

    rewired = 0
    rejected_for_reachability = 0
    for advantage, target, slot, candidate, probe_credit in proposals:
        if rewired >= config.replacements_per_phase:
            break
        if advantage <= config.credit_margin:
            break
        if candidate in model.sources[target].tolist():
            continue
        if not _endpoints_can_pay(state, target, candidate, config):
            continue
        if not _topology_remains_viable(
            model, target, slot, candidate
        ):
            rejected_for_reachability += 1
            continue

        model.sources[target, slot] = candidate
        model.edge_weight[target, slot] = _probe_equivalent_weight(model)
        model.edge_bias[target, slot] = 0
        _reset_optimizer_slot(
            optimizer, model.edge_weight, target, slot
        )
        _reset_optimizer_slot(
            optimizer, model.edge_bias, target, slot
        )
        state.edge_eligibility[:, target, slot] = 0
        state.fast_weight[:, target, slot] = 0
        model.structural_edge_credit[target, slot] = probe_credit
        model.structural_edge_age[target, slot] = 0
        _charge_growth(state, target, candidate, config.growth_cost)
        rewired += 1

    if rewired:
        model.total_rewires.add_(rewired)

    model.probe_sources.copy_(
        next_probe_sources(model.sources, model.probe_sources)
    )
    model.structural_probe_credit.zero_()
    model.structural_probe_confirmations.zero_()
    state.probe_eligibility.zero_()
    return StructuralUpdate(
        rewired_edges=rewired,
        total_rewires=int(model.total_rewires.item()),
        candidate_advantage=best_advantage,
        rejected_for_reachability=rejected_for_reachability,
    )


def structural_summary(
    model: "SparseAxonField",
    config: StructuralConfig,
) -> dict[str, Any]:
    """Return checkpoint-friendly final structural telemetry."""

    return {
        "config": asdict(config),
        "total_rewires": int(model.total_rewires.item()),
        "mean_edge_credit": float(
            model.structural_edge_credit.mean().item()
        ),
        "mean_probe_credit": float(
            model.structural_probe_credit.mean().item()
        ),
        "max_probe_confirmations": int(
            model.structural_probe_confirmations.max().item()
        ),
        "mean_edge_age": float(model.structural_edge_age.float().mean().item()),
        "probe_sources": model.probe_sources.detach().cpu().tolist(),
    }
