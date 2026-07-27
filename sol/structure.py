"""Bounded structural plasticity between differentiable SOL stream windows."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .routing_randomization import (
    crossover_schedule,
    deterministic_crossover_assignment_code,
    exact_one_sided_randomization_test,
)
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
    require_global_fitness: bool = False
    global_fitness_margin: float = 0.0
    credit_decay: float = 0.99
    credit_margin: float = 1e-3
    min_edge_age: int = 250
    growth_cost: float = 0.01
    min_endpoint_energy: float = 0.05
    probation_updates: int = 0
    probation_margin: float = 0.0
    probation_baseline_decay: float = 0.99
    probation_exploratory_traffic: bool = False
    probation_randomized_traffic: bool = False
    probation_randomization_alpha: float = 0.10
    variable_fan_in: bool = False
    min_active_dendrites: int = 1
    prune_usage_threshold: float = 0.0
    prune_credit_threshold: float = 0.0
    usage_gain: float = 1.0
    vector_credit_gain: float = 1.0
    locality_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("structural interval must be positive")
        if self.warmup_updates < 0:
            raise ValueError("structural warmup must be non-negative")
        if self.replacements_per_phase < 1:
            raise ValueError("replacements_per_phase must be positive")
        if self.confirmation_phases < 1:
            raise ValueError("confirmation_phases must be positive")
        if self.global_fitness_margin < 0:
            raise ValueError("global_fitness_margin must be non-negative")
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
        if self.probation_updates < 0:
            raise ValueError("probation_updates must be non-negative")
        if self.min_active_dendrites < 1:
            raise ValueError("min_active_dendrites must be positive")
        if self.prune_usage_threshold < 0:
            raise ValueError("prune_usage_threshold must be non-negative")
        if (
            self.usage_gain < 0
            or self.vector_credit_gain < 0
            or self.locality_gain < 0
        ):
            raise ValueError(
                "structural evidence gains must be non-negative"
            )
        if not 0 <= self.probation_baseline_decay < 1:
            raise ValueError(
                "probation_baseline_decay must be in [0, 1)"
            )
        if (
            self.probation_exploratory_traffic
            and (
                self.probation_updates < 2
                or self.probation_updates % 2 != 0
            )
        ):
            raise ValueError(
                "exploratory traffic probation requires an even number "
                "of at least two updates"
            )
        if (
            self.probation_randomized_traffic
            and not self.probation_exploratory_traffic
        ):
            raise ValueError(
                "randomized structural traffic requires exploratory traffic"
            )
        if self.probation_randomized_traffic and (
            self.probation_updates < 4
            or self.probation_updates % 4 != 0
            or self.probation_updates > 64
        ):
            raise ValueError(
                "randomized structural probation requires 4 to 64 "
                "updates divisible by four"
            )
        if not 0 < self.probation_randomization_alpha <= 1:
            raise ValueError(
                "structural probation randomization alpha must be in (0, 1]"
            )


@dataclass(frozen=True)
class StructuralUpdate:
    """Telemetry from one possible structural phase."""

    rewired_edges: int
    total_rewires: int
    candidate_advantage: float
    rejected_for_reachability: int
    probation_started: bool = False
    probation_active: bool = False
    probation_committed: int = 0
    probation_rolled_back: int = 0
    probation_rejected: int = 0
    probation_advantage: float = 0.0
    probation_candidate_exposed: bool = False
    spawned_edges: int = 0
    pruned_edges: int = 0


@dataclass
class StructuralProbation:
    """Checkpointable backup and observed fitness for one provisional graft."""

    active: bool = False
    virtual: bool = False
    exploratory_traffic: bool = False
    randomized_traffic: bool = False
    target: int = -1
    slot: int = -1
    old_source: int = -1
    old_active: bool = True
    candidate_source: int = -1
    started_update: int = 0
    advantage_sum: float = 0.0
    baseline_advantage: float = 0.0
    observations: int = 0
    total_started: int = 0
    total_committed: int = 0
    total_rolled_back: int = 0
    total_rejected: int = 0
    candidate_reward_sum: float = 0.0
    incumbent_reward_sum: float = 0.0
    candidate_observations: int = 0
    incumbent_observations: int = 0
    assignment_code: int = 0
    candidate_schedule: list[bool] = field(default_factory=list)
    reward_observations: list[float] = field(default_factory=list)
    randomization_p_value: float = 1.0
    randomization_extreme_assignments: int = 0
    randomization_total_assignments: int = 0
    candidate_edge_credit: float = 0.0
    candidate_vector_credit: float = 0.0
    last_mean_advantage: float = 0.0
    last_randomization_p_value: float = 1.0
    last_randomization_extreme_assignments: int = 0
    last_randomization_total_assignments: int = 0
    edge_weight: torch.Tensor | None = None
    edge_bias: torch.Tensor | None = None
    edge_credit: torch.Tensor | None = None
    edge_vector_credit: torch.Tensor | None = None
    edge_usage: torch.Tensor | None = None
    edge_age: torch.Tensor | None = None
    edge_eligibility: torch.Tensor | None = None
    fast_weight: torch.Tensor | None = None
    credit_routing_eligibility: torch.Tensor | None = None
    credit_routing_preference: torch.Tensor | None = None
    optimizer_slots: dict[str, dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    trial_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean_advantage(self) -> float:
        if not self.active and self.observations > 0:
            return self.last_mean_advantage
        if self.exploratory_traffic:
            if (
                self.candidate_observations == 0
                or self.incumbent_observations == 0
            ):
                return 0.0
            return (
                self.candidate_reward_sum / self.candidate_observations
                - self.incumbent_reward_sum / self.incumbent_observations
            )
        return self.advantage_sum / max(1, self.observations)

    @property
    def candidate_exposed(self) -> bool:
        """Return the assigned probe arm for the next observation."""

        if not self.active or not self.exploratory_traffic:
            return False
        if self.randomized_traffic:
            if self.observations >= len(self.candidate_schedule):
                raise RuntimeError(
                    "structural traffic schedule is exhausted"
                )
            return self.candidate_schedule[self.observations]
        return self.observations % 4 in (0, 3)

    def observe(
        self,
        advantage: float,
        candidate_exposed: bool | None = None,
    ) -> None:
        if self.active:
            if self.exploratory_traffic:
                if candidate_exposed is None:
                    raise ValueError(
                        "exploratory traffic observations require an arm"
                    )
                if (
                    self.randomized_traffic
                    and candidate_exposed != self.candidate_exposed
                ):
                    raise RuntimeError(
                        "observed structural arm differs from "
                        "checkpointed schedule"
                    )
                if self.randomized_traffic:
                    self.reward_observations.append(float(advantage))
                if candidate_exposed:
                    self.candidate_reward_sum += float(advantage)
                    self.candidate_observations += 1
                else:
                    self.incumbent_reward_sum += float(advantage)
                    self.incumbent_observations += 1
            else:
                self.advantage_sum += (
                    float(advantage) - self.baseline_advantage
                )
            self.observations += 1
            if (
                self.randomized_traffic
                and self.observations % 4 == 0
            ):
                result = exact_one_sided_randomization_test(
                    self.reward_observations,
                    self.candidate_schedule[: self.observations],
                )
                self.randomization_p_value = result.p_value
                self.randomization_extreme_assignments = (
                    result.extreme_assignments
                )
                self.randomization_total_assignments = (
                    result.total_assignments
                )

    def exploratory_probe_mask(
        self,
        model: "SparseAxonField",
    ) -> tuple[torch.Tensor | None, bool]:
        """Gate one candidate probe with its checkpointed traffic schedule."""

        if not self.active or not self.exploratory_traffic:
            return None, False
        candidate_exposed = self.candidate_exposed
        mask = torch.ones(
            model.cfg.cells,
            device=model.sources.device,
            dtype=model.edge_weight.dtype,
        )
        if not candidate_exposed:
            mask[self.target] = 0
        return mask, candidate_exposed

    def state_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "edge_weight",
            "edge_bias",
            "edge_credit",
            "edge_vector_credit",
            "edge_usage",
            "edge_age",
            "edge_eligibility",
            "fast_weight",
            "credit_routing_eligibility",
            "credit_routing_preference",
        ):
            value = getattr(self, name)
            payload[name] = (
                None if value is None else value.detach().cpu()
            )
        payload["optimizer_slots"] = {
            parameter: {
                name: value.detach().cpu()
                for name, value in slots.items()
            }
            for parameter, slots in self.optimizer_slots.items()
        }
        return payload

    @classmethod
    def from_state_dict(
        cls,
        payload: dict[str, Any] | None,
        device: torch.device | str,
    ) -> "StructuralProbation":
        if not payload:
            return cls()
        restored = dict(payload)
        restored.setdefault("randomized_traffic", False)
        restored.setdefault("assignment_code", 0)
        restored.setdefault("candidate_schedule", [])
        restored.setdefault("reward_observations", [])
        restored.setdefault("randomization_p_value", 1.0)
        restored.setdefault("randomization_extreme_assignments", 0)
        restored.setdefault("randomization_total_assignments", 0)
        restored.setdefault("last_randomization_p_value", 1.0)
        restored.setdefault(
            "last_randomization_extreme_assignments",
            0,
        )
        restored.setdefault(
            "last_randomization_total_assignments",
            0,
        )
        device = torch.device(device)
        for name in (
            "edge_weight",
            "edge_bias",
            "edge_credit",
            "edge_vector_credit",
            "edge_usage",
            "edge_age",
            "edge_eligibility",
            "fast_weight",
            "credit_routing_eligibility",
            "credit_routing_preference",
        ):
            value = restored.get(name)
            restored[name] = None if value is None else value.to(device)
        restored["optimizer_slots"] = {
            parameter: {
                name: value.to(device)
                for name, value in slots.items()
            }
            for parameter, slots in restored.get(
                "optimizer_slots", {}
            ).items()
        }
        return cls(**restored)

    def begin(
        self,
        model: "SparseAxonField",
        state: "FieldState",
        optimizer: "Optimizer",
        target: int,
        slot: int,
        candidate: int,
        update: int,
        virtual: bool,
        exploratory_traffic: bool = False,
        randomized_traffic: bool = False,
        trial_updates: int = 0,
        candidate_edge_credit: float = 0.0,
        candidate_vector_credit: float = 0.0,
    ) -> None:
        if self.active:
            raise RuntimeError("structural probation is already active")
        if randomized_traffic and not exploratory_traffic:
            raise ValueError(
                "randomized structural traffic requires exploratory traffic"
            )
        self.active = True
        self.virtual = virtual
        self.exploratory_traffic = exploratory_traffic
        self.randomized_traffic = randomized_traffic
        self.target = target
        self.slot = slot
        self.old_source = int(model.sources[target, slot].item())
        self.old_active = bool(model.active_edges[target, slot].item())
        self.candidate_source = candidate
        self.started_update = update
        self.advantage_sum = 0.0
        self.baseline_advantage = 0.0
        self.observations = 0
        self.candidate_reward_sum = 0.0
        self.incumbent_reward_sum = 0.0
        self.candidate_observations = 0
        self.incumbent_observations = 0
        if randomized_traffic:
            self.assignment_code = (
                deterministic_crossover_assignment_code(
                    topology_seed=model.cfg.topology_seed,
                    identity=(
                        self.total_started,
                        target,
                        slot,
                        candidate,
                        update,
                    ),
                    trial_updates=trial_updates,
                )
            )
            self.candidate_schedule = list(
                crossover_schedule(
                    self.assignment_code,
                    trial_updates,
                )
            )
        else:
            self.assignment_code = 0
            self.candidate_schedule = []
        self.reward_observations = []
        self.randomization_p_value = 1.0
        self.randomization_extreme_assignments = 0
        self.randomization_total_assignments = 0
        self.candidate_edge_credit = float(candidate_edge_credit)
        self.candidate_vector_credit = float(candidate_vector_credit)
        self.last_mean_advantage = 0.0
        self.total_started += 1
        if virtual:
            return
        self.edge_weight = model.edge_weight[target, slot].detach().clone()
        self.edge_bias = model.edge_bias[target, slot].detach().clone()
        self.edge_credit = (
            model.structural_edge_credit[target, slot].detach().clone()
        )
        self.edge_vector_credit = (
            model.structural_edge_vector_credit[target, slot]
            .detach()
            .clone()
        )
        self.edge_usage = (
            model.structural_edge_usage[target, slot].detach().clone()
        )
        self.edge_age = (
            model.structural_edge_age[target, slot].detach().clone()
        )
        self.edge_eligibility = (
            state.edge_eligibility[:, target, slot].detach().clone()
        )
        self.fast_weight = (
            state.fast_weight[:, target, slot].detach().clone()
        )
        self.credit_routing_eligibility = (
            state.credit_routing_eligibility[:, target, slot]
            .detach()
            .clone()
        )
        self.credit_routing_preference = (
            state.credit_routing_preference[:, target, slot]
            .detach()
            .clone()
        )
        parameters = {
            "edge_weight": model.edge_weight,
            "edge_bias": model.edge_bias,
        }
        self.optimizer_slots = {}
        for parameter_name, parameter in parameters.items():
            slots: dict[str, torch.Tensor] = {}
            for name, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    slots[name] = value[target, slot].detach().clone()
            self.optimizer_slots[parameter_name] = slots

    @torch.no_grad()
    def resolve(
        self,
        model: "SparseAxonField",
        state: "FieldState",
        optimizer: "Optimizer",
        margin: float,
        growth_cost: float = 0.0,
        min_endpoint_energy: float = 0.0,
        randomization_alpha: float = 0.10,
        resolved_update: int | None = None,
    ) -> bool:
        """Commit positive probation or restore the exact pre-graft slot."""

        if not self.active:
            raise RuntimeError("no structural probation is active")
        if not 0 < randomization_alpha <= 1:
            raise ValueError(
                "structural probation randomization alpha must be in (0, 1]"
            )
        body_energy_before = float(state.energy.mean().item())
        target_energy_before = float(
            state.energy[:, self.target].mean().item()
        )
        candidate_energy_before = float(
            state.energy[:, self.candidate_source].mean().item()
        )
        decision_advantage = self.mean_advantage
        if self.randomized_traffic:
            if len(self.reward_observations) != len(
                self.candidate_schedule
            ):
                raise RuntimeError(
                    "randomized structural trial requires every reward"
                )
            result = exact_one_sided_randomization_test(
                self.reward_observations,
                self.candidate_schedule,
            )
            randomization_p_value = result.p_value
            randomization_extreme_assignments = (
                result.extreme_assignments
            )
            randomization_total_assignments = (
                result.total_assignments
            )
            clears_randomization = (
                randomization_p_value <= randomization_alpha
            )
        else:
            randomization_p_value = 1.0
            randomization_extreme_assignments = 0
            randomization_total_assignments = 0
            clears_randomization = True
        committed = (
            decision_advantage > margin
            and clears_randomization
            and (
                not self.exploratory_traffic
                or (
                    self.candidate_observations > 0
                    and self.incumbent_observations > 0
                )
            )
        )
        if self.exploratory_traffic and committed:
            threshold = min_endpoint_energy + growth_cost / 2
            committed = (
                float(state.energy[:, self.target].mean().item())
                >= threshold
                and float(
                    state.energy[:, self.candidate_source].mean().item()
                )
                >= threshold
            )
        if self.virtual:
            committed = False
        elif self.exploratory_traffic and committed:
            target, slot = self.target, self.slot
            model.sources[target, slot] = self.candidate_source
            model.active_edges[target, slot] = True
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
            state.credit_routing_eligibility[:, target, slot] = 0
            state.credit_routing_preference[:, target, slot] = 0
            model.structural_edge_credit[target, slot] = (
                self.candidate_edge_credit
            )
            model.structural_edge_vector_credit[target, slot] = (
                self.candidate_vector_credit
            )
            model.structural_edge_age[target, slot] = 0
            _charge_growth(
                state, target, self.candidate_source, growth_cost
            )
            model.total_rewires.add_(1)
            if not self.old_active:
                model.total_spawns.add_(1)
        elif not committed:
            if self.exploratory_traffic:
                self.total_rejected += 1
            else:
                target, slot = self.target, self.slot
                model.sources[target, slot] = self.old_source
                model.active_edges[target, slot] = self.old_active
                model.edge_weight[target, slot] = self.edge_weight
                model.edge_bias[target, slot] = self.edge_bias
                model.structural_edge_credit[target, slot] = self.edge_credit
                model.structural_edge_vector_credit[target, slot] = (
                    self.edge_vector_credit
                )
                model.structural_edge_usage[target, slot] = self.edge_usage
                model.structural_edge_age[target, slot] = self.edge_age
                state.edge_eligibility[:, target, slot] = (
                    self.edge_eligibility
                )
                state.fast_weight[:, target, slot] = self.fast_weight
                state.credit_routing_eligibility[:, target, slot] = (
                    self.credit_routing_eligibility
                )
                state.credit_routing_preference[:, target, slot] = (
                    self.credit_routing_preference
                )
                parameters = {
                    "edge_weight": model.edge_weight,
                    "edge_bias": model.edge_bias,
                }
                for parameter_name, slots in self.optimizer_slots.items():
                    parameter = parameters[parameter_name]
                    optimizer_state = optimizer.state.get(parameter, {})
                    for name, current in optimizer_state.items():
                        if (
                            torch.is_tensor(current)
                            and current.shape == parameter.shape
                        ):
                            if name in slots:
                                current[target, slot] = slots[name]
                            else:
                                current[target, slot] = 0

        if committed:
            self.total_committed += 1
        elif not self.virtual and not self.exploratory_traffic:
            self.total_rolled_back += 1
        outcome = (
            "virtual"
            if self.virtual
            else (
                "committed"
                if committed
                else (
                    "rejected"
                    if self.exploratory_traffic
                    else "rolled_back"
                )
            )
        )
        self.trial_history.append(
            {
                "mode": (
                    (
                        "exploratory_randomized_traffic"
                        if self.randomized_traffic
                        else "exploratory_traffic"
                    )
                    if self.exploratory_traffic
                    else "post_graft"
                ),
                "outcome": outcome,
                "virtual": self.virtual,
                "started_update": self.started_update,
                "resolved_update": (
                    self.started_update + self.observations
                    if resolved_update is None
                    else int(resolved_update)
                ),
                "target": self.target,
                "slot": self.slot,
                "old_source": self.old_source,
                "old_active": self.old_active,
                "candidate_source": self.candidate_source,
                "observations": self.observations,
                "candidate_observations": self.candidate_observations,
                "incumbent_observations": self.incumbent_observations,
                "mean_candidate_reward": (
                    self.candidate_reward_sum
                    / max(1, self.candidate_observations)
                ),
                "mean_incumbent_reward": (
                    self.incumbent_reward_sum
                    / max(1, self.incumbent_observations)
                ),
                "decision_advantage": decision_advantage,
                "decision_margin": float(margin),
                "randomized_traffic": self.randomized_traffic,
                "assignment_code": self.assignment_code,
                "candidate_schedule": list(self.candidate_schedule),
                "reward_observations": list(
                    self.reward_observations
                ),
                "randomization_p_value": (
                    randomization_p_value
                    if self.randomized_traffic
                    else None
                ),
                "randomization_extreme_assignments": (
                    randomization_extreme_assignments
                    if self.randomized_traffic
                    else None
                ),
                "randomization_total_assignments": (
                    randomization_total_assignments
                    if self.randomized_traffic
                    else None
                ),
                "randomization_alpha": (
                    float(randomization_alpha)
                    if self.randomized_traffic
                    else None
                ),
                "candidate_edge_credit": self.candidate_edge_credit,
                "candidate_vector_credit": self.candidate_vector_credit,
                "body_energy_before": body_energy_before,
                "body_energy_after": float(state.energy.mean().item()),
                "target_energy_before": target_energy_before,
                "target_energy_after": float(
                    state.energy[:, self.target].mean().item()
                ),
                "candidate_energy_before": candidate_energy_before,
                "candidate_energy_after": float(
                    state.energy[:, self.candidate_source].mean().item()
                ),
            }
        )
        self.last_mean_advantage = decision_advantage
        self.randomization_p_value = randomization_p_value
        self.randomization_extreme_assignments = (
            randomization_extreme_assignments
        )
        self.randomization_total_assignments = (
            randomization_total_assignments
        )
        self.last_randomization_p_value = randomization_p_value
        self.last_randomization_extreme_assignments = (
            randomization_extreme_assignments
        )
        self.last_randomization_total_assignments = (
            randomization_total_assignments
        )
        self.active = False
        self.virtual = False
        self.edge_weight = None
        self.edge_bias = None
        self.edge_credit = None
        self.edge_vector_credit = None
        self.edge_usage = None
        self.edge_age = None
        self.edge_eligibility = None
        self.fast_weight = None
        self.credit_routing_eligibility = None
        self.credit_routing_preference = None
        self.optimizer_slots = {}
        return committed


def next_probe_sources(
    sources: torch.Tensor,
    current: torch.Tensor | None = None,
    active_edges: torch.Tensor | None = None,
    prefer_local: bool = False,
) -> torch.Tensor:
    """Choose the next deterministic non-edge candidate for every target."""

    if sources.ndim != 2:
        raise ValueError("sources must have shape (cells, dendrites)")
    cells, dendrites = sources.shape
    if active_edges is None:
        active_edges = torch.ones_like(sources, dtype=torch.bool)
    if active_edges.shape != sources.shape:
        raise ValueError("active_edges must match sources")
    if dendrites >= cells:
        raise ValueError("structural probes require at least one non-edge source")
    device = sources.device
    if current is None:
        current = torch.arange(cells, device=device)
    if current.shape != (cells,):
        raise ValueError("current probe sources must have shape (cells,)")

    rows = sources.detach().cpu().tolist()
    active_rows = active_edges.detach().cpu().tolist()
    starts = current.detach().cpu().tolist()
    candidates: list[int] = []
    for target, row in enumerate(rows):
        occupied = {
            int(source)
            for source, active in zip(
                row,
                active_rows[target],
                strict=True,
            )
            if active
        }
        start = (int(starts[target]) + 1) % cells
        if prefer_local:
            ranked = sorted(
                range(cells),
                key=lambda candidate: (
                    _locality_score(target, candidate, cells) * -1,
                    (candidate - start) % cells,
                ),
            )
        else:
            ranked = [
                (start + offset) % cells
                for offset in range(cells)
            ]
        for candidate in ranked:
            if (
                candidate not in occupied
                and (
                    candidate != int(starts[target])
                    or len(occupied) >= cells - 1
                )
            ):
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
    probe_fitness: torch.Tensor,
    config: StructuralConfig,
    edge_usage: torch.Tensor | None = None,
    edge_vector_evidence: torch.Tensor | None = None,
    probe_vector_evidence: torch.Tensor | None = None,
) -> None:
    """Retain reward-addressed edge and candidate evidence across windows."""

    if not config.enabled:
        return
    if edge_evidence.shape != model.structural_edge_credit.shape:
        raise ValueError("edge structural evidence has the wrong shape")
    if probe_evidence.shape != model.structural_probe_credit.shape:
        raise ValueError("probe structural evidence has the wrong shape")
    if probe_fitness.shape != model.structural_probe_fitness.shape:
        raise ValueError("probe fitness evidence has the wrong shape")
    if (
        edge_usage is not None
        and edge_usage.shape != model.structural_edge_usage.shape
    ):
        raise ValueError("edge usage evidence has the wrong shape")
    if (
        edge_vector_evidence is not None
        and edge_vector_evidence.shape
        != model.structural_edge_vector_credit.shape
    ):
        raise ValueError("edge vector-credit evidence has the wrong shape")
    if (
        probe_vector_evidence is not None
        and probe_vector_evidence.shape
        != model.structural_probe_vector_credit.shape
    ):
        raise ValueError("probe vector-credit evidence has the wrong shape")
    keep = config.credit_decay
    model.structural_edge_credit.mul_(keep).add_(
        edge_evidence.detach().to(model.structural_edge_credit), alpha=1 - keep
    )
    model.structural_probe_credit.mul_(keep).add_(
        probe_evidence.detach().to(model.structural_probe_credit), alpha=1 - keep
    )
    model.structural_probe_fitness.mul_(keep).add_(
        probe_fitness.detach().to(model.structural_probe_fitness),
        alpha=1 - keep,
    )
    if edge_usage is not None:
        model.structural_edge_usage.mul_(keep).add_(
            edge_usage.detach().to(model.structural_edge_usage),
            alpha=1 - keep,
        )
        model.structural_edge_usage.mul_(model.active_edges)
    if edge_vector_evidence is not None:
        model.structural_edge_vector_credit.mul_(keep).add_(
            edge_vector_evidence.detach().to(
                model.structural_edge_vector_credit
            ),
            alpha=1 - keep,
        )
        model.structural_edge_vector_credit.mul_(model.active_edges)
    if probe_vector_evidence is not None:
        model.structural_probe_vector_credit.mul_(keep).add_(
            probe_vector_evidence.detach().to(
                model.structural_probe_vector_credit
            ),
            alpha=1 - keep,
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
    candidate: int | None,
) -> bool:
    proposed = model.sources.clone()
    proposed_active = model.active_edges.clone()
    if candidate is None:
        proposed_active[target, slot] = False
    else:
        proposed[target, slot] = candidate
        proposed_active[target, slot] = True
    metrics = analyze_topology(
        proposed,
        model.sensory_indices,
        model.output_indices,
        proposed_active,
    )
    return (
        metrics.reachable_fraction == 1.0
        and metrics.output_reachable_fraction == 1.0
    )


def _reset_edge_slot(
    model: "SparseAxonField",
    state: "FieldState",
    optimizer: "Optimizer",
    target: int,
    slot: int,
) -> None:
    """Clear transient evidence and optimizer momentum for a repurposed slot."""

    _reset_optimizer_slot(optimizer, model.edge_weight, target, slot)
    _reset_optimizer_slot(optimizer, model.edge_bias, target, slot)
    state.edge_eligibility[:, target, slot] = 0
    state.fast_weight[:, target, slot] = 0
    state.credit_routing_eligibility[:, target, slot] = 0
    state.credit_routing_preference[:, target, slot] = 0
    model.structural_edge_credit[target, slot] = 0
    model.structural_edge_vector_credit[target, slot] = 0
    model.structural_edge_usage[target, slot] = 0
    model.structural_edge_age[target, slot] = 0


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


def _locality_score(target: int, source: int, cells: int) -> float:
    """Return a circular proximity prior in [0, 1]."""

    distance = abs(target - source)
    distance = min(distance, cells - distance)
    return 1.0 - distance / max(1.0, cells / 2)


def structural_phase_due(
    config: StructuralConfig,
    update: int,
) -> bool:
    """Return whether this update contributes one structural evidence phase."""

    return (
        config.enabled
        and update >= config.warmup_updates
        and update % config.interval == 0
    )


def structural_decision_due(
    config: StructuralConfig,
    update: int,
) -> bool:
    """Return whether a due phase may begin a structural intervention."""

    if not structural_phase_due(config, update):
        return False
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
    return phase_number % config.confirmation_phases == 0


@torch.no_grad()
def apply_structural_phase(
    model: "SparseAxonField",
    state: "FieldState",
    optimizer: "Optimizer",
    config: StructuralConfig,
    update: int,
    probation: StructuralProbation | None = None,
) -> StructuralUpdate:
    """Replace mature low-credit edges with better causal probes when due."""

    total = int(model.total_rewires.item())
    if probation is not None and probation.active:
        return StructuralUpdate(
            0,
            total,
            0.0,
            0,
            probation_active=True,
            probation_committed=probation.total_committed,
            probation_rolled_back=probation.total_rolled_back,
            probation_rejected=probation.total_rejected,
            probation_advantage=probation.mean_advantage,
        )
    if not structural_phase_due(config, update):
        return StructuralUpdate(0, total, 0.0, 0)

    proposals: list[
        tuple[float, float, int, int, int, float, float]
    ] = []
    observed_advantages: list[float] = []
    for target in range(model.cfg.cells):
        inactive_slots = torch.nonzero(
            ~model.active_edges[target],
            as_tuple=False,
        ).flatten()
        if config.variable_fan_in and inactive_slots.numel() > 0:
            slot = int(inactive_slots[0].item())
            edge_score = 0.0
        else:
            mature = (
                model.structural_edge_age[target] >= config.min_edge_age
            ) & model.active_edges[target]
            if not bool(mature.any()):
                model.structural_probe_confirmations[target] = 0
                continue
            eligible_slots = torch.nonzero(mature, as_tuple=False).flatten()
            eligible_credit = model.structural_edge_credit[
                target, eligible_slots
            ]
            position = int(torch.argmin(eligible_credit).item())
            slot = int(eligible_slots[position].item())
            edge_score = (
                float(model.structural_edge_credit[target, slot].item())
                + config.vector_credit_gain
                * float(
                    model.structural_edge_vector_credit[
                        target, slot
                    ].item()
                )
                + config.usage_gain
                * float(
                    model.structural_edge_usage[target, slot].item()
                )
            )
        probe_credit = float(model.structural_probe_credit[target].item())
        probe_vector_credit = float(
            model.structural_probe_vector_credit[target].item()
        )
        probe_fitness = float(
            model.structural_probe_fitness[target].item()
        )
        candidate = int(model.probe_sources[target].item())
        causal_probe_score = (
            probe_credit
            + config.vector_credit_gain * probe_vector_credit
        )
        advantage = causal_probe_score - edge_score
        priority = (
            advantage
            + config.locality_gain
            * _locality_score(target, candidate, model.cfg.cells)
        )
        observed_advantages.append(advantage)
        globally_fit = (
            not config.require_global_fitness
            or probe_fitness > config.global_fitness_margin
        )
        if (
            causal_probe_score > 0
            and advantage > config.credit_margin
            and globally_fit
        ):
            model.structural_probe_confirmations[target].add_(1)
        else:
            model.structural_probe_confirmations[target] = 0

        if (
            (config.allow_rewiring or config.probation_updates > 0)
            and int(model.structural_probe_confirmations[target].item())
            >= config.confirmation_phases
        ):
            proposals.append(
                (
                    priority,
                    advantage,
                    target,
                    slot,
                    candidate,
                    probe_credit,
                    probe_vector_credit,
                )
            )

    proposals.sort(key=lambda value: (-value[0], -value[1], value[2]))
    best_advantage = (
        max(observed_advantages) if observed_advantages else 0.0
    )
    if not structural_decision_due(config, update):
        return StructuralUpdate(0, total, best_advantage, 0)

    rewired = 0
    spawned = 0
    pruned = 0
    probation_started = False
    rejected_for_reachability = 0
    if config.variable_fan_in and config.allow_rewiring:
        prune_candidates: list[tuple[float, float, int, int]] = []
        for target in range(model.cfg.cells):
            if (
                int(model.active_edges[target].sum().item())
                <= config.min_active_dendrites
            ):
                continue
            mature = (
                model.structural_edge_age[target] >= config.min_edge_age
            ) & model.active_edges[target]
            for slot_tensor in torch.nonzero(
                mature,
                as_tuple=False,
            ).flatten():
                slot = int(slot_tensor.item())
                usage = float(
                    model.structural_edge_usage[target, slot].item()
                )
                credit = float(
                    model.structural_edge_credit[target, slot].item()
                )
                retention = (
                    credit
                    + config.vector_credit_gain
                    * float(
                        model.structural_edge_vector_credit[
                            target, slot
                        ].item()
                    )
                    + config.usage_gain * usage
                )
                if (
                    usage <= config.prune_usage_threshold
                    and retention <= config.prune_credit_threshold
                ):
                    prune_candidates.append(
                        (usage, retention, target, slot)
                    )
        prune_candidates.sort()
        for _, _, target, slot in prune_candidates:
            if pruned >= config.replacements_per_phase:
                break
            if (
                int(model.active_edges[target].sum().item())
                <= config.min_active_dendrites
            ):
                continue
            if not _topology_remains_viable(
                model,
                target,
                slot,
                None,
            ):
                rejected_for_reachability += 1
                continue
            model.active_edges[target, slot] = False
            _reset_edge_slot(
                model,
                state,
                optimizer,
                target,
                slot,
            )
            pruned += 1

    for (
        _,
        advantage,
        target,
        slot,
        candidate,
        probe_credit,
        probe_vector_credit,
    ) in proposals:
        if rewired >= config.replacements_per_phase:
            break
        if advantage <= config.credit_margin:
            break
        if candidate in model.sources[target][
            model.active_edges[target]
        ].tolist():
            continue
        if not _endpoints_can_pay(state, target, candidate, config):
            continue
        if not _topology_remains_viable(
            model, target, slot, candidate
        ):
            rejected_for_reachability += 1
            continue

        if config.probation_updates > 0 and probation is not None:
            probation.begin(
                model,
                state,
                optimizer,
                target,
                slot,
                candidate,
                update,
                virtual=not config.allow_rewiring,
                exploratory_traffic=(
                    config.probation_exploratory_traffic
                ),
                randomized_traffic=(
                    config.probation_randomized_traffic
                ),
                trial_updates=config.probation_updates,
                candidate_edge_credit=probe_credit,
                candidate_vector_credit=probe_vector_credit,
            )
            probation_started = True
        if (
            config.allow_rewiring
            and not config.probation_exploratory_traffic
        ):
            was_active = bool(model.active_edges[target, slot].item())
            model.sources[target, slot] = candidate
            model.active_edges[target, slot] = True
            model.edge_weight[target, slot] = _probe_equivalent_weight(model)
            model.edge_bias[target, slot] = 0
            _reset_edge_slot(
                model,
                state,
                optimizer,
                target,
                slot,
            )
            model.structural_edge_credit[target, slot] = probe_credit
            model.structural_edge_vector_credit[
                target, slot
            ] = probe_vector_credit
            _charge_growth(state, target, candidate, config.growth_cost)
            rewired += 1
            if not was_active:
                spawned += 1
        if probation_started:
            break

    mutations = rewired + pruned
    if mutations:
        model.total_rewires.add_(mutations)
    if spawned:
        model.total_spawns.add_(spawned)
    if pruned:
        model.total_prunes.add_(pruned)

    if (
        probation_started
        and config.probation_exploratory_traffic
    ):
        return StructuralUpdate(
            rewired_edges=0,
            total_rewires=int(model.total_rewires.item()),
            candidate_advantage=best_advantage,
            rejected_for_reachability=rejected_for_reachability,
            probation_started=True,
            probation_active=True,
            probation_committed=probation.total_committed,
            probation_rolled_back=probation.total_rolled_back,
            probation_rejected=probation.total_rejected,
            spawned_edges=spawned,
            pruned_edges=pruned,
        )

    model.probe_sources.copy_(
        next_probe_sources(
            model.sources,
            model.probe_sources,
            model.active_edges,
            prefer_local=config.locality_gain > 0,
        )
    )
    model.structural_probe_credit.zero_()
    model.structural_probe_fitness.zero_()
    model.structural_probe_vector_credit.zero_()
    model.structural_probe_confirmations.zero_()
    state.probe_eligibility.zero_()
    return StructuralUpdate(
        rewired_edges=rewired,
        total_rewires=int(model.total_rewires.item()),
        candidate_advantage=best_advantage,
        rejected_for_reachability=rejected_for_reachability,
        probation_started=probation_started,
        probation_active=bool(probation and probation.active),
        probation_committed=(
            0 if probation is None else probation.total_committed
        ),
        probation_rolled_back=(
            0 if probation is None else probation.total_rolled_back
        ),
        probation_rejected=(
            0 if probation is None else probation.total_rejected
        ),
        spawned_edges=spawned,
        pruned_edges=pruned,
    )


def structural_summary(
    model: "SparseAxonField",
    config: StructuralConfig,
    probation: StructuralProbation | None = None,
) -> dict[str, Any]:
    """Return checkpoint-friendly final structural telemetry."""

    return {
        "config": asdict(config),
        "total_rewires": int(model.total_rewires.item()),
        "total_spawns": int(model.total_spawns.item()),
        "total_prunes": int(model.total_prunes.item()),
        "active_edges": int(model.active_edges.sum().item()),
        "mean_active_dendrites": float(
            model.active_edges.sum(dim=1).float().mean().item()
        ),
        "mean_edge_credit": float(
            model.structural_edge_credit.mean().item()
        ),
        "mean_edge_usage": float(
            model.structural_edge_usage.mean().item()
        ),
        "mean_probe_credit": float(
            model.structural_probe_credit.mean().item()
        ),
        "mean_probe_fitness": float(
            model.structural_probe_fitness.mean().item()
        ),
        "mean_edge_vector_credit": float(
            model.structural_edge_vector_credit.mean().item()
        ),
        "mean_probe_vector_credit": float(
            model.structural_probe_vector_credit.mean().item()
        ),
        "max_probe_confirmations": int(
            model.structural_probe_confirmations.max().item()
        ),
        "mean_edge_age": float(model.structural_edge_age.float().mean().item()),
        "probe_sources": model.probe_sources.detach().cpu().tolist(),
        "probation": (
            {
                "active": probation.active,
                "virtual": probation.virtual,
                "exploratory_traffic": probation.exploratory_traffic,
                "randomized_traffic": probation.randomized_traffic,
                "target": probation.target,
                "slot": probation.slot,
                "candidate_source": probation.candidate_source,
                "started_update": probation.started_update,
                "observations": probation.observations,
                "assignment_code": probation.assignment_code,
                "candidate_schedule": list(
                    probation.candidate_schedule
                ),
                "reward_observations": list(
                    probation.reward_observations
                ),
                "candidate_observations": (
                    probation.candidate_observations
                ),
                "incumbent_observations": (
                    probation.incumbent_observations
                ),
                "mean_candidate_reward": (
                    probation.candidate_reward_sum
                    / max(1, probation.candidate_observations)
                ),
                "mean_incumbent_reward": (
                    probation.incumbent_reward_sum
                    / max(1, probation.incumbent_observations)
                ),
                "baseline_advantage": probation.baseline_advantage,
                "mean_advantage": probation.mean_advantage,
                "randomization_p_value": (
                    probation.randomization_p_value
                    if probation.active
                    else probation.last_randomization_p_value
                ),
                "randomization_extreme_assignments": (
                    probation.randomization_extreme_assignments
                    if probation.active
                    else probation.last_randomization_extreme_assignments
                ),
                "randomization_total_assignments": (
                    probation.randomization_total_assignments
                    if probation.active
                    else probation.last_randomization_total_assignments
                ),
                "total_started": probation.total_started,
                "total_committed": probation.total_committed,
                "total_rolled_back": probation.total_rolled_back,
                "total_rejected": probation.total_rejected,
                "trial_history": [
                    dict(trial) for trial in probation.trial_history
                ],
            }
            if probation is not None
            else {}
        ),
    }
