"""Live counterfactual trials for persistent reverse-credit routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .routing_randomization import (
    crossover_schedule,
    deterministic_assignment_code,
    exact_one_sided_randomization_test,
)

if TYPE_CHECKING:
    from .model import FieldState, SparseAxonField


@dataclass(frozen=True)
class RoutingTrafficConfig:
    """Schedule and bound one live reverse-credit routing intervention."""

    enabled: bool = False
    interval: int = 25
    warmup_updates: int = 75
    trial_updates: int = 20
    boundary_interval: int = 0
    randomization_alpha: float = 0.10
    margin: float = 0.0
    proposal_step: float = 0.05
    minimum_eligibility: float = 1e-6

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("routing traffic interval must be positive")
        if self.warmup_updates < 0:
            raise ValueError(
                "routing traffic warmup must be non-negative"
            )
        if (
            self.trial_updates < 4
            or self.trial_updates % 4 != 0
            or self.trial_updates > 64
        ):
            raise ValueError(
                "routing traffic trial requires 4 to 64 updates "
                "divisible by four"
            )
        if self.boundary_interval < 0:
            raise ValueError(
                "routing traffic boundary interval must be non-negative"
            )
        if not 0 < self.randomization_alpha <= 1:
            raise ValueError(
                "routing traffic randomization alpha must be in (0, 1]"
            )
        if self.margin < 0:
            raise ValueError("routing traffic margin must be non-negative")
        if self.proposal_step <= 0:
            raise ValueError(
                "routing traffic proposal step must be positive"
            )
        if self.minimum_eligibility < 0:
            raise ValueError(
                "routing traffic minimum eligibility must be non-negative"
            )


@dataclass(frozen=True)
class RoutingTrafficUpdate:
    """Telemetry from one possible routing-trial transition."""

    active: bool
    started: bool
    resolved: bool
    committed: bool
    candidate_exposed: bool
    total_started: int
    total_committed: int
    total_rejected: int
    advantage: float
    randomization_p_value: float
    target: int
    slot: int


@dataclass
class RoutingTrafficTrial:
    """Checkpointable randomized evidence for one routing proposal."""

    active: bool = False
    target: int = -1
    slot: int = -1
    direction: int = 0
    proposal_evidence: float = 0.0
    started_update: int = 0
    observations: int = 0
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
    legacy_fixed_schedule: bool = False
    total_started: int = 0
    total_committed: int = 0
    total_rejected: int = 0
    last_mean_advantage: float = 0.0
    last_randomization_p_value: float = 1.0
    last_randomization_extreme_assignments: int = 0
    last_randomization_total_assignments: int = 0
    last_target: int = -1
    last_slot: int = -1
    delta: torch.Tensor | None = None
    trial_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean_advantage(self) -> float:
        if (
            self.candidate_observations == 0
            or self.incumbent_observations == 0
        ):
            return 0.0
        return (
            self.candidate_reward_sum / self.candidate_observations
            - self.incumbent_reward_sum / self.incumbent_observations
        )

    @property
    def candidate_exposed(self) -> bool:
        """Return the checkpointed randomized arm for the next observation."""

        if not self.active:
            return False
        if self.candidate_schedule:
            if self.observations >= len(self.candidate_schedule):
                raise RuntimeError("routing traffic schedule is exhausted")
            return self.candidate_schedule[self.observations]
        return self.observations % 4 in (0, 3)

    def active_delta(self) -> torch.Tensor | None:
        """Return the fixed proposal only during candidate traffic windows."""

        return self.delta if self.candidate_exposed else None

    def observe(
        self,
        advantage: float,
        candidate_exposed: bool,
    ) -> None:
        """Retain one live-body reward under its actual routing arm."""

        if not self.active:
            return
        scheduled_exposure = self.candidate_exposed
        if candidate_exposed != scheduled_exposure:
            raise RuntimeError(
                "observed routing arm differs from checkpointed schedule"
            )
        self.reward_observations.append(float(advantage))
        if candidate_exposed:
            self.candidate_reward_sum += float(advantage)
            self.candidate_observations += 1
        else:
            self.incumbent_reward_sum += float(advantage)
            self.incumbent_observations += 1
        self.observations += 1
        if (
            not self.legacy_fixed_schedule
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

    def state_dict(self) -> dict[str, Any]:
        """Serialize active phase, proposal, evidence, and decision ledger."""

        payload = asdict(self)
        payload["delta"] = (
            None if self.delta is None else self.delta.detach().cpu()
        )
        return payload

    @classmethod
    def from_state_dict(
        cls,
        payload: dict[str, Any] | None,
        device: torch.device | str,
    ) -> "RoutingTrafficTrial":
        """Restore the exact next crossover arm on the requested device."""

        if not payload:
            return cls()
        restored = dict(payload)
        restored.setdefault("assignment_code", 0)
        restored.setdefault("candidate_schedule", [])
        restored.setdefault("reward_observations", [])
        restored.setdefault("randomization_p_value", 1.0)
        restored.setdefault("randomization_extreme_assignments", 0)
        restored.setdefault("randomization_total_assignments", 0)
        restored.setdefault(
            "legacy_fixed_schedule",
            bool(restored.get("active", False)),
        )
        restored.setdefault("last_randomization_p_value", 1.0)
        restored.setdefault(
            "last_randomization_extreme_assignments",
            0,
        )
        restored.setdefault(
            "last_randomization_total_assignments",
            0,
        )
        delta = restored.get("delta")
        restored["delta"] = (
            None
            if delta is None
            else delta.to(torch.device(device))
        )
        return cls(**restored)

    @torch.no_grad()
    def begin(
        self,
        model: "SparseAxonField",
        state: "FieldState",
        config: RoutingTrafficConfig,
        update: int,
    ) -> bool:
        """Nominate a bounded zero-sum branch perturbation from live eligibility."""

        if self.active:
            raise RuntimeError("routing traffic trial is already active")
        active = model.active_edges
        eligible_targets = active.sum(dim=1) >= 2
        evidence = (
            state.credit_routing_eligibility.mean(dim=0)
            * active
        )
        score = evidence.abs()
        score[~eligible_targets] = -1
        if self.last_target >= 0 and bool(
            eligible_targets[self.last_target]
        ):
            score[self.last_target, self.last_slot] = -1
        flat = int(torch.argmax(score).item())
        target = flat // model.cfg.dendrites
        slot = flat % model.cfg.dendrites
        proposal_evidence = float(evidence[target, slot].item())
        if (
            not bool(active[target, slot])
            or abs(proposal_evidence) < config.minimum_eligibility
        ):
            return False

        active_slots = torch.nonzero(
            active[target],
            as_tuple=False,
        ).flatten()
        peers = active_slots[active_slots != slot]
        if peers.numel() == 0:
            return False
        direction = 1 if proposal_evidence >= 0 else -1
        delta = torch.zeros_like(active, dtype=state.hidden.dtype)
        delta[target, slot] = direction * config.proposal_step
        delta[target, peers] = (
            -direction * config.proposal_step / peers.numel()
        )

        self.active = True
        self.target = target
        self.slot = slot
        self.direction = direction
        self.proposal_evidence = proposal_evidence
        self.started_update = int(update)
        self.observations = 0
        self.candidate_reward_sum = 0.0
        self.incumbent_reward_sum = 0.0
        self.candidate_observations = 0
        self.incumbent_observations = 0
        self.assignment_code = deterministic_assignment_code(
            topology_seed=model.cfg.topology_seed,
            trial_index=self.total_started,
            target=target,
            slot=slot,
            start_update=update,
            trial_updates=config.trial_updates,
        )
        self.candidate_schedule = list(
            crossover_schedule(
                self.assignment_code,
                config.trial_updates,
            )
        )
        self.reward_observations = []
        self.randomization_p_value = 1.0
        self.randomization_extreme_assignments = 0
        self.randomization_total_assignments = 0
        self.legacy_fixed_schedule = False
        self.delta = delta
        self.total_started += 1
        return True

    @torch.no_grad()
    def resolve(
        self,
        model: "SparseAxonField",
        state: "FieldState",
        config: RoutingTrafficConfig,
        update: int,
    ) -> bool:
        """Commit a causally favorable preference delta or retain incumbent routing."""

        if not self.active or self.delta is None:
            raise RuntimeError("no routing traffic trial is active")
        if (
            self.candidate_observations == 0
            or self.incumbent_observations == 0
        ):
            raise RuntimeError(
                "routing traffic trial requires both crossover arms"
            )
        advantage = self.mean_advantage
        if self.legacy_fixed_schedule:
            randomization_p_value = 1.0
            randomization_extreme_assignments = 0
            randomization_total_assignments = 0
            clears_randomization = True
        else:
            if len(self.reward_observations) != config.trial_updates:
                raise RuntimeError(
                    "randomized routing trial requires every reward observation"
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
                randomization_p_value
                <= config.randomization_alpha
            )
        committed = (
            advantage > config.margin and clears_randomization
        )
        if committed:
            preference = (
                state.credit_routing_preference
                + self.delta.unsqueeze(0)
            )
            active = model.active_edges.unsqueeze(0)
            count = active.sum(dim=2, keepdim=True).clamp_min(1)
            mean = (
                (preference * active).sum(dim=2, keepdim=True)
                / count
            )
            centered = (preference - mean) * active
            limit = model.cfg.credit_routing_preference_limit
            bounded = (
                limit * torch.tanh(centered / limit) * active
            )
            bounded_mean = (
                (bounded * active).sum(dim=2, keepdim=True)
                / count
            )
            bounded = (bounded - bounded_mean) * active
            maximum = bounded.abs().amax(dim=2, keepdim=True)
            scale = torch.where(
                maximum > limit,
                limit / maximum.clamp_min(1e-12),
                torch.ones_like(maximum),
            )
            state.credit_routing_preference = bounded * scale
            self.total_committed += 1
        else:
            self.total_rejected += 1

        self.trial_history.append(
            {
                "mode": "routing_exploratory_traffic",
                "outcome": "committed" if committed else "rejected",
                "started_update": self.started_update,
                "resolved_update": int(update),
                "target": self.target,
                "slot": self.slot,
                "direction": self.direction,
                "proposal_evidence": self.proposal_evidence,
                "proposal_step": config.proposal_step,
                "observations": self.observations,
                "candidate_observations": self.candidate_observations,
                "incumbent_observations": self.incumbent_observations,
                "mean_candidate_reward": (
                    self.candidate_reward_sum
                    / self.candidate_observations
                ),
                "mean_incumbent_reward": (
                    self.incumbent_reward_sum
                    / self.incumbent_observations
                ),
                "decision_advantage": advantage,
                "decision_margin": config.margin,
                "assignment_code": self.assignment_code,
                "candidate_schedule": list(
                    self.candidate_schedule
                ),
                "reward_observations": list(
                    self.reward_observations
                ),
                "randomization_p_value": (
                    None
                    if self.legacy_fixed_schedule
                    else randomization_p_value
                ),
                "randomization_extreme_assignments": (
                    None
                    if self.legacy_fixed_schedule
                    else randomization_extreme_assignments
                ),
                "randomization_total_assignments": (
                    None
                    if self.legacy_fixed_schedule
                    else randomization_total_assignments
                ),
                "randomization_alpha": config.randomization_alpha,
                "legacy_fixed_schedule": self.legacy_fixed_schedule,
                "active_slots": int(
                    model.active_edges[self.target].sum().item()
                ),
            }
        )
        self.last_mean_advantage = advantage
        self.randomization_p_value = randomization_p_value
        self.last_randomization_p_value = randomization_p_value
        self.randomization_extreme_assignments = (
            randomization_extreme_assignments
        )
        self.randomization_total_assignments = (
            randomization_total_assignments
        )
        self.last_randomization_extreme_assignments = (
            randomization_extreme_assignments
        )
        self.last_randomization_total_assignments = (
            randomization_total_assignments
        )
        self.last_target = self.target
        self.last_slot = self.slot
        self.active = False
        self.delta = None
        return committed


def routing_traffic_due(
    config: RoutingTrafficConfig,
    update: int,
    *,
    structural_decision_due: bool,
) -> bool:
    """Use a due phase only when no structural or reporting decision owns it."""

    if (
        not config.enabled
        or update < config.warmup_updates
        or update % config.interval != 0
        or structural_decision_due
        or (
            config.boundary_interval > 0
            and (
                update % config.boundary_interval == 0
                or (
                    update % config.boundary_interval
                    + config.trial_updates
                    > config.boundary_interval
                )
            )
        )
    ):
        return False
    return True


def routing_traffic_update(
    trial: RoutingTrafficTrial,
    *,
    started: bool = False,
    resolved: bool = False,
    committed: bool = False,
    candidate_exposed: bool = False,
) -> RoutingTrafficUpdate:
    """Snapshot routing-trial telemetry after one trainer transition."""

    return RoutingTrafficUpdate(
        active=trial.active,
        started=started,
        resolved=resolved,
        committed=committed,
        candidate_exposed=candidate_exposed,
        total_started=trial.total_started,
        total_committed=trial.total_committed,
        total_rejected=trial.total_rejected,
        advantage=trial.mean_advantage,
        randomization_p_value=(
            trial.randomization_p_value
            if trial.active
            else trial.last_randomization_p_value
        ),
        target=trial.target,
        slot=trial.slot,
    )


def routing_traffic_summary(
    config: RoutingTrafficConfig,
    trial: RoutingTrafficTrial,
) -> dict[str, Any]:
    """Return checkpoint-friendly routing policy and decision evidence."""

    return {
        "config": asdict(config),
        "active": trial.active,
        "target": trial.target,
        "slot": trial.slot,
        "direction": trial.direction,
        "proposal_evidence": trial.proposal_evidence,
        "started_update": trial.started_update,
        "observations": trial.observations,
        "assignment_code": trial.assignment_code,
        "candidate_schedule": list(trial.candidate_schedule),
        "reward_observations": list(trial.reward_observations),
        "candidate_observations": trial.candidate_observations,
        "incumbent_observations": trial.incumbent_observations,
        "mean_candidate_reward": (
            trial.candidate_reward_sum
            / max(1, trial.candidate_observations)
        ),
        "mean_incumbent_reward": (
            trial.incumbent_reward_sum
            / max(1, trial.incumbent_observations)
        ),
        "mean_advantage": trial.mean_advantage,
        "randomization_p_value": (
            trial.randomization_p_value
            if trial.active
            else trial.last_randomization_p_value
        ),
        "randomization_extreme_assignments": (
            trial.randomization_extreme_assignments
            if trial.active
            else trial.last_randomization_extreme_assignments
        ),
        "randomization_total_assignments": (
            trial.randomization_total_assignments
            if trial.active
            else trial.last_randomization_total_assignments
        ),
        "total_started": trial.total_started,
        "total_committed": trial.total_committed,
        "total_rejected": trial.total_rejected,
        "trial_history": [
            dict(event) for event in trial.trial_history
        ],
    }
