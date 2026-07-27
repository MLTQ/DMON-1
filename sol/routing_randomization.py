"""Exact block-randomized inference for live routing traffic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


_MASK_64 = (1 << 64) - 1
_ABBA = (True, False, False, True)
_BAAB = (False, True, True, False)


@dataclass(frozen=True)
class RandomizationResult:
    """Exact one-sided rank of one observed crossover assignment."""

    observed_advantage: float
    extreme_assignments: int
    total_assignments: int

    @property
    def p_value(self) -> float:
        """Return the exact fraction at least as favorable as observed."""

        return self.extreme_assignments / self.total_assignments


def _splitmix64(value: int) -> int:
    """Mix one integer without touching process-global random state."""

    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = (
        (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    ) & _MASK_64
    value = (
        (value ^ (value >> 27)) * 0x94D049BB133111EB
    ) & _MASK_64
    return value ^ (value >> 31)


def crossover_schedule(
    assignment_code: int,
    trial_updates: int,
) -> tuple[bool, ...]:
    """Expand assignment bits into balanced, trend-neutral ABBA/BAAB blocks."""

    if trial_updates < 4 or trial_updates % 4 != 0:
        raise ValueError(
            "randomized routing trial length must be divisible by four"
        )
    blocks = trial_updates // 4
    if blocks > 16:
        raise ValueError(
            "exact routing randomization supports at most sixteen blocks"
        )
    if assignment_code < 0 or assignment_code >= 1 << blocks:
        raise ValueError("routing assignment code exceeds trial capacity")
    schedule: list[bool] = []
    for block in range(blocks):
        schedule.extend(
            _BAAB if assignment_code & (1 << block) else _ABBA
        )
    return tuple(schedule)


def deterministic_assignment_code(
    *,
    topology_seed: int,
    trial_index: int,
    target: int,
    slot: int,
    start_update: int,
    trial_updates: int,
) -> int:
    """Choose one reproducible crossover schedule from proposal identity."""

    blocks = trial_updates // 4
    crossover_schedule(0, trial_updates)
    mask = (1 << blocks) - 1
    value = 0
    for component in (
        trial_index,
        target,
        slot,
        start_update,
    ):
        value = _splitmix64(
            value ^ (int(component) & _MASK_64)
        )
    return (value ^ int(topology_seed)) & mask


def schedule_advantage(
    rewards: Sequence[float],
    schedule: Sequence[bool],
) -> float:
    """Return candidate-minus-incumbent mean reward for one assignment."""

    if len(rewards) != len(schedule) or not rewards:
        raise ValueError("rewards and schedule must have equal nonzero length")
    candidate = [
        float(reward)
        for reward, exposed in zip(rewards, schedule, strict=True)
        if exposed
    ]
    incumbent = [
        float(reward)
        for reward, exposed in zip(rewards, schedule, strict=True)
        if not exposed
    ]
    if not candidate or len(candidate) != len(incumbent):
        raise ValueError("routing assignment must balance both traffic arms")
    return sum(candidate) / len(candidate) - sum(incumbent) / len(
        incumbent
    )


def exact_one_sided_randomization_test(
    rewards: Sequence[float],
    observed_schedule: Sequence[bool],
) -> RandomizationResult:
    """Return the observed effect and its exact crossover null rank."""

    trial_updates = len(observed_schedule)
    crossover_schedule(0, trial_updates)
    if len(rewards) != trial_updates:
        raise ValueError(
            "rewards and observed schedule must have equal length"
        )
    blocks = trial_updates // 4
    observed_code = 0
    for block in range(blocks):
        pattern = tuple(
            observed_schedule[block * 4 : block * 4 + 4]
        )
        if pattern == _BAAB:
            observed_code |= 1 << block
        elif pattern != _ABBA:
            raise ValueError(
                "observed routing schedule is not block-randomized"
            )
    canonical = crossover_schedule(observed_code, trial_updates)
    observed = schedule_advantage(rewards, canonical)
    assignments = 1 << blocks
    at_least_observed = 0
    for code in range(assignments):
        null_advantage = schedule_advantage(
            rewards,
            crossover_schedule(code, trial_updates),
        )
        if null_advantage >= observed - 1e-12:
            at_least_observed += 1
    return RandomizationResult(
        observed_advantage=observed,
        extreme_assignments=at_least_observed,
        total_assignments=assignments,
    )


def exact_one_sided_randomization_p_value(
    rewards: Sequence[float],
    observed_schedule: Sequence[bool],
) -> float:
    """Return the exact one-sided p-value for compatibility with analyses."""

    return exact_one_sided_randomization_test(
        rewards,
        observed_schedule,
    ).p_value
