"""Persistent plasticity-pressure decisions for endogenous SOL2 growth."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DevelopmentDecision:
    update: int
    b_accuracy: float
    pressure: float
    improvement: float | None
    high_pressure_streak: int
    plateau_streak: int
    eligible: bool
    trigger: bool
    reason: str | None


class DevelopmentController:
    """Trigger bounded growth from persistent pressure or pressure plus plateau."""

    def __init__(
        self,
        *,
        high_pressure: float = 0.75,
        plateau_pressure: float = 0.60,
        plateau_gain: float = 0.03,
        patience_checks: int = 2,
        min_update: int = 600,
        refractory_updates: int = 600,
        max_events: int = 2,
        target_accuracy: float = 0.80,
    ) -> None:
        self.high_pressure = high_pressure
        self.plateau_pressure = plateau_pressure
        self.plateau_gain = plateau_gain
        self.patience_checks = patience_checks
        self.min_update = min_update
        self.refractory_updates = refractory_updates
        self.max_events = max_events
        self.target_accuracy = target_accuracy
        self.high_pressure_streak = 0
        self.plateau_streak = 0
        self.last_b_accuracy: float | None = None
        self.last_growth_update: int | None = None
        self.events = 0

    def observe(
        self, *, update: int, b_accuracy: float, pressure: float
    ) -> DevelopmentDecision:
        improvement = (
            None
            if self.last_b_accuracy is None
            else b_accuracy - self.last_b_accuracy
        )
        self.high_pressure_streak = (
            self.high_pressure_streak + 1
            if pressure >= self.high_pressure
            else 0
        )
        plateau = (
            improvement is not None
            and improvement < self.plateau_gain
            and pressure >= self.plateau_pressure
        )
        self.plateau_streak = self.plateau_streak + 1 if plateau else 0
        refractory_done = (
            self.last_growth_update is None
            or update - self.last_growth_update >= self.refractory_updates
        )
        eligible = (
            update >= self.min_update
            and b_accuracy < self.target_accuracy
            and self.events < self.max_events
            and refractory_done
        )
        reason = None
        if eligible and self.high_pressure_streak >= self.patience_checks:
            reason = "persistent_high_pressure"
        elif eligible and self.plateau_streak >= self.patience_checks:
            reason = "pressure_with_plateau"
        trigger = reason is not None
        if trigger:
            self.events += 1
            self.last_growth_update = update
            self.high_pressure_streak = 0
            self.plateau_streak = 0
        self.last_b_accuracy = b_accuracy
        return DevelopmentDecision(
            update=update,
            b_accuracy=b_accuracy,
            pressure=pressure,
            improvement=improvement,
            high_pressure_streak=self.high_pressure_streak,
            plateau_streak=self.plateau_streak,
            eligible=eligible,
            trigger=trigger,
            reason=reason,
        )

    def state_dict(self) -> dict:
        return {
            "high_pressure_streak": self.high_pressure_streak,
            "plateau_streak": self.plateau_streak,
            "last_b_accuracy": self.last_b_accuracy,
            "last_growth_update": self.last_growth_update,
            "events": self.events,
        }

    def configuration(self) -> dict:
        return {
            "high_pressure": self.high_pressure,
            "plateau_pressure": self.plateau_pressure,
            "plateau_gain": self.plateau_gain,
            "patience_checks": self.patience_checks,
            "min_update": self.min_update,
            "refractory_updates": self.refractory_updates,
            "max_events": self.max_events,
            "target_accuracy": self.target_accuracy,
        }

    def load_state_dict(self, payload: dict) -> None:
        for key in self.state_dict():
            setattr(self, key, payload[key])


def decision_payload(decision: DevelopmentDecision) -> dict:
    return asdict(decision)
