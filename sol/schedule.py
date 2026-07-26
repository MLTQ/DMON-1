"""Parameter-free learning-rate schedules for reproducible SOL experiments."""

from __future__ import annotations

import math

from torch.optim import Optimizer


def cosine_decay_learning_rate(
    base_learning_rate: float,
    update: int,
    decay_start: int = 0,
    decay_end: int = 0,
    minimum_ratio: float = 0.1,
) -> float:
    """Hold the base rate, then decay smoothly to a fixed minimum.

    A zero ``decay_end`` disables the schedule. Updates name the optimizer step
    about to run, making the result independent of checkpoint boundaries.
    """

    if base_learning_rate <= 0:
        raise ValueError("base learning rate must be positive")
    if update < 1:
        raise ValueError("update must be positive")
    if decay_start < 0 or decay_end < 0:
        raise ValueError("decay boundaries must be non-negative")
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("minimum learning-rate ratio must be in [0, 1]")
    if decay_end == 0:
        return base_learning_rate
    if decay_end <= decay_start:
        raise ValueError("decay end must be greater than decay start")
    if update <= decay_start:
        return base_learning_rate
    if update >= decay_end:
        return base_learning_rate * minimum_ratio

    progress = (update - decay_start) / (decay_end - decay_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = minimum_ratio + (1.0 - minimum_ratio) * cosine
    return base_learning_rate * multiplier


def set_optimizer_learning_rate(
    optimizer: Optimizer,
    learning_rate: float,
) -> None:
    """Apply one scheduled rate to every optimizer parameter group."""

    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
