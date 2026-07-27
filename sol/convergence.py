"""Terminal-trend diagnostics for continuous SOL experiments."""

from __future__ import annotations

import math
from typing import Any


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return math.inf
    return _T_CRITICAL_95.get(degrees_of_freedom, 1.96)


def summarize_convergence(
    history: list[tuple[int, float]],
    *,
    window: int = 5,
    max_terminal_slope_bpc_per_100_updates: float = 0.01,
) -> dict[str, Any]:
    """Measure whether the observed horizon supports a terminal plateau."""

    if window < 3:
        raise ValueError("convergence window must contain at least 3 evaluations")
    if max_terminal_slope_bpc_per_100_updates < 0:
        raise ValueError("maximum terminal slope must be non-negative")
    points = [
        (int(update), float(bpc))
        for update, bpc in sorted(dict(history).items())
        if math.isfinite(float(bpc))
    ]
    terminal = points[-window:]
    base: dict[str, Any] = {
        "horizon_informative": False,
        "status": "insufficient_history",
        "window_requested": window,
        "evaluations_available": len(points),
        "evaluations_used": len(terminal),
        "max_terminal_slope_bpc_per_100_updates": (
            max_terminal_slope_bpc_per_100_updates
        ),
        "first_update": terminal[0][0] if terminal else None,
        "final_update": terminal[-1][0] if terminal else None,
        "terminal_span_updates": (
            terminal[-1][0] - terminal[0][0]
            if len(terminal) >= 2
            else 0
        ),
        "terminal_change_bpc": (
            terminal[-1][1] - terminal[0][1]
            if len(terminal) >= 2
            else None
        ),
        "terminal_slope_bpc_per_100_updates": None,
        "terminal_slope_standard_error": None,
        "terminal_slope_ci95_low": None,
        "terminal_slope_ci95_high": None,
        "terminal_rmse_bpc": None,
    }
    if len(terminal) < 3:
        return base

    mean_update = sum(update for update, _ in terminal) / len(terminal)
    mean_bpc = sum(bpc for _, bpc in terminal) / len(terminal)
    centered_update = [
        update - mean_update for update, _ in terminal
    ]
    sum_squared_update = sum(value * value for value in centered_update)
    if sum_squared_update == 0:
        return base
    slope_per_update = sum(
        centered * (bpc - mean_bpc)
        for centered, (_, bpc) in zip(centered_update, terminal)
    ) / sum_squared_update
    intercept = mean_bpc - slope_per_update * mean_update
    residuals = [
        bpc - (intercept + slope_per_update * update)
        for update, bpc in terminal
    ]
    residual_sum_squares = sum(value * value for value in residuals)
    degrees_of_freedom = len(terminal) - 2
    residual_variance = residual_sum_squares / degrees_of_freedom
    slope_standard_error = math.sqrt(
        residual_variance / sum_squared_update
    )
    slope = slope_per_update * 100.0
    slope_error = slope_standard_error * 100.0
    margin = _t_critical_95(degrees_of_freedom) * slope_error
    low = slope - margin
    high = slope + margin
    threshold = max_terminal_slope_bpc_per_100_updates
    full_window = len(terminal) == window
    if high < -threshold:
        status = "still_improving"
    elif low > threshold:
        status = "worsening"
    elif full_window and low >= -threshold and high <= threshold:
        status = "plateau_supported"
    else:
        status = "unresolved_terminal_noise"

    base.update(
        {
            "horizon_informative": status == "plateau_supported",
            "status": status,
            "terminal_slope_bpc_per_100_updates": slope,
            "terminal_slope_standard_error": slope_error,
            "terminal_slope_ci95_low": low,
            "terminal_slope_ci95_high": high,
            "terminal_rmse_bpc": math.sqrt(
                residual_sum_squares / len(terminal)
            ),
        }
    )
    return base
