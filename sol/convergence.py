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


def summarize_comparison_horizon(
    candidate_history: list[tuple[int, float]],
    control_history: list[tuple[int, float]],
    *,
    window: int = 5,
    max_terminal_slope_bpc_per_100_updates: float = 0.01,
    minimum_consistent_fraction: float = 0.8,
    minimum_effect_to_noise: float = 2.0,
) -> dict[str, Any]:
    """Decide whether a paired ordering survives terminal trend and noise."""

    if not 0.5 <= minimum_consistent_fraction <= 1:
        raise ValueError(
            "minimum consistent fraction must be between 0.5 and 1"
        )
    if minimum_effect_to_noise < 0:
        raise ValueError("minimum effect-to-noise ratio must be non-negative")
    candidate = {
        int(update): float(bpc)
        for update, bpc in candidate_history
        if math.isfinite(float(bpc))
    }
    control = {
        int(update): float(bpc)
        for update, bpc in control_history
        if math.isfinite(float(bpc))
    }
    common_updates = sorted(set(candidate) & set(control))
    terminal_updates = common_updates[-window:]
    base: dict[str, Any] = {
        "horizon_informative": False,
        "status": "insufficient_aligned_history",
        "winner": None,
        "window_requested": window,
        "aligned_evaluations_available": len(common_updates),
        "aligned_evaluations_used": len(terminal_updates),
        "first_update": (
            terminal_updates[0] if terminal_updates else None
        ),
        "final_update": (
            terminal_updates[-1] if terminal_updates else None
        ),
        "mean_candidate_minus_control_bpc": None,
        "endpoint_candidate_minus_control_bpc": None,
        "worst_candidate_minus_control_bpc": None,
        "candidate_win_fraction": None,
        "control_win_fraction": None,
        "combined_terminal_rmse_bpc": None,
        "effect_to_noise_ratio": None,
        "relative_slope_bpc_per_100_updates": None,
        "linear_updates_to_crossing": None,
        "minimum_consistent_fraction": minimum_consistent_fraction,
        "minimum_effect_to_noise": minimum_effect_to_noise,
        "candidate_terminal": None,
        "control_terminal": None,
    }
    if len(terminal_updates) < 3:
        return base

    candidate_terminal_history = [
        (update, candidate[update]) for update in terminal_updates
    ]
    control_terminal_history = [
        (update, control[update]) for update in terminal_updates
    ]
    candidate_terminal = summarize_convergence(
        candidate_terminal_history,
        window=window,
        max_terminal_slope_bpc_per_100_updates=(
            max_terminal_slope_bpc_per_100_updates
        ),
    )
    control_terminal = summarize_convergence(
        control_terminal_history,
        window=window,
        max_terminal_slope_bpc_per_100_updates=(
            max_terminal_slope_bpc_per_100_updates
        ),
    )
    gaps = [
        candidate[update] - control[update]
        for update in terminal_updates
    ]
    mean_gap = sum(gaps) / len(gaps)
    candidate_win_fraction = (
        sum(gap < 0 for gap in gaps) / len(gaps)
    )
    control_win_fraction = sum(gap > 0 for gap in gaps) / len(gaps)
    candidate_rmse = float(
        candidate_terminal["terminal_rmse_bpc"] or 0.0
    )
    control_rmse = float(
        control_terminal["terminal_rmse_bpc"] or 0.0
    )
    combined_rmse = math.hypot(candidate_rmse, control_rmse)
    effect_to_noise = (
        abs(mean_gap) / combined_rmse
        if combined_rmse > 0
        else (math.inf if mean_gap != 0 else 0.0)
    )
    if mean_gap < 0:
        winner = "candidate"
        consistent_fraction = candidate_win_fraction
    elif mean_gap > 0:
        winner = "control"
        consistent_fraction = control_win_fraction
    else:
        winner = None
        consistent_fraction = 0.0
    full_window = len(terminal_updates) == window
    ordering_supported = (
        full_window
        and winner is not None
        and consistent_fraction >= minimum_consistent_fraction
        and effect_to_noise >= minimum_effect_to_noise
    )
    both_plateaued = (
        candidate_terminal["status"] == "plateau_supported"
        and control_terminal["status"] == "plateau_supported"
    )
    if ordering_supported:
        status = f"{winner}_advantage_supported"
    elif both_plateaued:
        status = "no_ordering_detected_at_plateau"
    else:
        status = "inconclusive"

    candidate_slope = float(
        candidate_terminal["terminal_slope_bpc_per_100_updates"]
    )
    control_slope = float(
        control_terminal["terminal_slope_bpc_per_100_updates"]
    )
    relative_slope = candidate_slope - control_slope
    endpoint_gap = gaps[-1]
    closing = (
        (endpoint_gap < 0 and relative_slope > 0)
        or (endpoint_gap > 0 and relative_slope < 0)
    )
    updates_to_crossing = (
        abs(endpoint_gap / relative_slope) * 100.0
        if closing and relative_slope != 0
        else None
    )
    base.update(
        {
            "horizon_informative": ordering_supported or both_plateaued,
            "status": status,
            "winner": winner if ordering_supported else None,
            "mean_candidate_minus_control_bpc": mean_gap,
            "endpoint_candidate_minus_control_bpc": endpoint_gap,
            "worst_candidate_minus_control_bpc": max(gaps),
            "candidate_win_fraction": candidate_win_fraction,
            "control_win_fraction": control_win_fraction,
            "combined_terminal_rmse_bpc": combined_rmse,
            "effect_to_noise_ratio": effect_to_noise,
            "relative_slope_bpc_per_100_updates": relative_slope,
            "linear_updates_to_crossing": updates_to_crossing,
            "candidate_terminal": candidate_terminal,
            "control_terminal": control_terminal,
        }
    )
    return base
