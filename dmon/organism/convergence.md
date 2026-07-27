# `convergence.py`

## Purpose

Determines whether a completed SOL run stopped on a statistically supported plateau and
whether a paired treatment/control ordering remains meaningful despite nonzero terminal
trend. This prevents arbitrary endpoints without requiring learning curves to become
mathematically flat.

## Components

### `summarize_convergence`
- **Does**: Fits ordinary least squares to the final configurable validation window.
- **Does**: Reports terminal BPC change, slope per 100 updates, residual noise, slope
  standard error, and a two-sided 95% Student-t interval.
- **Does**: Classifies the horizon as `still_improving`, `worsening`,
  `plateau_supported`, `unresolved_terminal_noise`, or `insufficient_history`.
- **Rationale**: A plateau is supported only when the entire slope interval fits inside
  the configured practical-equivalence band; merely ending at the best checkpoint is
  not convergence.

### `summarize_comparison_horizon`
- **Does**: Aligns the final treatment/control evaluations and reports mean, endpoint,
  and worst paired gaps plus the fraction won by each arm.
- **Does**: Compares the paired effect with combined terminal residual noise and reports
  relative slope and a clearly labeled linear crossing extrapolation.
- **Does**: Accepts a comparison horizon when either both arms support a practical
  plateau or one ordering is sufficiently consistent and large relative to measured
  terminal noise.
- **Rationale**: Continued noisy improvement does not invalidate a treatment effect
  that persists throughout the terminal window; slope remains a caveat rather than an
  impossible zero-change prerequisite.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | Every completed SOL summary includes a terminal-horizon verdict | Summary keys or status names |
| Experiment reports | A meaningful comparison includes curves, paired gaps, and stored terminal slope/noise evidence | Slope units, alignment, or status semantics |

## Notes

- Per-run `horizon_informative` means the observed window supports a practical plateau;
  comparison-level `horizon_informative` can instead mean the paired ordering is robust.
- Linear crossing time assumes terminal slopes persist indefinitely. It is diagnostic,
  not a forecast or an automatic veto.
- Seed variance and replication remain separate requirements.
