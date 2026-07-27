# `convergence.py`

## Purpose

Determines whether a completed SOL run stopped on a statistically supported plateau or
while held-out loss was still changing. This prevents endpoint rankings from being
treated as capability evidence when the chosen training horizon is arbitrary.

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

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | Every completed SOL summary includes a terminal-horizon verdict | Summary keys or status names |
| Experiment reports | A meaningful comparison includes curves plus the stored terminal slope/noise evidence | Slope units or interval semantics |

## Notes

- `horizon_informative` means the observed window supports a practical plateau; it is
  not a claim that later optimization can never resume.
- Comparison gaps must still be judged against seed variance and terminal movement
  across every arm.
