# `schedule.py`

## Purpose

Provides parameter-free optimizer schedules shared by SOL and its matched behavioral
controls. The schedule is a pure function of the upcoming update, so checkpoint resume
does not reset or shift it.

## Components

### `cosine_decay_learning_rate`
- **Does**: Holds the base rate through a specified update, then follows a cosine curve
  to a fixed minimum ratio at an explicit end update.
- **Rationale**: SOL repeatedly improves early and regresses late under a constant
  learning rate; explicit boundaries allow that failure mode to be tested without
  changing the organism or parameter budget.
- **Compatibility**: A zero decay end returns the historical constant learning rate.

### `set_optimizer_learning_rate`
- **Does**: Applies the selected rate to every optimizer parameter group immediately
  before an update.
- **Interacts with**: `benchmark.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | The same update and arguments yield the same rate after resume | Boundary or interpolation semantics |
| SOL experiment reports | Disabled scheduling preserves the old optimizer trajectory | Non-identity disabled path |
| Matched controls | SOL, GRU, and transformer use identical schedule semantics | Model-specific scheduling |
