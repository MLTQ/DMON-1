# `optim.py`

## Purpose

Enforces one optimizer policy for SOL2 and its baselines while exposing which module
creates any gradient pathology and how broadly private cells receive learning signal.

## Components

- `schedule_factor` / `set_learning_rate` — shared constant or cosine schedule.
- `build_optimizer` — AdamW with explicit decay/no-decay groups.
- `add_optimizer_parameters` — appends only genuinely new attached-organ parameters
  while preserving all existing optimizer moments and decay policy.
- `gradient_health` — total and top-level module gradient norms.
- `cell_gradient_utilization` — material/nonzero row fractions plus an
  entropy-effective participation fraction; reserve is the complement of effective
  participation rather than a demand for exactly zero gradients.
- `guarded_step` — rejects non-finite or astronomically large gradients before they can
  poison parameters, then clips accepted updates.

## Contracts

- The guard is a fail-safe, not the stability mechanism; bounded operators should keep
  it inactive.
- Rejected updates are counted and surfaced by the trainer.
- Reserve fraction is telemetry, not a penalty; dormant plastic capacity is allowed.
- Expression statistics cover mutable cells only; stream-written memory has no dead
  private rows.
- Schedules are pure functions of configuration and absolute update.
- Every compared model uses the same optimizer family and schedule.
- Attaching an organ never rebuilds or discards the mature substrate's optimizer state;
  new parameters begin with empty Adam moments in matching decay/no-decay groups.
