# `report.py`

## Purpose

Builds the S0 comparison table from completed run artifacts while guarding the
denominators that previously produced misleading project verdicts.

## Components

### `load_run`
- **Does**: Requires a completed `summary.json`, positive parameter/update counts, and
  finite held-out BPC.
- **Rationale**: A missing or collapsed run cannot become a favorable baseline.

### `compare_runs`
- **Does**: Enforces parameter and update-budget tolerances before ranking runs.
- **Does**: Computes reset and shuffled-state penalties for SOL.
- **Rationale**: "The persistent substrate matters" must be a causal delta.

### `markdown_report`
- **Does**: Renders the validated data into a compact human-readable table.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | Summaries contain model, params, updates, best/final evaluation | Summary schema |
| S0 decision | Lower held-out BPC wins only after budget guards pass | Guard tolerances |
