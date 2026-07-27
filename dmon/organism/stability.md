# `stability.py`

## Purpose

Measures whether an uninterrupted SOL run remains viable after reaching its best
held-out character loss.

## Components

### `load_sol_evaluation_history`
- **Does**: Reads finite SOL validation BPC points from append-only JSONL and keeps the
  last row for each update.
- **Rationale**: Resume can repeat an evaluation update; scientific summaries should not
  double-count it.

### `summarize_stability`
- **Does**: Reports best, final, and worst post-best BPC plus their regressions and a
  thresholded stable/unstable verdict.
- **Rationale**: Early learning is expected to improve sharply, so only behavior after
  the best point is evidence of collapse.

### `summarize_exploratory_survival`
- **Does**: Aligns each non-virtual within-organism traffic trial with the last
  validation before exploration and all validations after its decision but before the
  next trial.
- **Does**: Reports committed/rejected counts, pending coverage, decision advantage,
  and whether the same developing body exceeded the allowed BPC regression.
- **Rationale**: Candidate traffic decides causality; decision-aligned validation checks
  that exploration or grafting did not merely produce a short-lived organ in a
  deteriorating body.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | Summary stability is derived from completed validation history | Metric keys |
| `promote.py` | A live candidate completed its final evaluation and stayed within the configured regression threshold | Verdict semantics |
| S6 reports | Every exploratory trial remains individually auditable against later whole-organism validation | Alignment window or ledger keys |
