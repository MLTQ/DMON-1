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

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | Summary stability is derived from completed validation history | Metric keys |
| `promote.py` | A live candidate completed its final evaluation and stayed within the configured regression threshold | Verdict semantics |
