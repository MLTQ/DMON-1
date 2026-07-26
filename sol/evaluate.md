# `evaluate.py`

## Purpose

Measures held-out character prediction and organism dynamics on a contiguous stream,
including controls that test whether persistent cellular state is doing useful work.

## Components

### `EvaluationMetrics`
- **Does**: Reports NLL, bits per character, perplexity, accuracy, energy, novelty, and
  measured edge flow together.
- **Rationale**: Behavioral and organism claims must come from the same evaluated tokens.

### `evaluate_sol`
- **Does**: Warms a fresh organism on held-out text and scores the following contiguous
  characters without updating weights.
- **Interacts with**: `SparseAxonField.tick` in `model.py`.
- **Rationale**: Evaluation follows the same one-character stream and delayed-reward
  semantics as training.

### `evaluate_state_ablations`
- **Does**: Compares intact persistence with per-token reset and deterministic cell-state
  shuffling.
- **Rationale**: A good loss is not evidence of memory unless breaking memory hurts.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | All policies score the same held-out prefix and token count | Split or warmup semantics |
| Experiment reports | BPC is NLL divided by `ln(2)` | Metric definitions |
