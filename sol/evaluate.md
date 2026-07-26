# `evaluate.py`

## Purpose

Measures held-out character prediction and organism dynamics on a contiguous stream,
including controls that test whether persistent cellular state is doing useful work.

## Components

### `EvaluationMetrics`
- **Does**: Reports NLL, bits per character, perplexity, accuracy, energy, novelty, and
  measured edge flow plus edge eligibility, mean fast synaptic efficacy, and saturation
  together with causal-probe flow/eligibility, backward-credit magnitude, and total
  structural rewrites.
- **Does**: Reports viability, quiescence, external energy input, spending, and
  transport drift on the same scored tokens.
- **Rationale**: Behavioral and organism claims must come from the same evaluated tokens.

### `evaluate_sol`
- **Does**: Warms a fresh organism on held-out text and scores the following contiguous
  characters without updating weights.
- **Does**: An optional `score_start` anchors evaluation so different warmups predict
  identical characters.
- **Interacts with**: `SparseAxonField.tick` in `model.py`.
- **Rationale**: Evaluation follows the same one-character stream and delayed-reward
  semantics and persistent surprise baseline as training.

### `evaluate_state_ablations`
- **Does**: Compares intact persistence with per-token reset and deterministic cell-state
  shuffling, including target-owned edge/probe eligibility, backward credit, and fast
  efficacy.
- **Does**: Scores the trained organism while zeroing fast efficacy before every tick,
  isolating whether online synaptic memory contributes at inference.
- **Does**: Replaces a grown source table with the deterministic birth topology while
  preserving learned weights and recurrent policy, then restores the live connectome.
- **Rationale**: A matched training gain is stronger evidence when the winning organism
  also causally depends on its installed morphology.
- **Rationale**: A good loss is not evidence of memory unless breaking memory hurts.

### `evaluate_warmup_sweep`
- **Does**: Compares unique warmup lengths against one fixed held-out scoring window.
- **Rationale**: Moving the scored text with the warmup would confound state settling
  with corpus difficulty.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | All policies score the same held-out prefix and token count | Split or warmup semantics |
| Warmup diagnostics | Every row predicts identical token indices | `score_start` semantics |
| Experiment reports | BPC is NLL divided by `ln(2)` | Metric definitions |
