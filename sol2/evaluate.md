# `evaluate.py`

## Purpose

Scores identical held-out tokens under interventions that separate state causality,
tissue work, private specialization, and sensory-memory contribution.

## Components

- `prequential` — one uninterrupted held-out walk with no label-feedback pathway.
- `evaluate_model` — ordinary control evaluation.
- `evaluate_with_ablations` — reset, freeze, within-tissue shuffle, and memory-zero
  interventions, including separate compute/relay freeze costs.

## Decisions

- Freeze measures whether tissue does work.
- Shuffle is restricted within each tissue type, so breaking a declared type contract
  cannot masquerade as private cell differentiation.
- Memory-zero is reported separately from total-state reset.

## Contracts

- Every intervention scores the same contiguous held-out prefix after the same warmup.
- Positive deltas mean the intervention harmed prediction.
- Evaluation never updates weights or supplies target information to persistent state.
