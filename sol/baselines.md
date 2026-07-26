# `baselines.py`

## Purpose

Provides conventional controls trained on the identical continuous character stream.
The first control is a parameter-matched GRU; a causal transformer control follows once
the recurrent benchmark pipeline is stable.

## Components

### `CharacterGRU`
- **Does**: Predicts every next character with an embedding, one recurrent layer, and a
  linear readout while retaining state across optimizer windows.
- **Interacts with**: `ContinuousCharStream` in `stream.py` and `benchmark.py`.

### `match_gru_hidden_size`
- **Does**: Selects the GRU width whose trainable parameter count is closest to SOL.
- **Rationale**: Comparing a tiny control to a much larger organism would make the
  headline ratio meaningless.

### `evaluate_gru`
- **Does**: Scores the same warmup and contiguous held-out window used for SOL.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | `forward(tokens, state)` returns time-aligned logits and next state | Shape or state semantics |
| S0 verdict | Matched parameter count is reported beside both models | Counting rules |
