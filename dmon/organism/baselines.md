# `baselines.py`

## Purpose

Provides conventional controls trained on the identical continuous character stream:
a parameter-matched GRU and a bounded-context NanoGPT-style causal transformer.

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

### `CausalCharacterTransformer`
- **Does**: Runs a pre-norm causal transformer over a rolling token context and returns
  logits only for newly streamed characters.
- **Rationale**: It is the conventional attention-based behavioral control, not part of
  the organism.

### `match_transformer_hidden_size`
- **Does**: Selects a head-compatible width nearest SOL's trainable parameter count.

### `evaluate_transformer`
- **Does**: Scores held-out characters through the same persistent rolling context used
  during training.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | `forward(tokens, state)` returns time-aligned logits and next state | Shape or state semantics |
| S0 verdict | Matched parameter count is reported beside both models | Counting rules |
