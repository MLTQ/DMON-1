# `train.py`

## Purpose

Runs continuous truncated-BPTT next-character training and samples through the same
one-character-at-a-time organism path used during training.

## Components

### `TrainMetrics`
- **Does**: Reports language loss alongside measured energy, novelty, cell credit, and
  edge credit, fast synaptic state, causal-probe traffic, candidate advantage, and
  bounded rewiring counts.

### `ContinuousTrainer`
- **Does**: Owns the persistent field state, corpus cursor, optimizer, and credit loop.
- **Interacts with**: `SparseAxonField` in `model.py`, structural phases in
  `structure.py`, and `ContinuousCharStream` in `stream.py`.
- **Rationale**: An optimizer boundary detaches history but never resets experience.
- **Structural order**: Accumulates causal evidence and performs any fixed-shape
  rewiring only after backward, gradient clipping, and the optimizer step.

### `ContinuousTrainer.state_dict` / `from_state_dict`
- **Does**: Round-trips model weights, optimizer moments, field state, stream cursor,
  vocabulary, structural policy and buffers, hyperparameters, update count, and
  random-number state.
- **Interacts with**: `save_checkpoint` and `load_checkpoint` in `checkpoint.py`.
- **Rationale**: Resuming only weights would silently create a new organism and a new
  stream position.

### Frozen-parameter policy
- **Does**: Excludes explicitly named parameters from optimization and persists that
  policy in checkpoints.
- **Interacts with**: `benchmark.py --freeze-edges`.
- **Rationale**: A fixed-connectome control must survive resume without silently making
  its edges trainable.

### `generate`
- **Does**: Feeds a prompt and sampled characters through ordinary `tick` calls.
- **Rationale**: Generation cannot use a separate decoder path that bypasses the field.

### CLI
- **Does**: Trains on a supplied text file or a bundled tiny corpus and prints a sample.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| CLI users | `python -m sol.train` is self-contained | Flag names |
| Tests | `step()` returns finite metrics and preserves `state` | Metric fields |
| `checkpoint.py` | Trainer snapshots contain the complete continuous process | State keys |

## Notes

- Cross-entropy is the current continuous task reward. A bounded per-example score is
  also delivered on the next tick through persistent cell and edge eligibility.
- This is a capability probe, not yet a competitive NanoGPT implementation.
