# `train.py`

## Purpose

Runs continuous truncated-BPTT next-character training and samples through the same
one-character-at-a-time organism path used during training.

## Components

### `TrainMetrics`
- **Does**: Reports language loss alongside measured energy, novelty, cell credit, and
  edge credit.

### `ContinuousTrainer`
- **Does**: Owns the persistent field state, corpus cursor, optimizer, and credit loop.
- **Interacts with**: `SparseAxonField` in `model.py` and `ContinuousCharStream` in
  `stream.py`.
- **Rationale**: An optimizer boundary detaches history but never resets experience.

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
| Future checkpointing | Trainer owns model, optimizer, stream position, and field state | Ownership changes |

## Notes

- Cross-entropy is the current continuous task reward. A bounded per-example score is
  also delivered on the next tick through persistent eligibility.
- This is a capability probe, not yet a competitive NanoGPT implementation.
