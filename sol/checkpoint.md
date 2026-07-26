# `checkpoint.py`

## Purpose

Persists the entire continuous SOL process so resume does not reset the creature,
optimizer, corpus position, or random stream.

## Components

### `save_checkpoint`
- **Does**: Writes a versioned trainer snapshot to a sibling temporary file and
  atomically replaces the destination.
- **Interacts with**: `ContinuousTrainer.state_dict` in `train.py`.
- **Rationale**: Interrupted writes must not destroy the last usable checkpoint.

### `load_checkpoint`
- **Does**: Validates the schema and reconstructs a trainer on the requested device.
- **Interacts with**: `ContinuousTrainer.from_state_dict` in `train.py`.

### `load_organism`
- **Does**: Loads one persistent batch lane, its model, vocabulary, update count, and
  provenance without needing the training corpus or optimizer.
- **Interacts with**: `serve.py`.
- **Rationale**: Local inference should not pretend to resume the training stream.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Benchmark and local UI backend | Checkpoint includes weights, live state, vocabulary, and metadata | Schema or state keys |
| Long training runs | A resumed update consumes the exact next corpus window | Stream or RNG restoration |

## Notes

- Checkpoint tensors are run artifacts and remain outside Git.
