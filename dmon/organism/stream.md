# `stream.py`

## Purpose

Provides deterministic character encoding and continuous truncated windows. Windows are
optimizer boundaries, not episode boundaries.

## Components

### `CharacterVocabulary`
- **Does**: Builds a sorted character vocabulary and performs strict encode/decode.
- **Interacts with**: `ContinuousCharStream` and `train.py`.

### `ContinuousCharStream`
- **Does**: Maintains staggered corpus cursors for a batch of uninterrupted streams.
- **Rationale**: The successor of the last character in one window is the first target
  implied by the next; no random-window reset silently erases experience.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `ContinuousTrainer` | `next(length)` returns `(batch, length)` input/target tensors | Shape or cursor semantics |
| Checkpointing | `state_dict()` preserves the corpus position | State key names |
