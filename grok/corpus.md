# corpus.py

## Purpose
Character vocabulary and multi-stream cursors over Tiny Shakespeare (or synthetic text).

## Components

### `CharCorpus`
- **Does**: stoi/itos, staggered per-batch cursors, `next_batch`, eval `window`
- **Rationale**: Parallel streams keep GPU busy without episodic resets; each row is an independent continuous text path

### `reset_cursors`
- **Does**: Place B streams at random staggered offsets (seeded)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Trainer | `next_batch(B, device) → (x[B], y[B])` | Shape/semantics of y |
| Model | `vocab_size` stable for a run | Mid-run vocab rebuild |

## Notes
- Download falls back to petridish local cache when SSL fails.
