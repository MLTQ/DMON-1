# train.py

## Purpose
Multi-lane continuous chunk trainer with held-out ablations and optional GRU control.

## Components

### `train_creature`
- **Does**: Stream windows of `chunk_length`, AdamW, detach state, periodic eval with reset/shuffle Δ

### `train_gru_control`
- **Does**: Parameter-budget-matched GRU on identical stream protocol

## Notes
- Prefer `--updates` over legacy `--steps`.
- `--structural` enables probe rewiring (default off).
