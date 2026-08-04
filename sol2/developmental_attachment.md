# `developmental_attachment.py`

## Purpose

Runs the frozen S1-P4 organ-attachment branches with cumulative checkpoint anchors,
plasticity-pressure telemetry, and optional pressure-triggered relay-cell growth.

## Components

### `run_developmental_attachment_branch`

- Reloads the mastered rank-4 organism and recalibrates directional A utility.
- Attaches the deterministic B organ without changing A or mature optimizer moments.
- Trains plastic, uniform-anchor, measured-anchor, or developmental branches.
- At evaluation boundaries, allows the developmental controller to append relay cells
  through `grow_relay_tissue` and validates fixed-example A perturbation.
- Replays the exact growth ledger before loading a grown checkpoint on resume.
- Writes interval pressure, anchor energy, drift, telemetry, decisions, lesions, and
  integrity data atomically.

### Training and evaluation helpers

- `_train_interval` measures raw pre-clip substrate pressure, performs guarded AdamW,
  and applies the proximal anchor only after accepted updates.
- `_drift_summary` measures mature-prefix displacement after any amount of growth.
- `_growth_lesions` freezes all born cells or zeros their private adapters.
- `_mature_utility_lesions` compares equal-size high/low A-utility mature cells without
  indexing newly appended rows into the frozen utility tensor.

## Contracts

- All branches consume identical B samples and evaluation examples.
- Growth decisions never observe A accuracy.
- A is evaluated before and after growth only as a structural validity check.
- Existing parameters, anatomy, optimizer moments, and living state survive growth.
- New rows remain outside the immutable A anchor for the entire episode.
- Resume reconstructs geometry from the growth ledger before loading model and optimizer
  state, then restores controller streaks and RNG state.
- Resume rejects changed controller thresholds and retains the true pre-adaptation
  baseline rather than reevaluating the partially trained checkpoint as update zero.
- A-organ parameters remain byte-identical and no A rehearsal occurs.
