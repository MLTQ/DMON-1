# `organ_attachment.py`

## Purpose

Runs one resumable branch of the S1-P2 true organ-attachment experiment from the
mastered acquisition checkpoint. Independent processes make the four branches easy to
place concurrently across available GPUs without sharing mutable state.

## Components

### `run_organ_attachment_branch`

- Loads the acquired A organism or reconstructs the exact seed-matched scratch body.
- Creates an independently seeded B sensor/effector bundle for every B branch.
- Trains in checkpointed intervals and writes raw records, per-length evaluations,
  A-return probes, causal lesions, integrity digests, and neuron telemetry.
- Uses compact 16-batch interval diagnostics and independent 64-batch terminal
  estimates; A return and A-control terminal estimates share exact examples.
- Reports total and trainable parameter counts separately so the organ-only branch does
  not make the physical organism appear smaller merely because its substrate is frozen.
- Resumes model, optimizer, living state, counters, histories, and RNG state.

### Branch construction

- `control` retains A and its acquisition optimizer moments.
- `full` retains those moments and appends fresh B parameters to AdamW without
  rebuilding the mature optimizer.
- `organ_only` freezes every parameter except B and builds an optimizer containing
  only that bundle.
- `scratch` deterministically rebuilds the initial substrate, retains an unused A port,
  and attaches the same initialized B used by acquired branches.

### Removal and reattachment

The full branch removes the exact B module object, trains A with the shared substrate,
then restores the same object and measures B before and during recovery. Existing
optimizer references remain valid while the off-body module receives no gradients.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| S1-P2 service manifests | One branch per output directory and GPU process | Changing branch names or CLI paths |
| Cross-branch analysis | Full, organ-only, and scratch use identical B sample seeds | Branch-specific B generator seeds |
| Retention claim | A organ digest is identical before/after B adaptation | Updating inactive A parameters |
| Modular-reuse claim | Organ-only substrate digest is identical | Any core parameter receiving an update |
| Resume | `adaptation.pt` restores the next interval exactly | Omitting optimizer, living state, histories, or RNG |

## Notes

- The result-bearing protocol remains batch 24; throughput experiments cannot change it
  after preregistration.
- Removal/recovery is rerun from the completed adaptation checkpoint after an
  interruption; the 2,000-update attribution phase itself is interval-resumable.
