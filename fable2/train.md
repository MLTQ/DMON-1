# train.py

## Purpose

The G1 trainer: fresh-episode paired counterfactual training of the Broca graft,
with the arm battery, strict-win counting, per-depth lesions, atomic checkpoints,
and JSON telemetry.

## Components

- `trainable_parameter_groups` — organism kernel and Broca organ as two AdamW
  groups (`organ_lr_multiplier`); raises if either is empty.
- `gradient_norm_or_raise` — non-finite gradients and norms above
  `reject_grad_norm` raise; nothing is skipped silently.
- `evaluate_split` — full battery over one split in eval mode, per-question losses
  retained for strict wins, per-depth lesion deltas.
- `save_checkpoint` / `load_checkpoint` — schema-tagged, atomic (`.tmp` + replace),
  RNG state included; depth mismatch on resume raises.
- `run_training` / CLI — episode loop, eval/checkpoint cadence, result JSON with
  peak VRAM.

## Decisions

- **Eval runs in eval mode.** `EffectiveLinear` advances its power-iteration buffer
  during training-mode forwards, so a training-mode evaluation would make measured
  numbers depend on arm order (caught by the checkpoint contract test).
- **The wrong-passage arm is computed live in the training step** (see
  `episodes.md`); only the no-exposure baseline runs under `no_grad`.
- **Guards raise instead of returning plausible numbers.** The C1-review found
  `nll = total / max(count, 1)` returning a perfect 0.0 BPC on zero tokens in
  `sol2/evaluate.py`; here empty splits and empty arm batteries raise.
- **Every result JSON carries the full config**, the resolved depths, and
  per-question losses — the numbers behind every headline mean are in the artifact.

## Contracts

- Checkpoint round-trip restores bitwise organism state and identical episode
  scores (contract-tested through the toy backbone).
- Training raises on non-finite or rejected gradient norms rather than skipping.
