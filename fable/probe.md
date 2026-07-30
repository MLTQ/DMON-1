# probe.py

## Purpose

Separate two questions about any cell population that a single ablation
conflates: **does this tissue do work** (freeze) and **is this tissue
differentiated** (shuffle).

## Components

- `load_creature` — rebuild a `Fable` from a checkpoint's own config.
- `probe(checkpoint, device)` — normal BPC plus freeze-internal,
  freeze-mirror, freeze-both, and shuffle-internal, with deltas.
- CLI: `python -m fable.probe <ckpt>... --device cuda:0`.

## Decisions

- **This module exists because of a wrong call.** F0's shuffle-internal
  deltas (+0.036..+0.066) were read as "internal tissue is near-inert" and
  used to block F1. The freeze probe then measured +0.280/+0.831/+0.409 on
  the same checkpoints: the tissue does substantial work and is merely
  permutation-invariant. Shuffle alone cannot distinguish a dead population
  from a mean-field one, so both probes now travel together.
- Frozen cells are held at **zero** every micro-step (`frozen_idx` in
  `Fable.step`), so they emit zero messages — an ablation of the tissue's
  contribution, not a perturbation of it.
- The shuffle permutation matches `evaluate.py`'s (generator seed 1729) so
  probe output is directly comparable to training-time eval records.

## Contracts

- Read-only: loads checkpoints, never writes them.
- `freeze_*_delta` and `shuffle_*_delta` are BPC costs (positive = ablation
  hurts).
- Any claim of the form "population P matters" must cite the **freeze**
  delta; any claim of the form "population P is specialized/structured" must
  cite the **shuffle** delta. They are not interchangeable.
