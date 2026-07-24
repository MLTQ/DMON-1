# train_m0.py

## Purpose

Trains a `Substrate` rule against sustained living mass and nothing else, then runs
the M0 verdict: whether morphology is *contingent on ecology*. Success is not "it
looks like a creature" — it is that different source geometries produce measurably
different bodies.

## Components

### `Pool`
- **Does**: Growing-NCA style sample pool. Short BPTT windows compose into long
  horizons; the worst sample in each draw is reset to seed.
- **Rationale**: Resetting the worst (not a random one) keeps the rule able to start
  from nothing, which is otherwise the first skill it forgets.

### `train`
- **Does**: Adam + cosine schedule, variable rollout length per iteration, grad clip.
- **Rationale**: Variable rollout length stops the rule from timing its behaviour to
  a fixed horizon.

### `evaluate`
- **Does**: Fresh seeds, no gradient, returns descriptors.
- **Caveat**: `seed()` is deterministic — one live cell, empty field — so the only
  stochasticity here is the firing mask. This measures firing jitter, **not** run-to-run
  variance, and must not be used as the contingency noise floor. See Notes.

### checkpointing
- **Does**: `--ckpt` writes weights + config + training meta periodically and at the
  end, via `checkpoint.py`.
- **Rationale**: the probe and renderer both consume checkpoints, and a 20k-iteration
  run that cannot be resumed or re-examined is a run you have to repeat.

### `contingency` — **moved to `contingency.py`**
- The version that lived here could not fail, in two separate ways: no noise-floor
  baseline, and cross-evaluation results computed then discarded. Both are fixed in
  `contingency.py`, which also keeps the trainer from being the thing that grades
  itself. See `contingency.md`.

## Decisions

- **Loss is `-mean(mass)/grid²`** — normalised so learning rates transfer across grid
  sizes.
- **Rejected any morphological regulariser** (symmetry, compactness, connectivity
  priors). Any such term reintroduces target supervision through the back door. If
  the creature needs a prior to be interesting, the ecology is wrong, not the loss.
- **Descriptors are the verdict, renders are not.** Renders are for intuition only;
  the M0 pass/fail is a number.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CLI | `--geom --iters --grid --steps --batch --lr --device --ckpt --seed --log` | Flag names |
| `contingency.py`, `sweep.py` | `train(..., seed=, cfg=)` returns `(sub, history)` | Signature |

## Notes

- **The light cone binds on the eval horizon.** Training samples rollout length from
  `[steps//2, steps]`, so roughly half of all training rollouts violate it by design —
  that is a curriculum and is fine. `train()` warns if `steps < grid`.
- Smoke run (300 iters, 24×24, CPU): mass 1 → 23, box dim → ~1.54. This establishes
  that the objective is learnable and the gradient path is intact. It establishes
  nothing about contingency; the grid is too small and the run far too short.
- Real runs want the 4090: 64×64 grid, ≥64 steps (light-cone), batch 32, 20k+ iters.
  Memory cost is the BPTT rollout, not the model — reach for gradient checkpointing
  before reaching for a smaller grid.
