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

### `contingency`
- **Does**: Trains one rule per geometry, cross-evaluates each rule under every other
  geometry, reports between-geometry spread.
- **Rationale**: Cross-evaluation separates two very different outcomes — a rule that
  *learned a shape* will carry it to a new field, whereas a rule that learned to
  *respond to gradients* will produce a different body there. The second is what M0
  is looking for.

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
| CLI | `--geom --iters --grid --steps --batch --lr --device --contingency` | Flag names |
| `PROJECT.md` M0 | contingency spread reported against within-geometry noise | Metric definition |

## Notes

- **The contingency test is incomplete as written, in two separate ways.**
  1. It reports between-geometry spread but does not measure the within-geometry
     baseline that spread must exceed. A test that cannot fail is not a test.
     **That baseline must come from independent *training* runs** — 5 seeds × 4
     geometries = 20 runs, not 4 — because re-evaluating one rule only samples firing
     jitter (see `evaluate` above), which will give a floor far too low to fail against.
     `--seed` exists for this.
  2. The cross-evaluation results are computed (`results[g][h]`) and then **never used**
     — `spread` is taken only over the `self` entries. Cross-evaluation is the half that
     catches the *dangerous* failure, a rule that memorised a shape and ignores the
     field. Currently it is collected and discarded.
- **Record the whole descriptor point cloud, not just the verdict.** Its dimensionality
  is what the bridge to M3 rests on; see `ARCHITECTURE.md` §M0. It costs one array and
  an SVD on a run that has to happen anyway.
- **The light cone binds on the eval horizon.** Training samples rollout length from
  `[steps//2, steps]`, so roughly half of all training rollouts violate it by design —
  that is a curriculum and is fine. `train()` warns if `steps < grid`.
- Smoke run (300 iters, 24×24, CPU): mass 1 → 23, box dim → ~1.54. This establishes
  that the objective is learnable and the gradient path is intact. It establishes
  nothing about contingency; the grid is too small and the run far too short.
- Real runs want the 4090: 64×64 grid, ≥64 steps (light-cone), batch 32, 20k+ iters.
  Memory cost is the BPTT rollout, not the model — reach for gradient checkpointing
  before reaching for a smaller grid.
