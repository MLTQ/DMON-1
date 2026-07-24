# sweep.py

## Purpose

Sweeps `field_diffusion`, the knob `ARCHITECTURE.md` §M0 identifies as deciding whether
M0 can work at all. Too fast and the field is uniform — no gradient, no reason to reach,
a blob. Too slow and only cells touching a source survive. Neither end can produce
contingent morphology, so this runs *before* any result is interpreted.

Not a hyperparameter search. It is looking for a band, and the thing being optimised is
not a scalar — a run with high mass and no structure is a failure, not a good score.

## Components

### `sweep`
- **Does**: trains one rule per diffusion value, holding everything else — including the
  torch seed — fixed, so the only variable is the physics. Saves a checkpoint, a GIF and
  final descriptors per value.
- **Rationale**: fixing the seed is what makes this a controlled comparison rather than
  six unrelated runs. It is also why the sweep is *not* a substitute for the contingency
  noise floor, which needs the seed to vary.

### `contact_sheet`
- **Does**: final frames side by side, one panel per value, left to right in table
  order.
- **Rationale**: the output is deliberately two artefacts because neither is sufficient
  alone. The table is the verdict — descriptors decide, per `PROJECT.md`. But a
  descriptor table cannot distinguish a body that grew and stalled from one that never
  left the seed site, and those call for opposite responses.

## Decisions

- **`train()` gained a `cfg` parameter rather than the sweep mutating a global.** The
  ecology a rule was trained under is recorded in its checkpoint meta
  (`field_diffusion`), so a swept checkpoint cannot later be mistaken for a default one.
- **No labels burned into the contact sheet.** That would mean a font dependency for no
  analytical gain; panel order is in `sweep.json`.
- **Default range 0.02 → 0.80, log-ish spacing.** The discrete laplacian here is
  normalised by 16 with a centre weight of −12, so values approaching ~1.3 are
  numerically unstable. 0.80 is the top of the usable range, not an arbitrary stopping
  point.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_m0.py` | `train(..., cfg=None)` returns `(sub, history)` | Signature |
| `render.py` | `render_state`, `rollout_frames`, `to_gif` | Signatures |
| next step | `runs/sweep/sweep.json` carries per-value descriptors | Schema |

## Notes

- 6 values × 4k iters ≈ 40 min on the 4090 at 64×64/64 steps/batch 32 (0.097 s/iter,
  8.4 GB peak). The full 20-run contingency sweep is ~11 h at 20k iters — an overnight
  job, which is worth knowing before anyone economises on the noise floor.
- 4k iters is a *screening* budget. It is enough to tell blob from starved; it is not
  enough to interpret morphology. Pick the band here, then spend a real run inside it.
- Peak memory is rollout, not model — the 2070S at 8 GB cannot hold these settings.
  Lower `--batch` before `--grid`.
