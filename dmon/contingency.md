# contingency.py

## Purpose

The M0 verdict: **is morphology contingent on ecology?**

Replaces the version that lived in `train_m0.py`, which could not fail. Split into its
own module so the trainer is not also the thing that grades itself.

## Components

### `run`
- **Does**: trains `len(geoms) × len(seeds)` rules, evaluates every rule under every
  geometry, and reports separation, responsiveness and morphospace dimensionality.

### `descriptor_vector`
- **Does**: one `(4,)` vector per (rule, geometry).

### `effective_dim`
- **Does**: participation ratio of the whole descriptor point cloud.
- **Rationale**: required by `ARCHITECTURE.md` §M0 — it is what the bridge to M3 rests
  on. ~1 means the ecology admits a single axis of variation, the body is barely
  steerable, and a display would have to carry all expressivity. ~3–4 means the body
  itself has room to be selected over.

### `verdict`
- **Does**: PASS / FAIL(not contingent) / FAIL(memorised) / INCONCLUSIVE.

## Decisions

- **The noise floor comes from retraining, not re-evaluation.** `seed()` is
  deterministic — one live cell, empty field — so repeated evaluation of one rule
  samples only the firing mask and gives a floor far too low to fail against. This costs
  20 training runs instead of 4 and there is no cheaper honest version.
- **Everything is reported in units of that noise floor.** Descriptors have wildly
  different scales (mass in the hundreds, box_dim near 1), so a raw Euclidean spread is
  just a mass measurement wearing a disguise. Each descriptor is divided by its
  across-seed standard deviation first, which makes a separation of 1.0 mean "exactly as
  different as two runs of the same experiment" — i.e. nothing.
- **Cross-evaluation is used, not merely computed.** The old version calculated
  `results[g][h]` and then dropped it from the report. That is the half that catches the
  dangerous failure: a rule that memorised a shape looks exactly like one that learned
  to read gradients, and only moving it to a field it never trained in separates them.
  `responsiveness` is that measurement, in the same noise units.
- **Degeneracy guards run before any ratio is believed.** Every headline number is a
  quotient by the noise floor, so a collapsed floor manufactures enormous confidence out
  of nothing. This is not hypothetical — the first smoke test of this module returned a
  confident `PASS — separation 34x noise` on four rules whose bodies were each a single
  dead seed cell. Two guards:
  - median self-evaluated mass < 50 cells → INCONCLUSIVE. `HANDOFF.md` already states
    that descriptors degenerate below ~50 occupied cells; the verdict now enforces it
    rather than trusting the reader to remember.
  - any descriptor with zero variance across independently trained rules → INCONCLUSIVE,
    because either training collapsed to one degenerate solution or the seeds are not
    actually independent.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_m0.py` | `train(..., seed=, cfg=)` returns `(sub, history)` | Signature |
| `substrate.py` | `descriptors_per_sample()` returns `(B,)` tensors | Return shape |
| `PROJECT.md` M0 | separation and responsiveness both in noise units | Metric definition |

## Notes

- ~11 h for the full 4 geometries × 5 seeds × 20k iters on the 4090. An overnight job,
  which is worth knowing before anyone economises on the noise floor.
- Thresholds (1.5× noise for both separation and responsiveness) are a judgement call,
  not a derived quantity. They are deliberately low: the point is to catch results that
  are *indistinguishable from noise*, not to declare a strong effect.
- A PASS here is necessary and not sufficient for M0. The renders still have to be
  looked at — a rule can satisfy every ratio and still be doing something silly that
  four scalars cannot see.
