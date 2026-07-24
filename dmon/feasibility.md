# feasibility.py

## Purpose

Answers "can this ecology support a body at all?" in about a second, with no training.

It exists because the first diffusion sweep spent 40 minutes on six values that were all
unsurvivable and returned six identical rows. Every one of `ARCHITECTURE.md` §4's three
design equations was violated by the defaults, and none of them needed a GPU to check.

The module simulates the **field alone** — no creature, no rule — and reports what any
creature would have to work with. `sweep.py` now calls it per value and refuses to train
inside an infeasible regime.

## Components

### `analyse`
- **Does**: field-only rollout; returns inflow, `n_max`, break-even `r*`, concentration
  at the seed, usable cell count, seed-to-source distance, diffusive reach, throttled
  fraction.
- **Note**: `throttled_fraction` is the one people miss. Once a source cell saturates at
  `field_cap`, raising `source_rate` changes *nothing* — the first parameter scan showed
  byte-identical reach for emission rates of 0.5 and 2.0, which looks like a bug and is
  physics.

### `verdict`
- **Does**: one line per equation, so a failure names itself and says what to change.

## Decisions

- **Field-only, deliberately.** Involving a rule would make the answer depend on
  training, and the question is precisely what the world offers *before* anyone learns
  anything. A negative here cannot be trained around.
- **`n_max` uses full-effort cost.** A cell paying only `maintenance` is far cheaper, so
  the true ceiling is higher — but a body of cells that never spend `effort_cost` earns
  nothing either. Full effort is the honest denominator for "how big a body can this
  world feed".
- **Reports `usable_cells`, not just concentration at the seed.** The seed test only
  applies at the start; a grown body lives wherever the field is above break-even, and
  the size of that territory is what bounds morphology. Cells outside it can still be
  fed through transport, which is the intended vasculature pressure.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `sweep.py` | `analyse(cfg, geom, grid, steps, spread=)`, `verdict(a, want_mass)` | Signatures |
| `substrate.py` | `make_sources(..., spread=)`, `cfg.light_cone_ok` | Signatures |

## Notes

- Run it before spending GPU time. That is the entire point of the file.
- It cannot tell you whether morphology will be *interesting*, only whether life is
  possible. A feasible ecology that produces a uniform mat is still an M0 failure.
- `spread` matters: check both ends of the curriculum. An infeasible final stage
  silently turns the last 40% of training into noise, which looks like a plateau rather
  than an error.
