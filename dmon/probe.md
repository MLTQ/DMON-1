# probe.py

## Purpose

Tests the assumption that everything above M2 silently depends on: **does internal
metabolic state have a visible signature?**

M2 wants mood readable from outside. M3's handicap wants a display that degrades when
the body is failing. M4 wants speech that goes urgent because cells are starving. Those
are one claim wearing three costumes, and until this file existed none of them had been
tested. It is the cheapest falsification in the project and it sits under three
milestones at once.

It needs **one** trained rule, not the twenty the contingency test needs, so in
wall-clock it runs first despite being conceptually later.

## Components

### `run_probe`
- **Does**: grow → hold fed → ramp source supply to zero → hold starved, recording
  per-sample descriptors and mean energy every step.
- **Rationale**: **ramp, not cut.** A cut conflates shock with starvation; the question
  is whether shape tracks condition, not whether it reacts to a discontinuity.
- **Interacts with**: `checkpoint.load`, `render.render_state` (optional GIF).

### `_analyse`
- **Does**: per-sample fed-state band, death step, first sustained departure, warning
  time.
- **Rationale**: each batch element is an independent run of the experiment, so the
  report is a distribution rather than a single number. A median warning time over 8
  samples is a claim; one sample is an anecdote.

### `verdict`
- **Does**: turns the numbers into PASS / WEAK / FAIL / INCONCLUSIVE.
- **Rationale**: the probe is explicitly allowed to return INCONCLUSIVE. A run where
  nothing died measured nothing, and should say so rather than reporting a warning time
  computed from a death that never happened.

## Decisions

- **`warning_shape` is the verdict, not `warning_any`.** Mass and gyration departures
  are near-tautological — starvation removes cells, so a shrinking body is not evidence
  that *form* tracks condition. Only compactness and box dimension are scale-insensitive
  enough to carry the claim. Both are reported; only the shape one decides.
- **The band is floored at 1% of the fed-state mean.** A descriptor that sits perfectly
  flat while fed would otherwise make any later wobble "significant". The floor widens
  the band, which makes the test *harder* to pass — the correct direction for a test
  whose entire purpose is to be capable of failing.
- **Departures must persist 3 steps.** The substrate fires stochastically
  (`fire_rate=0.5`); single-step excursions are firing noise, not signal.
- **Death is relative, not absolute**: mass below 5% of its own fed-state baseline. An
  absolute threshold would mean something different at every grid size.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `substrate.py` | `descriptors_per_sample()` returns `(B,)` tensors | Return shape |
| `checkpoint.py` | `load()` returns `(Substrate, meta)`; meta may carry `geom` | Return arity |
| `ARCHITECTURE.md` §M2 | verdict keyed on `warning_shape` | Metric definition |

## Notes

- Reading a `WEAK`/`FAIL` from an undertrained rule means nothing. The probe assumes a
  rule that has reached steady state under its ecology; run it on a real checkpoint.
- If `survived_ramp == reps`, the creature is living off transport reserves and `hold`
  is too short. That is a run configuration error, not a result.
- The GIF is the thing that catches setup mistakes. A starvation transition read purely
  off four scalars is exactly where you talk yourself into a trend that is not there.
