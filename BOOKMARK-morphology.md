# Bookmarked: the morphology / resource-field work

Everything in `dmon/` other than the streaming core belongs to the **display organ**, not
to the creature. It was built under an earlier reading in which a grown body on a
resource field *was* the organism. That reading was wrong — see `ARCHITECTURE.md` §1 —
but the work is sound and returns at **S4**.

This file exists so the next session does not either (a) resume building it as if it
were the main line, or (b) delete it as dead code.

## What is here and what state it is in

| File | State |
|---|---|
| `substrate.py` | Resource-field physics, metabolic ledger, gated + residual cells, descriptors. **The gated cell has already moved to the main line** — it survives the reset and matters more there. |
| `train_m0.py` | Episodic trainer, sample pool, distance curriculum, fresh-seed logging. Episodic by construction; does not transfer. |
| `contingency.py` | Morphology-contingency verdict with a retraining-derived noise floor, cross-evaluation, degeneracy guards. |
| `baseline.py` | Best-constant-policy null model + lever-authority measurement. **The pattern transfers; the specifics do not.** |
| `feasibility.py` | The three ecology design equations, runnable in ~1s. |
| `probe.py` | Legibility probe — does metabolic state have a visible signature. Directly relevant at S4. |
| `render.py` | Diagnostic renderer. Relevant at S4. |
| `sweep.py` | Diffusion-length sweep with a feasibility gate. |
| `checkpoint.py` | Config-with-weights persistence. Transfers as-is. |
| `test_conservation.py` | Energy conservation tests. **The discipline transfers immediately** — any energy pool needs its equivalent. |

## What was established, so it is not rediscovered

- **Energy conservation is not automatic.** The original transport operator let the rule
  mint currency through a conductance it controlled; a trained rule grew a stable body
  on a grid containing no food. Fixed by conservative pairwise flux exchange. Tested.
- **The ecology could support about four cells** before recalibration. Three design
  equations (supply vs demand, diffusive reach, break-even margin) are in
  `feasibility.py` and were all violated by the original defaults.
- **A fourth**: the horizon must exceed passive starvation time, or doing nothing wins
  and the mass objective rewards inaction.
- **Source geometries must be supply-matched.** Unnormalised they differed by 248x in
  total food, so a contingency test would have compared abundance, not arrangement.
- **The M0 pass condition does not discriminate learning from physics.** "Move the
  sources, get a different morphology" is satisfied by a null model with zero learning.
  Any resumption needs the best-constant baseline as its control, not a dead grid.
- **Lever authority is HIGH** (mass spans 0→52 from a seed, 36x dense), so the two
  levers do steer. A rule that cannot beat constant is failing at computation.

## Open when it resumes

- Residual vs gated cell head-to-head was launched and never completed.
- The diffusion sweep's fixed-curriculum confound (see `sweep.md`).
- Whether the creature should be seeded *at* a source rather than at the grid centre —
  the last unresolved design question before the reset.
