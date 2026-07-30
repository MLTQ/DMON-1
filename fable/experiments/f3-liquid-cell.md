# F3 (DRAFT): liquid cell rule head-to-head

*Draft until the F0 and F2 verdicts are recorded; geometry and bars inherit
from F0's outcome. Filed 2026-07-30 after discussing Liquid Neural Network
lessons (Max's steer: stability, timescales, and the composability
precondition).*

## Question

Does replacing the shared GRUCell with a CfC-style liquid cell — leaky
integration toward an input-defined target with a **learned, bounded,
input-dependent time constant** — improve the organism on any of the three
axes it is currently weakest on?

1. **Stability**: the liquid update is contractive by construction (per-step
   factor |1 − Δt/τ| < 1 with τ bounded), so state boundedness and tame
   through-time Jacobians are properties of the operator, not of guards.
   Fable's current stack is clamp + clip + skip-counter around a GRU whose
   128-step backward overflowed twice this week ("check the operators, not
   the masks", applied to dynamics).
2. **Timescales**: per-cell input-dependent τ lets tissue self-organize into
   fast and slow populations, replacing two hand-set timescales
   (steps_per_token; the fixed-delay mirror ring's dominance of ResetΔ).
3. **Adaptability**: the LNN literature's headline result is robustness under
   distribution shift — exactly F2's savings/interference axis. F3 therefore
   reports F2 metrics for the liquid arm, not only BPC.

## Design sketch (to be finalized)

- `LiquidRule` drop-in for `SharedRule`: h ← h + (Δt/τ(x,h))·(g(x,h) − h),
  τ = τ_min + (τ_max−τ_min)·σ(·), closed-form (no ODE solver), Δt=1 for now
  (Δt-awareness is the composability precondition and gets its own test).
- Arms at F0-winning geometry, 3 seeds: gru-rule (incumbent) vs liquid-rule,
  identical stream/schedule/params-matched (τ/g networks sized so totals
  match within 1%). Matched external GRU baseline rides along as always.
- Report: BPC gap, skipped-update counts + gradient-norm trajectories
  (stability margin), reset/shuffle/mirror deltas, and an F2 cycled run for
  both rules.

## Priors, stated before running

- sol's history: biologically-motivated mechanisms usually arrive
  capability-neutral. The liquid cell must *earn* its place on stability or
  adaptability even if BPC is a wash; BPC regression > 0.05 at matched
  budget rejects it regardless of elegance.
- LNN evidence is strongest in control/time-series at small scale; char-LM
  evidence is thin. This is a hypothesis test, not an adoption.

## Why this is on the roadmap at all

The composability goal (grafting a second organism; organs as grown index
sets) requires per-tissue rules and Δt-aware cells for clock-mismatched
composition. The liquid cell is the smallest step that de-risks the Δt
half; the per-tissue-rule refactor is the other half and is tracked
separately.
