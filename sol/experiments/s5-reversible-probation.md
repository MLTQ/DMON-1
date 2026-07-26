# S5 reversible post-graft probation

Date: 2026-07-26

## Motivation

S2–S4 show that useful anatomy emerges only through co-adaptation of the whole living
organism. Pre-graft local credit, repeated confirmation, and first-order global gradient
sign all fail to predict long-horizon developmental fitness reliably. S4's global sign
gate still installed all 15 possible grafts and worsened final-six mean BPC by 0.03048.

## Hypothesis

The relevant causal object is not an isolated candidate edge. It is the body after
learning with that edge for long enough to reorganize around it. A graft should therefore
remain reversible while the full organism experiences it, and its retention decision
should use observed post-graft prequential fitness.

## Mechanism

- One confirmed candidate at a time replaces its chosen incumbent provisionally.
- Before mutation, SOL backs up that slot's source, slow weight/bias, optimizer moments,
  structural credit/age, and every stream lane's edge eligibility and fast efficacy.
- All other weights, recurrent state, reward memory, probes, stream position, and
  ordinary backprop continue without interruption.
- Before grafting, a checkpointed EMA records the body's ordinary signed-reward trend.
- During probation, token-wise reward relative to each lane's surprise baseline is
  compared with that pre-graft developmental baseline.
- Positive mean improvement above the configured margin commits the graft.
- Otherwise the backed-up anatomical slot is restored exactly. Energy spent growing the
  failed graft is not refunded, and unrelated body adaptation is retained.
- Only one probation may be active. New structural decisions pause, but exploratory
  probe traffic continues.
- The probes-only control enters a virtual probation of identical duration without
  mutating topology.

## Protected contracts

1. exact slot rollback after its parameters, optimizer moments, and stream-local state
   have changed;
2. unrelated body adaptation survives rollback;
3. positive probation retains the adapted candidate;
4. virtual control never mutates permanent sources;
5. active-probation checkpoint resume produces the identical later decision;
6. reachability, fixed fan-in, energy provenance, and all S2–S4 controls remain.

## Experimental gate

The deterministic suite must exercise both commit and rollback. A natural CPU smoke must
finish with no active probation and demonstrate nontrivial developmental-baseline
telemetry before any GPU run. The first full comparison will use three confirmation
phases, a 300-update probation, no global sign gate, and enough total updates to resolve
the final provisional graft. It must report attempts, commits, and rollbacks separately.

Capability requires lower paired second-half and late-window mean BPC than the virtual
probes-only control. A lower best checkpoint or functional morphology alone is
insufficient for live promotion.

## CPU integrity smoke

The natural smoke used 16 cells/channels, three confirmation phases, 75 probation
updates, a 0.99 developmental-baseline decay, and 550 total updates so the final trial
resolved before summary.

| Variant | Attempts | Commits | Rollbacks | Final BPC | Birth BPC |
|---|---:|---:|---:|---:|---:|
| Virtual probes-only probation | 3 | 0 | 0 | 3.99328 | 3.99328 |
| Reversible graft probation | 3 | 0 | 3 | 3.99063 | 3.99063 |

All provisional topology was restored, so the growing organism's final persistent and
birth-topology evaluations are exactly identical. The last graft earned positive raw
reward during ordinary learning, but underperformed the body's pre-graft developmental
baseline by 0.00674 and was correctly rolled back. The virtual control's corresponding
developmental improvement was likewise negative by 0.00624.

The 0.00265 BPC intact difference is below the scale of a capability claim. The smoke
establishes that natural streamed experience can exercise rollback—not only forced unit
evidence—and that the body retains its learned experience after failed anatomy is
removed.

The full GPU run will keep the zero improvement margin. Unlike S2–S4, its primary
mechanistic question is whether any provisional anatomy improves on the organism's own
ongoing developmental trend strongly enough to survive probation.
