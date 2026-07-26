# S3 selective structural consolidation

Date: 2026-07-26

## Motivation

S2 proved that continuous causal probes can grow anatomy the neural field uses, but its
consolidation rule rewired in all 46 eligible phases. The growing organism became deeply
dependent on its new morphology while performing worse than the probes-only organism on
average. Instantaneous positive local credit was consequential but not selective enough
to serve as a fitness proxy.

## Hypothesis

A candidate source that remains better than a target's weakest mature incumbent across
several independent reward windows is less likely to be a transient fluctuation.
Requiring consecutive confirmation phases should reduce topology churn while preserving
continuous exploratory traffic and ordinary learning.

## Mechanism

- A candidate probe remains active for `confirmation_phases` structural intervals.
- At each interval it earns one confirmation only when its retained signed credit is
  positive and exceeds the weakest mature incumbent by the configured margin.
- One adverse or insufficient interval resets that target's confirmation streak.
- Installation is possible only at the end of the shared hold period and only for
  candidates whose streak spans every required phase.
- All candidates then rotate, whether or not any rewrite occurred.
- The probes-only control uses the identical hold, evidence, reset, and rotation schedule
  but never mutates permanent sources.

`confirmation_phases=1` preserves the S2 policy and checkpoint behavior. Confirmation
streaks are additive checkpoint buffers so an older organism loads with zero provisional
evidence rather than losing any permanent anatomy.

## Deterministic contracts

- no permanent topology or probe rotation before the required confirmation count;
- installation after three consecutive positive phases;
- one negative phase resets a partially accumulated streak;
- exact checkpoint resume includes confirmation state;
- older checkpoints gain zero confirmation state additively;
- all S2 reachability, fan-in, optimizer, energy, and graft-integrity guards remain.

The full repository suite passes 40 tests.

## CPU integrity smoke

The matched smoke used the S2 16-cell configuration and three confirmation phases.

| Variant | Rewrites | Final BPC | Birth-topology BPC | Reachability |
|---|---:|---:|---:|---:|
| Probes only | 0 | 4.25434 | 4.25434 | 16/16 cells, 4/4 outputs |
| Three-phase consolidation | 2 | 4.24933 | 4.25758 | 16/16 cells, 4/4 outputs |

This scale cannot establish a capability gain. It does show the intended selectivity:
the prior smoke installed eight connections, while consecutive evidence installs two,
and those two connections already have a small positive causal effect.

## Full-scale gate

The first GPU comparison keeps the S2 64-cell, 5,000-update, seed-7 budget and changes
only `confirmation_phases` from one to three in both arms. Selective consolidation must:

1. preserve finite training and complete sensory/output reachability;
2. install materially fewer than S2's 46 rewrites;
3. outperform its matched probes-only control on paired second-half and late-window
   mean BPC, not only one favorable checkpoint;
4. show a positive birth-topology ablation penalty;
5. remain within the completed-run stability guard.

No live checkpoint promotion follows from rewrite selectivity or morphology alone.
