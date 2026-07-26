# S4 organism-level structural fitness

Date: 2026-07-26

## Motivation

S3 reduced rewrites from 46 to 15 and improved every aggregate comparison with S2, but
still lost paired second-half and late-window mean BPC. Three consecutive local credit
confirmations approved anatomy that was directly harmful at update 2500. More waiting
cannot repair a selection signal that is misaligned with the organism's output.

## Hypothesis

A probe should become anatomy only when two independent memories agree:

1. delayed signed reward repeatedly associates its causal activity with useful events;
2. the probe's measured hidden-state intervention points in a direction that reduces
   the exact global sequence loss.

The first signal spans optimizer windows. The second connects candidate traffic directly
to language fitness through the live organism's ordinary backpropagation.

## Mechanism

The existing with/without-probe cell-rule intervention yields a detached hidden-state
delta for every target and token. After normal loss backward, S4 computes:

`fitness[target] = mean(-dLoss/dHidden * probe_delta)`

The hidden gradient contains the probe's effect on current output and downstream
computation within the uninterrupted BPTT window. A positive value is the first-order
prediction that adding the measured probe traffic lowers organism-level loss.

Fitness is retained with the same EMA as local structural credit. With
`--structural-global-fitness`, a candidate earns a confirmation only when local signed
credit clears its incumbent margin and global fitness is positive. Adverse evidence from
either memory resets the streak.

Both arms receive identical probes, loss gradients, local reward, persistent state,
candidate holds, and rotation. The control still freezes only permanent installation.

## Deterministic gate

- measured probe fitness is finite, nonzero, and target-addressed after backward;
- negative global fitness vetoes otherwise positive local structural credit;
- positive local and global evidence can still install a viable connection;
- fitness buffers survive exact resume and upgrade older checkpoints additively;
- every S2/S3 topology, energy, optimizer, and living-control contract remains.

No checkpoint promotion follows from mechanism behavior alone.

The full repository suite passes 42 tests.

## CPU integrity smoke

The matched 16-cell, three-confirmation smoke changes only the global-fitness
requirement from S3.

| Variant | Rewrites | Final BPC | Birth-topology BPC | Reachability |
|---|---:|---:|---:|---:|
| Probes only | 0 | 4.25434 | 4.25434 | 16/16 cells, 4/4 outputs |
| Local + global fitness | 1 | 4.27272 | 4.27774 | 16/16 cells, 4/4 outputs |

Global gating rejects one of the two connections S3 installed at this scale. The one
accepted connection has a small positive causal effect (+0.00503 BPC when removed), but
overall BPC is 0.01838 worse than control. The smoke validates selectivity and mechanism,
not capability.

## Full-scale gate

The first GPU comparison keeps S3's 64-cell, seed-7, 5,000-update, three-confirmation
budget and changes only the global-fitness requirement in both arms. The growing arm
must:

1. remain finite, stable, and completely reachable;
2. reject at least one full confirmation cycle and install no more than S3's 15 edges;
3. improve paired second-half and final-six mean BPC over the living probes-only control;
4. retain a positive late birth-topology ablation penalty;
5. avoid an S3-style checkpoint where confirmed anatomy explains nearly the entire
   paired deficit.

No live promotion follows from best BPC alone.

## Full-scale result

Both organisms completed stably with full reachability. The global-fitness arm installed
15 connections: every complete confirmation cycle still found at least one target whose
EMA fitness was positive. The zero-margin sign gate therefore provided no decision-level
selectivity relative to S3.

| Statistic | Global-fitness growth | Probes only | Growth − control |
|---|---:|---:|---:|
| Final BPC | 2.60450 | 2.50610 | +0.09840 |
| Mean BPC, all 20 paired evaluations | 2.71155 | 2.69246 | +0.01910 |
| Mean BPC, updates 2500–5000 | 2.59171 | 2.56235 | +0.02936 |
| Mean BPC, updates 3750–5000 | 2.55713 | 2.52665 | +0.03048 |

Lower is better. Growth won 5 of 20 checkpoints, 1 of 11 in the second half, and 1 of
the final 6. The paired difference ranged from 0.02505 BPC better to 0.09840 worse.

The final learned anatomy was functional: birth-topology ablation worsened growth from
2.60450 to 2.63015 (+0.02565 BPC), and the final-six mean morphology penalty was
+0.03400. Functional anatomy again failed to imply a better organism.

## Verdict

The first-order global signal is measurable and useful as telemetry, but positive sign
alone is neither scarce nor predictive enough in a 64-target search. It sometimes avoids
an acute harmful graft—S4's update-2500 anatomy was beneficial where S3's was directly
harmful—but produces a worse overall and late developmental trajectory.

S4 fails both explicit gates:

- it rejects zero full confirmation cycles and matches S3's 15 rewrites;
- it loses second-half and final-six mean BPC by larger margins than S3.

No checkpoint is promoted. A further pre-graft proxy is not justified by these results.
The next experiment should make grafts reversible: preserve the incumbent source during
a bounded post-graft probation, allow the whole organism to co-adapt under ordinary
traffic and learning, then retain or roll back the anatomical change using its observed
prequential fitness. This directly measures the body-with-organ developmental response
the pre-graft local and gradient proxies failed to predict.
