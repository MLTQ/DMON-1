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
