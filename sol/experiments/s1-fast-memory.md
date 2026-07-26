# S1 reward-modulated fast synaptic memory

Date: 2026-07-26

## Question

Can delayed character-level reward improve the particular directed dendrites associated
with an event, across truncated-BPTT boundaries, without replacing slow gradient
learning or destabilizing the continuous field?

## Mechanism

- Every stream lane carries per-edge eligibility and bounded fast efficacy in
  `FieldState`.
- A target cell tags each named dendrite from local source activity and its own state
  change.
- The next observed character yields a bounded scalar reward. On the following tick,
  that reward changes only efficacy associated with retained edge tags.
- Fast efficacy decays and is added inside the slow edge's signed `tanh` weight.
- Slow parameters continue to learn by exact truncated BPTT.
- Checkpoints preserve all fast state and upgrade older organisms with zero-valued
  additive fields.
- A pending reward is consumed exactly once, including when it crosses an optimizer
  boundary.

## Protected contracts

Tests require delayed edge specificity, zero-reward invariance, boundedness, gradient
flow through fast efficacy, checkpoint persistence and backward compatibility, no
metabolic energy creation, single reward consumption, and aligned state shuffling.

The complete repository suite passes 32 tests. A 500-update deterministic micro-run
remained finite while reducing loss from 3.093 to 0.001; mean fast efficacy ended at
0.174 under a 0.25 absolute bound.

## Initial gain falsification

The first GPU run used gain 0.04, limit 0.75, and no metabolism, directly matching the
best S0 control. It tracked that control through 2,000 updates and reached 2.621 BPC at
update 1,750 versus 2.631 for the matched control: only a 0.010 improvement.

It then failed sharply near update 2,250:

- Mean fast efficacy had repeatedly approached 0.6 against the 0.75 bound.
- Held-out BPC moved from 2.692 at update 2,000 to 5.630 at update 2,250.
- Cell and edge credit magnitudes exploded while gradient clipping prevented non-finite
  parameters.
- Directed flow had risen to roughly three times the S0 level.

This falsifies broad high-gain potentiation as a stable memory rule.

## Stabilized rule

The revised local rule centers activity tags across each target's dendrite fan, making
eligibility competitive rather than allowing mostly positive reward to potentiate all
incoming edges together. The default gain is 0.02 and the fast-efficacy limit is 0.25.

The original gain runs remain useful negative controls. A clean long-run comparison of
the stabilized rule is still required before fast synaptic memory can be declared
successful or used to drive axon growth.
