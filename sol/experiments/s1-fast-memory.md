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

The complete repository suite passes 34 tests. A 500-update deterministic micro-run
remained finite while reducing loss from 3.093 to 0.001; mean fast efficacy ended at
0.174 under a 0.25 absolute bound.

## Initial gain falsification

Both initial GPU runs used gain 0.04 and limit 0.75.

| Variant | Best BPC | Best update | Final BPC | Outcome |
|---|---:|---:|---:|---|
| Fast memory, no metabolism | 2.621 | 1,750 | 6.528 | Failed sharply near update 2,250 |
| Fast memory, metabolism | 2.466 | 3,500 | 12.214 | Metabolism delayed failure until after update 3,750 |

The no-metabolism run directly matched the best S0 control. It tracked that control
through 2,000 updates and reached 2.621 BPC at update 1,750 versus 2.631 for the matched
control: only a 0.010 improvement. It then failed sharply near update 2,250:

- Mean fast efficacy had repeatedly approached 0.6 against the 0.75 bound.
- Held-out BPC moved from 2.692 at update 2,000 to 5.630 at update 2,250.
- Cell and edge credit magnitudes exploded while gradient clipping prevented non-finite
  parameters.
- Directed flow had risen to roughly three times the S0 level.

This falsifies broad high-gain potentiation as a stable memory rule.

Metabolism damped the field long enough to produce a more informative transient result.
Its best checkpoint reached 2.466 BPC: 0.018 better than the default S0 organism, but
still 0.015 worse than the no-metabolism S0 live checkpoint. It then moved to 2.515 at
update 3,750, 8.040 at update 4,000, and 12.214 at the final update. Metabolism delayed
the same failure rather than solving it.

Neither initial fast-memory checkpoint is eligible for live promotion: the existing
no-metabolism checkpoint remains better and stable.

## Stabilized rule

The revised local rule centers activity tags across each target's dendrite fan, making
eligibility competitive rather than allowing mostly positive reward to potentiate all
incoming edges together. The fast-efficacy limit is 0.25.
A scaled `tanh` homeostat contracts efficacy smoothly near the bound instead of leaving
many synapses pinned by a hard clamp.

Each stream lane also retains a moving expectation of character surprise. Reward is
signed relative to that baseline: better-than-expected events potentiate their tagged
edges and worse-than-expected events depress them. The selected defaults are baseline
decay 0.99 and plasticity gain 0.04.

## Matched CPU-scale Shakespeare comparison

The refinement was tested on the same Tiny Shakespeare split with 32 cells, 32 channels,
6 dendrites, 4 sensory cells, 4 output cells, 2 message steps, seed 13, no metabolism,
and 2,000 updates × 8 lanes × 32 characters = 512,000 training characters. All variants
had 32,002 trainable parameters.

| Variant | Best BPC | Final BPC | Final regression | Final saturation |
|---|---:|---:|---:|---:|
| No fast efficacy | 2.964 | 2.969 | +0.004 | 0% |
| Competitive tags, hard bound | 2.945 | 2.945 | +0.000 | 33% |
| Competitive tags, smooth homeostat | **2.907** | **2.921** | +0.014 | **0%** |

The smooth rule beat the no-fast control at every validation from update 1,000 onward.
It improved best BPC by 0.058 and final BPC by 0.047 while passing the completed-run
stability guard. The hard-bound rule also improved the endpoint, but accumulated roughly
one third of its fast weights at the cap.

An inference-time causal ablation on the smooth rule's best checkpoint preserved the
same model, edge tags, reward path, and recurrent state policy while forcing fast
efficacy to zero before every tick. Held-out BPC worsened from 2.907 to 2.990
(+0.083). The trained organism therefore uses online synaptic memory directly; the gain
is not only an incidental optimization effect.

## Signed-reward replication and gain selection

The first smooth result used reward relative to chance and proved that the pathway could
help, but two additional seeds showed a seed-sensitive endpoint. Reward was therefore
centered around each lane's recent surprise expectation and all models were retrained
from scratch.

At gain 0.02, signed fast memory improved final BPC on every 1,500-update seed:

| Seed | No-fast best/final | Fast best/final | Final improvement |
|---:|---:|---:|---:|
| 7 | 3.002 / 3.021 | 2.980 / 2.988 | +0.033 |
| 13 | 3.001 / 3.087 | 3.001 / 3.051 | +0.035 |
| 21 | 2.979 / 3.046 | 2.948 / 3.000 | +0.047 |

A seed-13 gain sweep selected 0.04 over 0.01, 0.02, and 0.08. Gain 0.08 learned faster
early but overshot late; 0.04 retained zero saturation. Replicating 0.04:

| Seed | No-fast best/final | Gain-0.04 best/final | Final improvement |
|---:|---:|---:|---:|
| 7 | 3.002 / 3.021 | 3.012 / 3.013 | +0.008 |
| 13 | 3.001 / 3.087 | 2.957 / 3.033 | +0.054 |
| 21 | 2.979 / 3.046 | 2.952 / 2.982 | +0.064 |

Mean final improvement is 0.042 BPC. All three runs pass the stability guard, remain at
zero saturation, and worsen when only fast efficacy is removed from their best
checkpoints (+0.001, +0.004, and +0.001 BPC). The direct tick-time contribution is
small but consistently positive; the larger training delta indicates that continuous
fast state also shapes the slow learner's trajectory.

This was positive reduced-scale evidence for event-specific fast synaptic credit and
motivated the original-budget comparison below. The full-scale result, rather than this
smaller sweep, determines the S1 conclusion.

## Full-scale GPU confirmation

The selected rule and a centered-reward no-fast control were trained from scratch at the
original 64-cell, 64-channel budget: 5,000 updates × 16 lanes × 32 characters =
2.56 million training characters. Both used seed 7, eight dendrites, three message
steps, and no metabolism so the S0 capability winner remained the exact comparison.
The candidate ran on the RTX 4090 and the control ran independently on the RTX 2070
Super.

| Variant | Best BPC | Final BPC | Worst post-best regression | Final saturation |
|---|---:|---:|---:|---:|
| Centered reward, no fast efficacy | 2.44185 | **2.53440** | +0.10863 | 0% |
| Centered reward, smooth fast efficacy | **2.43925** | 2.55127 | +0.15154 | **0%** |

Both uninterrupted runs pass the 0.5-BPC stability guard. The fast candidate improves
best BPC by 0.00259 over its matched control and by 0.01163 over the prior 2.45088-BPC
live checkpoint. Both candidates reach their best at update 3,750.

At the fast candidate's best evaluation, forcing only fast efficacy to zero worsens BPC
from 2.43925 to 2.44200 (+0.00275). The online synaptic state therefore carries a small
amount of predictive information at the winning checkpoint. Its contribution changes
sign across the full training history, however, and the candidate finishes 0.01687 BPC
worse than the no-fast control.

The 0.00259 best-BPC difference is only about 0.1%, comes from one topology/training
seed and a 2,048-character scoring window, and is selected as the minimum of 20
evaluations. It is not credible evidence of a capability improvement. A seed-13
replication was started, then stopped once this effect-size assessment made clear that
another expensive near-tie would not change the next engineering decision.

S1 therefore closes as a stable but capability-neutral negative result. The candidate
remains the operational live checkpoint because it is stable and has the lowest
measured BPC, not because fast memory has beaten control. Credit-guided axon growth
proceeds as a separate hypothesis and must earn its own matched improvement; it does
not inherit a positive claim from S1.

## Live checkpoint policy

Promotion now audits the complete validation history. A candidate is excluded when its
final or worst post-best BPC regresses by more than 0.5. Applied to the real artifacts:

- The smooth centered-reward candidate is stable with 0.152 worst post-best regression.
- Its 2.439-BPC best checkpoint ranks ahead of the 2.442 matched no-fast control and
  the 2.451 S0 no-metabolism checkpoint, but only by an unresolved single-seed margin.
- High-gain fast memory with metabolism is rejected at 9.748 regression.
- High-gain fast memory without metabolism is rejected at 3.948 regression.

The local UI uses the stable smooth fast-memory checkpoint from update 3,750 as an
operational best-measured snapshot, not as a claim that fast memory has beaten control.
