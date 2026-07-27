# S21: Reward-addressed backward routing memory

## Question

Can a branch learn whether routing decoder credit through a remembered event improved a
later prediction?

S19 routed from instantaneous reverse-vector/source-memory alignment but was too weak.
S20 increased that alignment to a material, non-saturating scale; it changed reverse
traffic and anatomy yet regressed capability. Together they reject amplitude as the
missing ingredient. Neither treatment assigned later fitness back to the earlier
routing choice.

## Mechanism

S21 adds two parameter-free, per-stream states to each fixed dendrite slot:

1. a **routing eligibility trace** remembers the relative signed alignment between
   source event memory and decoder credit that actually crossed that branch;
2. a bounded **routing preference** changes only when later signed prequential reward
   meets that trace.

Positive and negative reward reinforce the remembered routing choice in opposite
directions. The preference is competitive within each target-owned dendrite fan and
allocates future decoder credit with fan-in-normalized softmax weights. Dormant slots
carry neither trace nor preference. Spawned, pruned, replaced, rolled-back, shuffled,
and checkpointed anatomy must treat these states as part of the branch.

This is a local three-factor loop:

```text
forward participation → backward corrective alignment → delayed reward
```

It adds persistent memory but no learned parameters, global teaching graph, frozen
organ, or second organism. Ordinary truncated BPTT and uninterrupted exploratory
traffic remain active.

## Preflight

Before GPU capability work:

- the default-disabled path reproduces historical behavior;
- zero reward cannot create a routing preference;
- delayed positive and negative reward move only tagged active branches in opposite
  directions;
- preference remains bounded and competitive within each target fan;
- equal preference reproduces historical reverse transport;
- dormant and repurposed slots carry no stale routing memory;
- checkpoint, active probation, evaluation shuffling, and old-checkpoint compatibility
  preserve their scientific contracts;
- parameter count is unchanged;
- reduced matched curves demonstrate finite, nonzero routing state without instability.

The first capability comparison uses the S17 organism as control. Treatment enables the
reward-plastic routing flag and uses alignment scale `100`, the smallest S20-calibrated
scale that resolved real branch differences without saturating them. In S21 that scale
tags an event for later reward; it no longer routes credit directly.

One delayed-reward update was replayed from each final S20 checkpoint to calibrate the
new preference gain:

| Preference gain | Seed 7 routing change | Seed 13 | Seed 21 | Any saturation |
|---:|---:|---:|---:|---:|
| 0.3 | 0.91% | 0.50% | 0.19% | no |
| 1.0 | 2.42% | 1.46% | 0.59% | no |
| 3.0 | 4.34% | 2.45% | 1.27% | yes |

Preference gain `1` is therefore the preregistered treatment: it makes one
reward-addressed event comparable to S20's material routing scale without immediately
saturating any seed.

Run seeds 7/13/21 for a reduced decision horizon, graph every validation, and advance
to a full 2,000-update RTX 4090 trial only if the mechanism is active and the paired
ordering is not already adverse.

## Results

### 1,000-update continuation gate

All three RTX 4090 services completed successfully and used the same data, parameter
budget, optimizer, anatomy, exploratory-traffic policy, evaluation windows, and seeds
as S17. The only scientific differences were the preregistered reward-plastic routing
state and its calibrated scales.

The mechanism was active without meaningful saturation:

| Seed | Mean routing eligibility | Mean routing preference | Preference saturation |
|---:|---:|---:|---:|
| 7 | 0.004483 | 0.007768 | 0.00153% |
| 13 | 0.003282 | 0.005356 | 0% |
| 21 | 0.000642 | 0.002368 | 0% |

Against the corresponding S17 organisms, the final three preflight validations
(updates 500/750/1,000) produced:

| Seed | Mean S21 − control BPC | Endpoint | S21 wins | Effect / noise |
|---:|---:|---:|---:|---:|
| 7 | -0.000855 | -0.001428 | 3/3 | 0.046 |
| 13 | -0.000115 | -0.000011 | 3/3 | 0.010 |
| 21 | -0.000016 | -0.000042 | 2/3 | 0.001 |
| Three-seed mean curve | **-0.000329** | **-0.000494** | **3/3** | **0.115** |

This is not evidence of improved capability: the paired effect remains far below
terminal curve noise, and every organism is still improving materially. It is,
however, the preregistered reason to continue rather than stop: the new state is
measurably active, bounded, and not already adverse (8/9 seed-checkpoint wins).

The exact checkpointed organisms were therefore resumed, without resetting stream,
body, anatomy, optimizer, or routing memory, to the full 2,000-update decision horizon.
The preflight curves and magnified paired gaps are retained in
`s21-preflight-gate.html`.

### 2,000-update decision

All three resumed services exited successfully at update 2,000. Resume preserved the
exact next corpus window and checkpointed organism; the 4090 was released cleanly.
Against S17 over the five preregistered terminal validations:

| Seed | Mean S21 − control BPC | S21 wins | Effect / noise | Endpoint |
|---:|---:|---:|---:|---:|
| 7 | -0.000497 | 4/5 | 0.037 | -0.003840 |
| 13 | +0.000295 | 2/5 | 0.010 | +0.000967 |
| 21 | +0.000600 | 2/5 | 0.038 | -0.002895 |
| Three-seed mean curve | **+0.000133** | **3/5** | **0.014** | **-0.001922** |

Pooled across seed/checkpoint pairs, S21 wins 8 of 15. Its average regression is
`0.000133` BPC—about 1.4% of measured terminal curve noise. The favorable endpoint is
not independently meaningful: one validation earlier, all three seeds regressed and
the mean gap was `+0.003098` BPC. At update 2,000 that mean gap reversed to `-0.001922`.
This oscillation around zero is the result, not evidence for choosing either endpoint.

Both arms continue learning, but their terminal movement is effectively parallel:
the fitted S21-minus-control slope is only `+0.0000217` BPC per 100 updates. Per-run
status remains:

| Seed | Terminal status | Slope / 100 updates | Best BPC | Final BPC |
|---:|---|---:|---:|---:|
| 7 | unresolved terminal noise | -0.00978 | 3.38356 | 3.38356 |
| 13 | unresolved terminal noise | -0.01236 | 3.36577 | 3.36577 |
| 21 | still improving | -0.02562 | 3.48041 | 3.48041 |

The full graph, including the 1,750-to-2,000 reversal, is retained in
`s21-reward-routing-decision.html`.

### Mechanistic diagnosis

The null capability result is not an inactive implementation. Replaying the normalized
routing policy from each final live checkpoint gives:

| Seed | Mean routing deviation | Maximum deviation | Saturated live slots | Active edges | Spawns | Prunes |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 3.20% | 27.51% | 1.39% | 36 / 64 | 18 | 14 |
| 13 | 1.25% | 19.86% | 0% | 31 / 64 | 12 | 13 |
| 21 | 0.64% | 9.88% | 0% | 31 / 64 | 5 | 6 |

This is comparable to the material S20 routing scale, yet it remains bounded. Anatomy
ends exactly at the S17 edge count for seeds 7 and 13; seed 21 retains one additional
edge because it performs one fewer prune. Reward-addressed memory therefore learns a
real routing policy but neither improves prediction nor systematically redirects
development.

Relative to S20's instantaneous gain-100 router, S21 improves the five-checkpoint
three-seed mean by `0.00501` BPC and its endpoint by `0.00815`. Delayed reward removes
S20's measurable harm, but it only returns the organism to the unrouted S17 control.

### Decision

Do not promote reward-plastic routing to the live checkpoint. Retain the
default-disabled implementation and telemetry because the negative result is
reproducible and the mechanism is useful experimental infrastructure.

S19 through S21 now distinguish three failures:

1. unscaled instantaneous alignment is too weak to matter;
2. calibrated instantaneous alignment changes routing and anatomy but harms capability;
3. delayed correlational reward learns a bounded policy but is indistinguishable from
   no routing.

The next routing experiment should measure a routing choice by its own live
counterfactual traffic, as morphology already does, rather than infer fitness from a
noisy global reward correlation. Ordinary weights, stream state, anatomy, and all other
exploration must continue in both arms.
