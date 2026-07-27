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

Pending.
