# S2 credit-guided structural plasticity

Date: 2026-07-26

## Question

Can continuously measured exploratory traffic identify a better source for a target
cell, and can installing that source as a permanent dendrite improve character
prediction without disconnecting the organism or resetting its learned process?

## Mechanism

- Every target receives one weak candidate message from a source not already in its
  fixed dendrite fan.
- The shared cell rule runs once with the probe and once without it under `no_grad`.
  Their hidden-state difference is the candidate's measured causal effect.
- Delayed signed reward addresses retained candidate and incumbent edge eligibility
  across truncated-BPTT windows. Credit retains its sign, so a strongly harmful probe
  is not mistaken for a useful one because of magnitude alone.
- At a bounded structural phase, a candidate may replace the target's mature
  lowest-credit slot only when its retained credit is positive and exceeds that
  incumbent by a nonzero margin.
- A tentative replacement must keep every cell and output reachable from the sensory
  population.
- Source and target pay the growth energy cost. The reused slot's slow weight, bias,
  optimizer moments, edge eligibility, and fast efficacy reset; its slow weight starts
  at the approximate coefficient that carried the successful probe. Every other slot
  and optimizer value remains intact.

Tensor shapes and parameter counts never change. This is morphology over named
connections, not model growth by adding parameters.

## Matched control

`--structural-probes-only` receives the same continuously rotating candidate messages,
retains the same candidate credit, and keeps training the entire neural field. It
freezes only installation into `sources`; it does not freeze the character organ,
weights, recurrent state, reward, or exploratory traffic.

Both GPU arms will disable fast efficacy because S1 found no meaningful net language
effect. Cell reward remains active and supplies the structural credit signal.

## Protected contracts

The deterministic suite requires:

- probe source IDs are genuine non-edges;
- probe eligibility is nonzero only when the causal intervention has an effect;
- fixed fan-in and unique target-owned source slots;
- complete sensory-to-cell and sensory-to-output reachability;
- bounded replacements per phase and minimum edge maturity;
- endpoint energy decreases by exactly the growth cost;
- reused parameter/optimizer/state slots reset while untouched slots remain bitwise
  unchanged;
- probes-only rotation never changes permanent topology;
- checkpoints resume structural policy, candidates, credit, age, topology, and rewrite
  count, while older schema-1 checkpoints upgrade additively.

The full repository suite passes 38 tests.

## CPU integrity smoke

Both runs used 16 cells, 16 channels, four dendrites, four lanes × 16 characters,
300 updates, seed 7, no metabolism, no fast efficacy, probe gain 0.03, a 25-update
phase interval, and a 50-update warmup.

| Variant | Rewrites | Best/final BPC | Reachable cells | Reachable outputs |
|---|---:|---:|---:|---:|
| Probes only, fixed topology | 0 | 4.300 / 4.300 | 16 / 16 | 4 / 4 |
| Credit-guided rewiring | 8 | 4.308 / 4.308 | 16 / 16 | 4 / 4 |

The BPC difference is immaterial at this scale. The smoke establishes that ordinary
streamed reward produces real, paid, viable rewrites and that the matched control
experiences the same exploratory traffic without topology mutation.

The first guarded version reset a successful probe's permanent slow weight to zero at
installation; reverting its learned topology improved BPC from 4.283 to 4.267, exposing
that graft-time discontinuity as harmful. Initializing the reused slot to the probe's
approximate message coefficient and rotating candidate IDs round-robin reverses the
causal morphology result: the installed topology scores 4.308 BPC and restoring birth
sources worsens it to 4.394 (+0.086). Overall BPC remains effectively tied with the
4.300 probes-only control, so this is mechanism validation rather than a capability
claim.

## Full-scale gate

The first GPU comparison used 64 cells/channels, eight dendrites, 5,000 updates,
16 lanes × 32 characters, seed 7, no metabolism, no fast efficacy, and identical probes
on both arms. The growing arm had to:

1. complete with finite metrics and full output reachability;
2. perform nonzero but bounded rewrites;
3. beat the probes-only control by a practically meaningful BPC margin rather than a
   single-evaluation rounding difference;
4. worsen when its installed source table is replaced at inference by the deterministic
   birth topology;
5. pass the completed-run stability guard.

No live checkpoint promotion follows from morphology alone.

## Full-scale exploratory-traffic result

Both organisms completed stably with complete sensory and output reachability. The
probes-only control continuously rotated the same candidates and trained every ordinary
weight and persistent state, but never installed a candidate into permanent topology.
The growing organism installed 46 rewrites across 46 eligible post-warmup phases, or
8.98% of its 512 fixed source slots.

| Statistic | Growing anatomy | Probes only | Growth − control |
|---|---:|---:|---:|
| Best BPC | 2.51217 | 2.47096 | +0.04122 |
| Final BPC | 2.53904 | 2.55185 | −0.01281 |
| Mean BPC, all 20 paired evaluations | 2.71039 | 2.69758 | +0.01281 |
| Mean BPC, updates 2500–5000 | 2.59221 | 2.57194 | +0.02027 |
| Mean BPC, updates 3750–5000 | 2.55726 | 2.54362 | +0.01364 |

Lower is better. Growth won 8 of 20 paired checkpoints, 3 of 11 checkpoints in the
second half, and 2 of the final 6. Its paired advantage ranged from 0.05224 BPC better
to 0.06891 worse. The favorable final checkpoint therefore does not satisfy the
predeclared capability gate.

The causal morphology intervention tells a different and important story. Replacing
only the growing model's learned source table with its deterministic birth topology
worsened final BPC from 2.53904 to 2.80070. The mean ablation penalty was 0.22423 BPC
over the second half and 0.26035 over the final six evaluations. By contrast, the
probes-only control's birth-topology evaluation was exactly identical to its persistent
evaluation.

## Interpretation

Continuous exploratory traffic can be consolidated into anatomy that the neural field
uses deeply and increasingly relies upon. This is not a frozen-organ comparison: both
arms remain stimulated, recurrent, reward-addressed, and fully trainable throughout.

The first consolidation policy nevertheless fails as a fitness rule. It accepted a
rewrite in all 46 eligible phases and rejected none, so the 0.001 instantaneous credit
margin provided no practical selectivity. The model repeatedly co-adapted to consequential
new wiring, but its held-out language performance was worse on average than exploration
without installation.

The next structural experiment should retain continuous probes while holding each
candidate across multiple structural phases and requiring repeated positive evidence
before installation. That tests selective consolidation rather than increasing
exploration or rewrite volume. The current live checkpoint is unchanged.
