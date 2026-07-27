# S20: Calibrated eligibility routing

## Question

Does event-memory routing improve held-out character prediction once its selectivity is
large enough to alter real reverse-credit traffic?

S19 established that eligibility-aligned routing is safe and directional, but its
typical branch multiplier moved only `0.035%`. The resulting learning curves were
empirically identical to the unrouted S17 control. S20 changes only the alignment
scale, preserving the reverse vector, eligibility memory, topology policy, streamed
state, and ordinary truncated BPTT.

## Calibration

Offline measurement on S19's final seed-7 checkpoint swept the existing alignment score
without retraining:

| Alignment gain | Mean routing deviation | Maximum routing deviation |
|---:|---:|---:|
| 1 | 0.03% | 0.73% |
| 10 | 0.35% | 7.32% |
| 30 | 1.03% | 21.49% |
| 100 | 2.91% | 48.29% |
| 300 | 5.61% | 48.65% |
| 1,000 | 9.75% | 53.02% |

Gain `100` is the smallest tested scale that makes typical routing material while
keeping the typical branch far from saturation. This is a preregistered calibration
from observed organism state, not a post-hoc capability sweep.

## Matched protocol

The controls are the same S17 globally gated adaptive organisms used for S18 and S19.
Treatment adds only:

```text
--eligibility-routed-output-credit
--eligibility-routing-gain 100
```

Use Tiny Shakespeare, a 16-cell × 16-channel field, two active slots of four at birth,
output-error credit gain `0.5`, global-loss-gated variable fan-in, uninterrupted ABBA
exploratory traffic, seeds 7/13/21, one RTX 4090, and a 2,000-update horizon. Keeping
every matched arm on the same device avoids a GPU-specific numerical confound.

## Gates

Before GPU work:

- default gain `1` reproduces S19;
- equal evidence remains a no-op at any gain;
- gain `100` increases matching-versus-misaligned branch selectivity;
- negative gain is rejected;
- parameter count is unchanged;
- checkpoint resume preserves the gain exactly;
- focused and full local suites pass.

After GPU work:

- graph every validation for all three seeds;
- compare the five aligned evaluations from update 1,000 through 2,000;
- report per-seed and pooled terminal wins, mean and endpoint BPC advantage,
  effect relative to residual noise, anatomy, and terminal movement;
- retain the gain only if capability replicates beyond seed and terminal noise.

## Results

All three gain-100 organisms completed update 2,000 on the RTX 4090. Their
manifests differ from S19 only by output path and the newly explicit routing gain.
Against the same-seed unrouted S17 controls, the five evaluations from update 1,000
through 2,000 gave:

| Seed | Mean treatment − control BPC | Terminal wins | Effect / residual noise | Endpoint treatment − control |
|---|---:|---:|---:|---:|
| 7 | +0.00283 | 2 / 5 | 0.18x | +0.01020 |
| 13 | -0.00163 | 5 / 5 | 0.06x | -0.00023 |
| 21 | +0.01422 | 1 / 5 | 0.85x | +0.00871 |

Lower BPC is better. Pooled across seed/checkpoint pairs, treatment wins 8 of 15
measurements and regresses by `0.00514` BPC on average. The three-seed mean control
wins four of five terminal checkpoints; treatment regresses by `0.00623` BPC at the
endpoint. The mean-curve effect/noise ratio is only `0.57x`, so the result does not
clear the `2x` comparison gate.

All organisms remain stable, fully sensory/output reachable, and still learning:

| Seed | Terminal status | Slope per 100 updates | Best BPC | Final BPC |
|---|---|---:|---:|---:|
| 7 | unresolved terminal noise | -0.00906 | 3.39156 | 3.39760 |
| 13 | unresolved terminal noise | -0.01270 | 3.36457 | 3.36457 |
| 21 | still improving | -0.02472 | 3.49202 | 3.49202 |

The complete paired curves and magnified terminal differences are in
`s20-calibrated-routing-decision.html`. Continued learning is disclosed, but extending
the horizon is not justified for a treatment already behind at four of five
three-seed mean checkpoints with an effect smaller than residual noise.

## Mechanistic diagnosis

Gain `100` did make the S19 mechanism materially active without saturating it:

| Seed | Mean routing deviation | Maximum deviation | Saturated alignments | Active edges | Spawns | Prunes |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 3.62% | 47.74% | 0% | 36 / 64 | 18 | 14 |
| 13 | 2.87% | 40.42% | 0% | 33 / 64 | 11 | 10 |
| 21 | 0.77% | 15.82% | 0% | 32 / 64 | 7 | 7 |

Unlike S19, anatomy no longer remains identical to S17: seed 13 ends at 33 rather than
31 active edges and seed 21 at 32 rather than 30. The gain therefore changes both
reverse traffic and development; its failure is not another implementation-scale
no-op. Seed 21 supplies the strongest falsification: routing changes are present while
capability regresses at four of five terminal measurements.

## Decision

Reject gain `100` and do not promote eligibility-routed credit to the live organism.
Keep the explicit configuration knob, defaulting to `1`, so the negative result remains
reproducible.

S19 and S20 jointly reject amplitude as the missing ingredient: gain `1` is too weak,
while a calibrated non-saturating gain changes credit and anatomy without improving
prediction. The next backward-credit mechanism should learn *which past routing event
helped a later prediction*, rather than treating instantaneous
reverse-vector/eligibility alignment as fitness. A branch-specific three-factor trace
can remember forward participation, backward corrective alignment, and subsequent
prequential reward before changing future credit allocation.
