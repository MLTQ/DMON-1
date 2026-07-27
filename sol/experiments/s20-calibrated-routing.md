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

Pending.
