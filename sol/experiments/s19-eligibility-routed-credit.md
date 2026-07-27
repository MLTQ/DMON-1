# S19: Eligibility-routed vector credit

## Question

Can decoder-shaped reverse credit become more selective by following installed axons
whose source cells remember an event aligned with the correction?

S18 showed that a scalar reverse wave reaches internal cells but does not improve
capability. S19 therefore changes routing, not amplitude: it retains S17's zero scalar
gain and channel-shaped output-error credit, then uses the existing event eligibility
memory to distribute that vector within each target's live dendrite fan.

## Mechanism

For each installed target-owned dendrite:

1. transpose the learned forward message-value transform;
2. multiply by the exact signed forward message coefficient;
3. measure channel alignment with the named source cell's eligibility memory;
4. softmax those alignment scores across active slots owned by the same target;
5. rescale the routing weights so equal evidence exactly reproduces historical
   transport magnitude;
6. scatter-add the routed vector into the named source cells.

Dormant slots are excluded. The mechanism adds no parameters, no global communication
graph, and no detached organ. Ordinary truncated BPTT, persistent state, live
exploratory traffic, and global-loss-gated morphology continue unchanged.

## Matched protocol

The S17 globally gated adaptive organisms are the controls. Treatment adds only:

```text
--eligibility-routed-output-credit
```

Use the same Tiny Shakespeare stream, 16×16 field, two-of-four birth fan-in, output-error
gain `0.5`, structural cadence, ABBA candidate traffic, seeds 7/13/21, RTX 4090, and
2,000-update decision horizon.

## Gates

Before GPU work:

- default transport remains compatible;
- equal branch evidence is a no-op;
- matching event memory receives more credit than a misaligned memory;
- dormant anatomy receives zero credit;
- parameter count is unchanged;
- checkpoint and full local suites pass.

After GPU work, require complete curves and the same five-point terminal
ordering/noise analysis used in S16–S18. Promote only replicated capability, not merely
nonzero routing activity.

## Results

All three routed organisms completed update 2,000 on the RTX 4090. The treatment and
S17 control are empirically identical:

| Seed | Mean routed − control BPC | Terminal wins | Effect / residual noise | Endpoint routed − control |
|---|---:|---:|---:|---:|
| 7 | +0.000046 | 3 / 5 | 0.0032x | +0.000285 |
| 13 | -0.000861 | 3 / 5 | 0.0316x | -0.000119 |
| 21 | -0.000059 | 5 / 5 | 0.0036x | -0.000138 |

Pooled mean improvement is only `0.000291` BPC, endpoint difference is a `0.000009`
BPC regression, and effect/noise is `0.033x`. One transient seed-13 improvement at
update 1,750 accounts for nearly all of the pooled mean; it does not persist. Structural
event counts and final anatomy are identical to S17 in every seed.

## Mechanistic diagnosis

The final seed-7 checkpoint confirms that the unscaled router is too weak:

| Measurement | Value |
|---|---:|
| Mean absolute alignment score | 0.000413 |
| Maximum absolute alignment score | 0.012666 |
| Mean absolute routing-weight deviation from 1 | 0.000345 |
| Maximum absolute routing-weight deviation from 1 | 0.007262 |

Thus the implementation changes a typical installed branch's reverse contribution by
only `0.035%`. The learning curves are identical because the real event-credit product
is much smaller than the synthetic unit-memory contract test.

Checkpoint calibration gives a principled next range:

| Alignment gain | Mean routing deviation | Maximum routing deviation |
|---:|---:|---:|
| 1 | 0.03% | 0.73% |
| 10 | 0.35% | 7.32% |
| 30 | 1.03% | 21.49% |
| 100 | 2.91% | 48.29% |
| 300 | 5.61% | 48.65% |
| 1,000 | 9.75% | 53.02% |

Gain `100` is the next treatment: it makes routing material without driving the typical
branch near saturation. This is calibration against observed organism state rather than
an arbitrary amplitude sweep.

## Decision

Do not promote the unscaled router. Preserve the mechanism and add an explicit
checkpointed alignment gain, defaulting to `1` for exact S19 reproduction. S20 will
change only that gain to `100` against S17's unrouted control.
