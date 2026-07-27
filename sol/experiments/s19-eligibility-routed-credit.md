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
