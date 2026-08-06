# S1-P2A: mastery-gated acquisition scaling

Status: protocol frozen 2026-08-04 before GPU results.

## Question

Did S1-P fail its acquisition gate because the 310k-parameter organism received too
little full-depth practice, or does a larger and deeper organism acquire the procedure
materially more efficiently?

This is an acquisition calibration, not yet the organ-attachment result. Its winning
checkpoint preserves the living state and optimizer for `DMON-1-xkp`.

## Matched arms

Both arms use seed 7, bound-4 operators, private cell identity, batch size 24, identical
generated examples, 1e-3 constant learning rate after 200 warmup updates, and 64-batch
fixed-length-four evaluations every 1,000 updates. The curriculum admits length one for
updates 1–1,000, lengths one/two for 1,001–2,000, one through three for 2,001–3,000,
and one through four thereafter. Unlike S1-P, extending the cap does not postpone the
introduction of length four.

| Arm | Cells | Hidden | Dendrites active/capacity | Steps/token | Queries | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 112 | 96 | 8/12 | 3 | 4 | 310,390 |
| Larger/deeper | 208 | 192 | 12/16 | 5 | 8 | 1,224,278 |

The standard arm runs on physical RTX 2070S UUID
`GPU-4e207c93-ed93-c35e-f0f2-e37c8df2b047`; the larger arm runs on physical RTX 4090
UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. UUID selection avoids CUDA ordinal
ambiguity.

## Stop and selection rules

- Mastery is at least 80% exact held-out length-four accuracy on two consecutive
  1,000-update evaluations.
- Each arm stops independently at mastery or 10,000 updates.
- Zero rejected updates and finite telemetry are required.
- If only one arm masters, it supplies the organ-attachment checkpoint.
- If both master, prefer standard unless larger/deeper masters at least 2,000 updates
  earlier. A one-interval advantage is not enough to justify quadrupling parameters and
  increasing microsteps.
- If neither masters, do not scale again blindly; inspect neuron utilization and
  learning curves before changing curriculum, processing depth, or task structure.

The comparison is intentionally a composite scale-up, not a pure width ablation. It
tests the practical package proposed for a larger organism: more compute/relay cells,
private channels, dendritic capacity, queries, and internal evolution time. Results are
reported against examples, wall time, peak memory, and parameter count.

## Visualization telemetry

Every mastery checkpoint records the real living state rather than a rendered fiction:
per-cell activation magnitude and variation, private-identity magnitude, tissue label,
installed in-degree, active source/target edges, and learned edge biases. The first
visualizer will use these checkpoints to scrub through development, inspect a cell, and
compare standard and larger organisms. The topology is expected to remain fixed in this
calibration; later growth experiments will add structural events to the same timeline.
