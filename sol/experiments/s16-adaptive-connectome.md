# S16: Credit-guided adaptive connectome

## Question

Can SOL grow and prune actual directed communication paths from continuous traffic and
event-specific reverse credit, without resizing the model, resetting the optimizer, or
freezing the rest of the body?

## Representation

Every target retains a fixed maximum number of dendrite parameter slots, but a
checkpointed Boolean mask determines which slots are anatomically active. This
separates parameter capacity from live fan-in:

- dormant slots carry zero attention, message flow, reverse credit, energy transport,
  eligibility, and fast efficacy;
- activating or deactivating a slot never changes parameter count or optimizer tensor
  shape;
- older checkpoints reconstruct every historical slot as active;
- `initial_active_dendrites` allows an organism to begin below capacity and grow.

Topology analysis counts only active edges. Every prune, replacement, and spawn must
retain complete sensory-to-cell and sensory-to-output reachability.

## Evidence

Installed edges retain three distinct signals:

1. signed scalar reward times edge eligibility;
2. actual forward traffic;
3. signed alignment between source-cell event eligibility and decoder-shaped credit
   transported backward through that edge.

Exploratory candidates retain the corresponding scalar and decoder-credit evidence plus
their exact first-order global loss effect. Optional circular locality is a weak prior,
not a substitute for causal traffic. Edge age, endpoint energy, repeated confirmation,
and live ABBA candidate-on/candidate-off probation remain independent gates.

Retention score combines scalar credit, traffic, and decoder-credit alignment. A mature
edge can be pruned only when both traffic and combined retention fall below explicit
thresholds, target fan-in stays above its floor, and reachability survives. A qualified
candidate uses dormant capacity when available; otherwise it competes with the weakest
mature incumbent.

## Protected contracts

The local suite currently passes 68 tests. New tests require:

- dormant slots to carry no traffic or transient synaptic state;
- topology metrics to count only active anatomy;
- decoder-credit evidence to be nonzero only when reverse credit meets source event
  memory;
- a decoder-credit-qualified candidate to spawn into dormant capacity without adding
  parameters;
- an unused redundant edge to prune while full reachability survives;
- old checkpoints to restore an all-active historical connectome.

## Local live-traffic smoke

A 12-cell CPU organism began with two of four slots active per target and ran 20
continuous updates with vector-only credit gain `0.5`, locality gain `0.01`, and
four-window exploratory probation.

- three candidate probes won live ABBA traffic trials and spawned;
- active anatomy grew from 24 to 27 directed edges;
- mean fan-in grew from `2.00` to `2.25`;
- all 12 cells and both output cells remained sensory-reachable;
- no pruning was enabled in this growth smoke;
- held-out BPC at update 20 was `5.499`, with reset `5.704` and shuffled `5.521`.

Each committed spawn records old dormant state, candidate identity, scalar/vector
evidence, arm rewards, exact decision updates, and endpoint/body energy before and after
growth. Ordinary body learning continued in both traffic arms.

## RTX 4090 preflight

Use the reduced 16-by-16 Tiny Shakespeare protocol and vector-only credit gain `0.5`.
Compare:

1. fixed topology with four active slots;
2. probes-only with two active of four slots;
3. adaptive fan-in with two active of four slots.

All arms receive identical causal probe traffic. Adaptive and probes-only use balanced
exploratory probation; only adaptive can mutate anatomy or spend growth energy.

Before a full three-seed comparison, calibrate structural cadence, traffic threshold,
credit threshold, and weak locality gain on seed 7. Require:

- both spawn and prune events without reachability rejection dominating;
- no abrupt held-out BPC regression around decisions;
- nonzero decoder-credit anatomical evidence;
- bounded fan-in rather than monotonic saturation at slot capacity;
- persistent/reset/shuffle behavior at least as strong as the fixed control.

The 2070S may run matched controls only after both cards remain stable under isolated
PyTorch initialization. The primary adaptive comparison remains on the 4090 to avoid
cross-architecture causality claims.
