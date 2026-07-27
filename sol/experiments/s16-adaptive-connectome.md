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
used only to rank candidates that already clear causal qualification; it can never
create eligibility. Edge age, endpoint energy, repeated confirmation,
and live ABBA candidate-on/candidate-off probation remain independent gates.

Retention score combines scalar credit, traffic, and decoder-credit alignment. A mature
edge can be pruned only when both traffic and combined retention fall below explicit
thresholds, target fan-in stays above its floor, and reachability survives. A qualified
candidate uses dormant capacity when available; otherwise it competes with the weakest
mature incumbent.

## Protected contracts

The local suite currently passes 73 tests. New tests require:

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

This representation smoke preceded the stricter locality qualification guard; its
three trials establish live spawn mechanics and bookkeeping, not credit causality. The
GPU preflight below requires positive scalar/vector causal evidence independently of
locality before beginning probation.

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

## Four-hundred-update horizon check

Three adaptive/probes-only pairs ran on the 4090 at seeds 7, 13, and 21. The fixed
four-live-edge control ran on the 4090 for seed 7 and, as explicitly auxiliary
cross-architecture references, on the 2070S for seeds 13 and 21.

At update 400, adaptive sparse beat probes-only in two of three seeds and by `0.0793`
BPC on the three-seed mean. Fixed full fan-in was lower than adaptive in every seed.
Those rankings are not yet capability conclusions:

| Condition | Mean BPC at 400 | Mean change, updates 300–400 |
|---|---:|---:|
| Adaptive sparse | 4.1750 | -0.2162 |
| Probes-only sparse | 4.2543 | -0.2309 |
| Fixed full fan-in | 3.8525 | -0.1764 |

Every run ended at its best measured checkpoint and every three-seed mean was still
falling much faster than the `0.01` BPC-per-100 practical plateau bound. The
adaptive/probes endpoint gap was smaller than either condition's movement over the last
100 updates. The 400-update horizon is therefore explicitly **uninformative** about
converged relative capability.

Resume these exact organisms rather than restarting them. Use longer validation windows
and require complete curves plus terminal slope/noise intervals before deciding whether
global-fitness or stricter vector-credit gating is the next structural intervention.

## Two-thousand-update comparison horizon

Long-window evaluation resumed the exact checkpoints with 4,096 validation characters.
The per-run curves still had small nonzero terminal slopes, so mathematical plateau
tests remained inconclusive. The paired hypothesis does not require zero learning:
adaptive morphology must retain a consistent advantage large relative to observed
terminal noise.

Across the five aligned evaluations from update 1,000 through 2,000:

| Seed | Adaptive mean advantage | Terminal wins | Effect / residual noise | Endpoint advantage |
|---|---:|---:|---:|---:|
| 7 | 0.1667 BPC | 5 / 5 | 5.69x | 0.1721 BPC |
| 13 | 0.1601 BPC | 5 / 5 | 7.34x | 0.1094 BPC |

Both comparisons therefore support the claim that learned sparse anatomy improves
capability over equally sparse fixed anatomy at this horizon. Probes-only was descending
faster in both seeds; relative slopes and naive linear crossing estimates remain
reported caveats, not forecasts or automatic vetoes. Seed 21 is required before treating
the result as a replicated structural-policy effect.

The fixed-full controls remained lower in absolute BPC, so S16 supports adaptive
communication relative to sparse birth anatomy, not parity with maximum live fan-in.
