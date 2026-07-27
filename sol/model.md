# `model.py`

## Purpose

Implements the first SOL organism: homogeneous recurrent cells with private persistent
state, explicit directed dendrites, streamed character input, character output, metabolic
state, reward-addressable eligibility memory, bounded fast synaptic efficacy, and
causally measured exploratory axon probes. It also supports a parameter-neutral,
channel-shaped prediction-error signal that moves backward through the installed
connectome.

## Components

### `SolConfig`
- **Does**: Holds field size, directed topology, recurrence, eligibility, energy, and
  optional scalar and output-error backward-credit transport constants, including
  energy transport and reversible quiescence thresholds. An optional maintenance-flow
  floor moves only existing energy along installed non-self axons when information
  traffic is silent.
- **Does**: Optionally routes channel-shaped reverse credit toward installed sources
  whose remembered event eligibility aligns with the signed corrective contribution,
  with an explicit parameter-free alignment gain that defaults to the original scale.
- **Rationale**: All scientific knobs travel with the model rather than hiding in a CLI.
- **Does**: Separates maximum dendrite-slot capacity from the number active at birth;
  the disabled default activates every slot for historical checkpoint equivalence.

### `FieldState`
- **Does**: Carries hidden state, energy, causal stimulation, eligibility, sensory memory,
  delayed reward, persistent scalar output-originating backward credit, persistent
  channel-shaped output-error credit, per-edge and candidate-probe eligibility, and
  per-stream fast weights across every character and optimizer window.
- **Rationale**: Detaching a graph must not reset the organism.
- **Compatibility**: `state_from_snapshot` initializes additive plasticity fields to
  zero when loading checkpoints made before fast synaptic or structural memory existed.

### `FieldTrace`
- **Does**: Retains token-local hidden states and measured traffic for post-backward
  credit inspection, including edge eligibility, fast-weight saturation, and
  reward-addressed incumbent/candidate structural evidence.
- **Does**: Combines each candidate's measured hidden-state intervention with the exact
  post-backward sequence-loss gradient, yielding a first-order organism-level fitness
  estimate without removing exploratory traffic.
- **Does**: Retains token-wise signed reward relative to each lane's surprise baseline
  so post-graft probation measures observed prequential advantage.
- **Does**: Retains external energy input, actual spending, transport drift, mean
  viability, and quiescent fraction so metabolism is auditable beside behavior.

### `SparseAxonField`
- **Does**: Applies one shared GRU rule at every cell while messages travel only through
  named source-to-target dendrites.
- **Interacts with**: `ContinuousTrainer` in `train.py`.

### Structural buffers and `load_compatible_state_dict`
- **Does**: Persist current candidate sources, incumbent/candidate credit, edge age, and
  per-candidate global fitness and confirmation streaks, and total rewrites with the
  model while upgrading older checkpoints additively.
- **Does**: Persist the active-slot mask, traffic EMA, decoder-credit alignment, and
  spawn/prune counters. Older checkpoints reconstruct an all-active mask.
- **Interacts with**: `apply_structural_phase` in `structure.py`.

### `birth_sources`
- **Does**: Reconstructs the deterministic pre-growth source table for a causal
  morphology ablation without storing another full graph.
- **Interacts with**: `evaluate_state_ablations` in `evaluate.py`.

### `_probe_message`
- **Does**: Adds one weak candidate source per target and measures its causal effect by
  comparing the shared cell rule with and without that message.
- **Does**: Accepts a target mask so one candidate can be alternated on and off during
  an exploratory traffic trial without silencing any other candidate traffic.
- **Rationale**: Growth evidence comes from an intervention, not geometric proximity or
  an all-to-all correlation.

### Active dendrite slots
- **Does**: Masks attention, signed coefficients, fast efficacy, eligibility, reverse
  credit, and maintenance energy for dormant slots while preserving fixed tensor and
  optimizer shapes.
- **Rationale**: Axons and dendrites can be born and pruned without rebuilding the model
  or resetting unrelated organism memory.

### `_transport_backward_credit`
- **Does**: Scatter-adds target credit to each target's named sources using the signed
  coefficient of the corresponding forward axon.
- **Rationale**: Reward can travel against the directed causal path without inventing a
  second all-to-all teaching network.

### `_transport_output_error_credit`
- **Does**: Applies the transpose of the learned forward `message_value` transform,
  multiplies by each target-owned signed message coefficient, and scatter-adds the
  resulting channel vector into the named source cell.
- **Does**: When eligibility routing is enabled, scores each installed branch from the
  signed reverse contribution's alignment with source event memory, then normalizes
  multipliers within the target dendrite fan. Equal evidence reproduces historical
  transport scale and dormant slots receive no credit.
- **Does**: Applies `eligibility_routing_gain` before the bounded alignment nonlinearity,
  making weak real-organism event traces experimentally calibratable without changing
  decoder-credit amplitude or adding learned parameters.
- **Rationale**: The reverse signal preserves which hidden-state directions would
  correct the decoder instead of collapsing error to one scalar.
- **Rationale**: Persistent credit should preferentially reach cells tied to the event
  being corrected, without creating a second learned network or changing parameter
  count.

### `observe_prediction`
- **Does**: Converts decoder error to both the existing scalar surprise-relative reward
  and a detached corrective vector
  `W_readout^T(one_hot(target) - softmax(logits))`, split across output cells and
  smoothly bounded before persistence.
- **Rationale**: Future reward can meet a cell memory of an event in the relevant hidden
  channels, without retaining an unbounded autograd graph across stream windows.

### `_transport_energy`
- **Does**: Redistributes source-owned energy through named dendrites and active
  candidate probes, normalizing total outbound demand so a source cannot spend more
  than it owns.
- **Does**: Optionally adds a parameter-free maintenance request on installed non-self
  axons so low-activity tissue can receive externally sourced energy without undirected
  diffusion. Candidate probes receive no maintenance subsidy.
- **Does**: Scales accepted transfer by each target's remaining capacity; unaccepted
  transfer stays with its source, so transport conserves total energy exactly.
- **Rationale**: Recurrent stimulation may carry information but cannot become food;
  only external input can increase total energy.

### `tick`
- **Does**: Consumes one optional streamed character, performs recurrent graph updates,
  tags individual dendrites and causal probes from local activity, applies delayed
  reward to remembered tags, updates energy, and emits next-character logits.
- **Does**: Measures whether channel-shaped reverse credit crossing each installed edge
  or causal probe actually aligns with event eligibility at the source cell, providing
  signed anatomical evidence distinct from raw traffic.
- **Rationale**: `token=None` remains a live interval and proves energy depletion without
  stimulation.
- **Reward contract**: A pending reward is consumed once. `forward_sequence` stores the
  newest score after each prediction, so the final score crosses an optimizer boundary
  exactly once while free generation cannot replay stale prompt reward.
- **Reward baseline**: Each stream lane retains an exponential moving expectation of
  surprise. Observed error becomes signed reward relative to that expectation, avoiding
  permanently positive potentiation once the model merely beats chance.
- **Backward credit**: When enabled, pending scalar reward enters output cells once,
  propagates toward source cells against signed axons, persists across stream windows,
  and affects a cell only where it meets that cell's remembered eligibility.
- **Output-error credit**: When enabled, the latest corrective decoder vector is
  launched from output cells, moves sourceward through the transpose of the actual
  forward message transform, and contributes only where its channels align with a
  cell's eligibility trace.
- **Eligibility-routed output credit**: The optional router uses that same source event
  memory to distribute reverse credit among installed dendrites. It is disabled by
  default for checkpoint and experiment compatibility; routing gain `1` exactly
  reproduces the original S19 mechanism.
- **Energy provenance**: A character's novelty supplies one bounded external budget to
  sensory cells. Basal/activity costs destroy energy and directed traffic only moves it;
  no internal path may mint or discard energy.
- **Maintenance provenance**: Maintenance flow is an optional directed redistribution
  request, not an inflow. It remains subject to the same per-source budget and
  per-target capacity normalization as activity-driven transport.
- **Viability**: Energy maps through a parameter-free ramp from quiescent to fully
  active. Quiescent cells stop updating and emitting but can recover from later sensory
  or incoming axonal energy.
- **Finite starvation**: Any non-quiescent cell pays full basal maintenance, ensuring
  silence crosses the lower threshold in finite time instead of asymptotically hovering
  just above it.
- **Ablation contract**: `allow_fast_plasticity=False` forces zero efficacy through the
  tick without disabling edge-tag measurement or the rest of the recurrent field.

### `forward_sequence`
- **Does**: Runs truncated differentiable windows without resetting field state.
- **Does**: Can gate selected structural probes for an entire streamed window while
  preserving the same hidden, metabolic, reward, and gradient path.
- **Rationale**: Exact BPTT handles within-window credit; scalar reward and optional
  channel-shaped decoder correction gate persistent eligibility on the following tick,
  including across truncation boundaries.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train.py` | Tokens and logits use `(batch, time, ...)`; returned state remains attached | Shapes, return arity |
| Tests and diagnostics | `sources[target, slot]` plus `active_edges[target, slot]` define the directed graph | Topology representation |
| `structure.py` | Probe sources are non-edges and structural buffers retain fixed shapes | Buffer shapes |
| Future modalities | A modality injects cell-aligned stimulus and causal stimulation | Replacing `tick` semantics |
| Metabolic experiments | Total energy can rise only by reported external input | Energy update or transport normalization |

## Notes

- Rewiring is disabled by default. When enabled, one weak non-edge probe per target
  supplies counterfactual evidence to a bounded between-window grow/prune phase.
- Slow edge parameters learn by exact truncated BPTT. Fast efficacy is stream-local,
  smoothly bounded, decays, and changes only when delayed reward meets a remembered
  edge tag.
- Locally computed tags are centered across each target's dendrites before entering
  eligibility memory. This makes plasticity competitive within a dendrite fan instead
  of allowing globally positive reward to potentiate every incoming edge together.
- Fast efficacy uses a scaled `tanh` homeostat rather than a hard clamp. Small updates
  remain nearly linear, while values approaching the bound receive increasing
  contraction instead of accumulating as clipped, effectively frozen synapses.
- Energy now governs reversible quiescence but does not yet cause irreversible death,
  cell birth, or reproduction.
- Both backward-credit paths are parameter-neutral and disabled by default for
  checkpoint compatibility. Separate gains permit direct reward, scalar reverse
  reward, and channel-shaped decoder credit to be tested as matched controls.
- The forced sensory-to-output axons are organ plumbing, not a learned language-specific
  connectome; all synaptic signs and strengths remain trainable.
