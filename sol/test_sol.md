# `test_sol.py`

## Purpose

Protects the scientific contracts of the first SOL prototype rather than only checking
tensor shapes.

## Components

### Topology tests
- **Does**: Proves the graph is sparse, directed, and connects sensory cells to outputs.
- **Does**: Requires complete sensory reachability and reports shortest output paths.
- **Does**: Proves causal candidate probes name non-edges and produce retained
  reward-addressable effects.
- **Does**: Proves causal probe effects contract with exact post-backward sequence-loss
  gradients into finite, nonzero organism-level fitness evidence.
- **Does**: Forces a beneficial candidate rewrite and proves fixed fan-in, unique source
  slots, complete reachability, bounded replacement count, endpoint energy payment,
  probe-equivalent graft initialization, reused-slot transient-state reset, and
  untouched-slot parameter/optimizer integrity.
- **Does**: Proves the probes-only control rotates candidates on schedule while leaving
  the authoritative topology and rewrite count unchanged.
- **Does**: Holds matched probes across multiple reward windows, requires consecutive
  positive confirmations before installation, and resets a candidate's streak after
  adverse evidence.
- **Does**: Proves negative organism-level fitness vetoes otherwise positive local
  structural credit, while positive local and global evidence can still rewire.
- **Does**: Proves failed post-graft probation restores the incumbent source, parameters,
  optimizer moments, structural memory, and stream-local edge state exactly while
  preserving unrelated body adaptation and spent growth energy.
- **Does**: Proves positive probation retains the adapted graft and the probes-only
  organism enters a virtual wait without topology mutation.
- **Does**: Proves a graft that earns positive raw reward but underperforms the body's
  pre-graft developmental baseline is rolled back.
- **Does**: Proves a causally harmful candidate cannot replace even a worse incumbent
  merely because its credit magnitude is large.

### Stream and persistence tests
- **Does**: Proves optimizer windows are adjacent and prior characters affect later
  predictions.

### Metabolism and eligibility tests
- **Does**: Proves unstimulated energy depletion and delayed reward dependence on a
  remembered event trace.
- **Does**: Proves delayed reward changes only tagged dendrites, zero reward cannot
  create fast efficacy, fast weights remain bounded and differentiable, and plasticity
  cannot mint metabolic energy.
- **Does**: Proves learned edge tags are competitive and zero-sum within each target's
  dendrite fan, preventing broad reward-driven potentiation.
- **Does**: Proves the smooth homeostatic efficacy update remains bounded and consumes
  pending reward without falling back to hard clipping.
- **Does**: Proves a pending reward is consumed exactly once instead of being replayed
  during unstimulated or generated ticks.
- **Does**: Proves reward is positive or negative relative to a checkpointed moving
  expectation of surprise rather than permanently positive after beating chance.

### Credit-path test
- **Does**: Proves exact loss gradients reach retained cell states and directed synapses.

### Learning smoke test
- **Does**: Requires substantial loss reduction on a deterministic character stream
  without resetting organism state.

### Checkpoint test
- **Does**: Proves save/load preserves the exact next update, stream cursor, live field
  state, optimizer, vocabulary, structural policy/credit/global-fitness/confirmation
  buffers, and metadata.
- **Does**: Resumes an active provisional graft through its decision and requires the
  same loss, topology, body state, and commit/rollback counters.
- **Does**: Proves checkpoints made before fast synaptic fields existed load with safe
  zero-valued additive fast/structural state, deterministic probes, and a chance-level
  surprise baseline.
- **Does**: Proves a frozen-connectome control neither changes edges nor becomes
  trainable after resume.
- **Does**: Loads an inference-only lane through `LiveOrganism`, generates output, and
  requires nonzero measured prompt credit plus real edge-eligibility and fast-weight
  telemetry.
- **Does**: Validates multiple completed candidates, promotes the lowest-BPC checkpoint,
  rejects a transient best from a collapsed run, and reloads the atomic destination.

### Evaluation and control tests
- **Does**: Exercises persistent/reset/shuffled held-out policies and verifies the GRU
  and causal-transformer controls are stateful and genuinely parameter matched.
- **Does**: Requires evaluation to report fast synaptic efficacy and shuffle every
  target-owned cell and edge state together.
- **Does**: Zeroes only fast efficacy during held-out inference to prove whether the
  trained organism causally uses online synaptic memory.
- **Does**: Restores deterministic birth sources for one held-out pass and proves both
  source/probe tables return bitwise to their live values afterward.
- **Does**: Requires edge eligibility and fast-weight saturation telemetry in training,
  held-out evaluation, and the local live bridge together with probe and rewrite
  telemetry.
- **Does**: Requires a multi-length warmup sweep to score one fixed token window.

### Report guard test
- **Does**: Requires finite completed summaries, matched parameter/update budgets, and
  correct reset/shuffle penalties before a winner can be named.
- **Does**: Distinguishes ordinary early learning from post-best collapse and deduplicates
  repeated evaluation updates after resume.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| SOL development | All causal, metabolic, learning, resume, and control paths remain real | Removing assertions or replacing measured state with synthetic diagnostics |
