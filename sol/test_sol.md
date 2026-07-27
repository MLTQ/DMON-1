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
- **Does**: Proves a target mask can silence exactly one selected candidate probe while
  every other exploratory connection remains active.
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
- **Does**: Proves exploratory probation leaves incumbent anatomy and energy untouched
  during an ABBA candidate-on/incumbent-only traffic trial, balances observations, and
  grafts only after a positive within-organism reward difference.
- **Does**: Requires each resolved exploratory trial to preserve its identity, traffic
  evidence, decision update, outcome, and whole-body energy change in checkpoint state.
- **Does**: Proves rejected exploratory candidates never mutate anatomy or spend growth
  energy, while shared body parameters continue learning during both traffic arms.
- **Does**: Rechecks endpoint energy at the later commit boundary so an initially
  affordable candidate cannot endanger a body whose energy declined during exploration.
- **Does**: Proves a causally harmful candidate cannot replace even a worse incumbent
  merely because its credit magnitude is large.

### Stream and persistence tests
- **Does**: Proves optimizer windows are adjacent and prior characters affect later
  predictions.
- **Does**: Requires terminal-trend analysis to reject a still-improving horizon and
  accept only a statistically supported practical plateau.
- **Does**: Accepts a consistently separated paired comparison before either arm is
  flat, while rejecting an inconsistent ordering buried in terminal noise.
- **Does**: Proves optional cosine learning-rate decay is disabled by default, uses
  absolute update boundaries, reaches its configured floor, updates every optimizer
  group, and rejects malformed policies.
- **Does**: Requires benchmark memory telemetry to identify non-CUDA devices explicitly
  without invoking CUDA-only APIs.

### Metabolism and eligibility tests
- **Does**: Proves unstimulated energy depletion and delayed reward dependence on a
  remembered event trace.
- **Does**: Proves directed edge/probe energy transport conserves energy exactly, recurrent
  stimulation is not food, reported external input accounts for the only inflow, and
  low-energy cells quiesce then recover from new sensory energy without new parameters.
- **Does**: Proves optional maintenance flow funds a silent target only through installed
  non-self axons, remains exactly conservative, and is absent from the default control.
- **Does**: Requires an unfed active population to reach complete quiescence in a finite
  number of ticks and remain frozen until funded again.
- **Does**: Proves output reward launches a persistent credit wave that moves against
  signed target-owned axons to their named sources, affects only cells with remembered
  eligibility, cannot appear from zero reward, and adds no parameters.
- **Does**: Proves decoder error launches a bounded channel-shaped correction from only
  output cells, moves through the signed transpose of the actual message transform,
  acts only where its channels meet event eligibility, cannot appear without an
  observed target, and adds no parameters.
- **Does**: Proves optional eligibility routing sends more decoder credit through an
  otherwise equal branch whose source remembers a matching event, less through a
  misaligned branch, reproduces historical transport when evidence is equal, excludes
  dormant slots, and adds no parameters. A calibrated gain must amplify selectivity,
  reject negative values, and survive exact checkpoint resume.
- **Does**: Proves reward-plastic routing is inert without a remembered tag and delayed
  reward, responds oppositely to positive and negative reward, routes future correction
  according to bounded preference, preserves equal-evidence transport and parameter
  count, and cannot coexist with instantaneous routing.
- **Does**: Proves exploratory routing preserves committed preference without
  correlational reward updates, proposes a local zero-sum fan perturbation, maintains
  total reverse-credit scale, commits/rejects from balanced ABBA evidence, keeps shared
  body learning active in both arms, sequences topology decisions, and resumes the
  exact next arm and outcome.
- **Does**: Proves dormant dendrite slots carry no traffic or transient synaptic state,
  topology metrics count only active anatomy, a qualified decoder-credit candidate can
  spawn into dormant capacity, and a redundant unused edge can be pruned without
  violating reachability or changing parameter count. Repurposed and dormant slots also
  carry no stale backward-routing trace or preference.
- **Does**: Proves reverse decoder credit meeting source eligibility produces separate
  signed anatomical evidence for installed edges and exploratory candidates.
- **Does**: Proves even a maximal locality prior cannot qualify a candidate with zero
  causal evidence.
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
  state including scalar and output-error backward credit, optimizer, vocabulary, structural
  policy/credit/global-fitness/confirmation buffers, and metadata.
- **Does**: Includes branch routing eligibility/preference in exact resume, probation
  rollback, old-checkpoint additive initialization, and target-owned cell shuffling.
- **Does**: Includes exploratory routing policy and active trial proposal, phase,
  aggregates, counters, and ledger in exact resume and old-checkpoint compatibility.
- **Does**: Resumes an active provisional graft through its decision and requires the
  same loss, topology, body state, and commit/rollback counters.
- **Does**: Resumes midway through an exploratory ABBA trial with the same next traffic
  arm, losses, shared body state, topology decision, and reward aggregates.
- **Does**: Proves checkpoints made before fast synaptic fields existed load with safe
  zero-valued additive fast/structural state, deterministic probes, and a chance-level
  surprise baseline.
- **Does**: Proves a frozen-connectome control neither changes edges nor becomes
  trainable after resume.
- **Does**: Loads an inference-only lane through `LiveOrganism`, generates output, and
  requires nonzero measured prompt credit plus real edge-eligibility and fast-weight
  telemetry.
- **Does**: Advances genuine no-input bridge ticks, requires zero novelty and input
  energy, samples the output organ, and exposes the exact cell/dendrite counts with
  per-cell activity and per-dendrite measured flow.
- **Does**: Validates multiple completed candidates, promotes the lowest-BPC checkpoint,
  rejects a transient best from a collapsed run, and reloads the atomic destination.

### Evaluation and control tests
- **Does**: Exercises persistent/reset/shuffled held-out policies and verifies the GRU
  and causal-transformer controls are stateful and genuinely parameter matched.
- **Does**: Requires evaluation to report fast synaptic efficacy and shuffle every
  target-owned cell and edge state together.
- **Does**: Requires training, held-out evaluation, and the local bridge to expose
  viability, quiescence, energy input/spending, and transport provenance.
- **Does**: Zeroes only fast efficacy during held-out inference to prove whether the
  trained organism causally uses online synaptic memory.
- **Does**: Restores deterministic birth sources for one held-out pass and proves both
  source/probe tables return bitwise to their live values afterward.
- **Does**: Requires edge eligibility and fast-weight saturation telemetry in training,
  held-out evaluation, and the local live bridge together with probe, backward-credit,
  output-error-credit, and rewrite telemetry.
- **Does**: Requires a multi-length warmup sweep to score one fixed token window.

### Report guard test
- **Does**: Requires finite completed summaries, matched parameter/update budgets, and
  correct reset/shuffle penalties before a winner can be named.
- **Does**: Distinguishes ordinary early learning from post-best collapse and deduplicates
  repeated evaluation updates after resume.
- **Does**: Separates stability from convergence so a non-collapsing run cannot be
  mistaken for an informative stopping horizon.
- **Does**: Separates per-run plateau evidence from comparison-level robustness so
  continued noisy learning does not automatically erase a persistent treatment effect.
- **Does**: Aligns each real exploratory intervention with validation before and after
  it, excludes virtual controls, and distinguishes survived from unstable bodies.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| SOL development | All causal, metabolic, learning, resume, bridge-clock, and control paths remain real | Removing assertions or replacing measured state with synthetic diagnostics |
