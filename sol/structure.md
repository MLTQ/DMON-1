# `structure.py`

## Purpose

Runs bounded non-differentiable connectome changes between truncated-BPTT windows.
Targets compare mature dendrites with one causally measured candidate while preserving
fixed fan-in, reachability, persistent state, optimizer integrity, and energy provenance.

## Components

### `StructuralConfig`
- **Does**: Defines phase cadence, credit memory, replacement budget, edge maturity,
  improvement margin, required consecutive confirmation phases, endpoint energy cost,
  whether global predictive fitness is required, and whether matched probes may actually
  rewrite topology.
- **Does**: Optionally defines a post-graft probation duration and minimum observed
  prequential improvement for committing provisional anatomy, plus the developmental
  baseline decay used to distinguish graft benefit from ordinary learning.
- **Does**: Can instead request exploratory-traffic probation, which leaves the
  incumbent anatomy live and tests the selected candidate probe on and off inside one
  continuously adapting organism.
- **Interacts with**: `ContinuousTrainer` in `train.py`.

### `next_probe_sources`
- **Does**: Deterministically rotates every target to a source not already in its
  dendrite fan, walking the cell IDs round-robin rather than repeatedly sampling a
  topology-dependent subset.
- **Rationale**: Candidate exploration remains reproducible and never duplicates an
  incumbent edge.

### `accumulate_structural_credit`
- **Does**: Retains reward-addressed incumbent and causal-probe evidence across
  optimizer windows together with each probe's first-order global loss benefit.
- **Interacts with**: Structural evidence in `FieldTrace`.

### `apply_structural_phase`
- **Does**: Replaces at most a bounded number of mature low-credit edges when a probe is
  better across every required confirmation phase, both endpoints can pay, and complete
  sensory/output reachability survives.
- **Does**: Initializes the reused slow weight to the probe's approximate message
  coefficient, while resetting that slot's bias, optimizer moments, eligibility, and
  fast efficacy before rotating probes.
- **Rationale**: Fixed tensor shapes let morphology change without rebuilding the
  optimizer or discarding the organism's other memories.
- **Rationale**: A successful candidate must not lose its causal traffic at the moment
  it becomes a permanent dendrite.

### `StructuralProbation`
- **Does**: Backs up one incumbent source, edge parameters, optimizer moments,
  structural credit/age, and stream-local edge eligibility/fast efficacy before a
  provisional graft.
- **Does**: Accumulates signed prequential reward while the entire organism continues
  ordinary training, subtracts the pre-graft developmental reward baseline, then commits
  positive improvement or restores the backed-up slot.
- **Does**: Serializes active and historical probation state for exact checkpoint
  resume; probes-only uses the same waiting cadence as a virtual probation without
  topology mutation.
- **Does**: In exploratory-traffic mode, assigns streamed windows in a balanced ABBA
  candidate-on/incumbent-only schedule, retains separate reward aggregates, and grafts
  only after the candidate wins within the same organism.
- **Rationale**: Failed anatomy can be removed without pretending the rest of the body
  did not experience and adapt during the trial.
- **Rationale**: A frozen organ in a separate organism cannot serve as the primary
  fitness signal when body and organ develop together.

### `structural_summary`
- **Does**: Reports policy, rewrite attempts, retained credit, edge age, active probes,
  and probation start/commit/rollback telemetry.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train.py` | Structural phases run only after a differentiable window and optimizer step | Phase ordering |
| `benchmark.py` | Fixed and growing runs retain identical parameter counts and fan-in | Resizing tensors |
| Checkpoint resume | Model buffers and trainer policy preserve exact future rewiring | Buffer or config keys |
| Topology guard | All cells and outputs remain reachable from sensory cells | Removing viability test |

## Notes

- The probes-only control accumulates and rotates the same candidates on the same
  schedule, including the same multi-phase hold time, while leaving `sources` unchanged.
- Probe credit comes from an explicit with/without-probe cell-rule counterfactual in
  `model.py`; it is not inferred from geometric proximity.
- Structural evidence retains the sign of reward times eligibility; a large harmful
  intervention cannot qualify merely because its magnitude is large.
- When global fitness gating is enabled, local credit cannot earn a confirmation unless
  the same causal probe also points against the exact sequence-loss gradient after
  ordinary backpropagation.
- A candidate must retain positive credit and clear a nonzero advantage margin before
  it earns one confirmation. It must do so in every consecutive confirmation phase
  before it can replace even the target's worst mature incumbent; one adverse phase
  resets the streak.
- Growth consumes energy and never creates it. Rewiring remains disabled by default.
- Rollback does not refund growth energy or revert unrelated model/state changes:
  probation is anatomical reversibility, not erasure of lived experience.
- Exploratory-traffic rejection never mutates anatomy or spends growth energy. The
  incumbent, hidden state, body parameters, and optimizer remain live in both arms; only
  the selected weak probe is gated, while every other exploratory probe continues.
- Endpoint energy is checked again at the actual commit boundary; developmental success
  cannot force a graft that the organism can no longer afford.
