# `structure.py`

## Purpose

Runs bounded non-differentiable connectome changes between truncated-BPTT windows.
Targets compare mature dendrites with one causally measured candidate while preserving
fixed fan-in, reachability, persistent state, optimizer integrity, and energy provenance.

## Components

### `StructuralConfig`
- **Does**: Defines phase cadence, credit memory, replacement budget, edge maturity,
  improvement margin, required consecutive confirmation phases, endpoint energy cost,
  and whether matched probes may actually rewrite topology.
- **Interacts with**: `ContinuousTrainer` in `train.py`.

### `next_probe_sources`
- **Does**: Deterministically rotates every target to a source not already in its
  dendrite fan, walking the cell IDs round-robin rather than repeatedly sampling a
  topology-dependent subset.
- **Rationale**: Candidate exploration remains reproducible and never duplicates an
  incumbent edge.

### `accumulate_structural_credit`
- **Does**: Retains reward-addressed incumbent and causal-probe evidence across
  optimizer windows.
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

### `structural_summary`
- **Does**: Reports policy, rewrite count, retained credit, edge age, and active probes.

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
- A candidate must retain positive credit and clear a nonzero advantage margin before
  it earns one confirmation. It must do so in every consecutive confirmation phase
  before it can replace even the target's worst mature incumbent; one adverse phase
  resets the streak.
- Growth consumes energy and never creates it. Rewiring remains disabled by default.
