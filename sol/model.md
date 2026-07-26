# `model.py`

## Purpose

Implements the first SOL organism: homogeneous recurrent cells with private persistent
state, explicit directed dendrites, streamed character input, character output, metabolic
state, reward-addressable eligibility memory, bounded fast synaptic efficacy, and
causally measured exploratory axon probes.

## Components

### `SolConfig`
- **Does**: Holds field size, directed topology, recurrence, eligibility, and energy
  constants.
- **Rationale**: All scientific knobs travel with the model rather than hiding in a CLI.

### `FieldState`
- **Does**: Carries hidden state, energy, causal stimulation, eligibility, sensory memory,
  delayed reward, per-edge and candidate-probe eligibility, and per-stream fast weights
  across every character and optimizer window.
- **Rationale**: Detaching a graph must not reset the organism.
- **Compatibility**: `state_from_snapshot` initializes additive plasticity fields to
  zero when loading checkpoints made before fast synaptic or structural memory existed.

### `FieldTrace`
- **Does**: Retains token-local hidden states and measured traffic for post-backward
  credit inspection, including edge eligibility, fast-weight saturation, and
  reward-addressed incumbent/candidate structural evidence.

### `SparseAxonField`
- **Does**: Applies one shared GRU rule at every cell while messages travel only through
  named source-to-target dendrites.
- **Interacts with**: `ContinuousTrainer` in `train.py`.

### Structural buffers and `load_compatible_state_dict`
- **Does**: Persist current candidate sources, incumbent/candidate credit, edge age, and
  per-candidate confirmation streaks, and total rewrites with the model while upgrading
  older checkpoints additively.
- **Interacts with**: `apply_structural_phase` in `structure.py`.

### `birth_sources`
- **Does**: Reconstructs the deterministic pre-growth source table for a causal
  morphology ablation without storing another full graph.
- **Interacts with**: `evaluate_state_ablations` in `evaluate.py`.

### `_probe_message`
- **Does**: Adds one weak candidate source per target and measures its causal effect by
  comparing the shared cell rule with and without that message.
- **Rationale**: Growth evidence comes from an intervention, not geometric proximity or
  an all-to-all correlation.

### `tick`
- **Does**: Consumes one optional streamed character, performs recurrent graph updates,
  tags individual dendrites and causal probes from local activity, applies delayed
  reward to remembered tags, updates energy, and emits next-character logits.
- **Rationale**: `token=None` remains a live interval and proves energy depletion without
  stimulation.
- **Reward contract**: A pending reward is consumed once. `forward_sequence` stores the
  newest score after each prediction, so the final score crosses an optimizer boundary
  exactly once while free generation cannot replay stale prompt reward.
- **Reward baseline**: Each stream lane retains an exponential moving expectation of
  surprise. Observed error becomes signed reward relative to that expectation, avoiding
  permanently positive potentiation once the model merely beats chance.
- **Ablation contract**: `allow_fast_plasticity=False` forces zero efficacy through the
  tick without disabling edge-tag measurement or the rest of the recurrent field.

### `forward_sequence`
- **Does**: Runs truncated differentiable windows without resetting field state.
- **Rationale**: Exact BPTT handles within-window credit; scalar reward gates persistent
  eligibility on the following tick, including across truncation boundaries.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train.py` | Tokens and logits use `(batch, time, ...)`; returned state remains attached | Shapes, return arity |
| Tests and diagnostics | `sources[target, slot]` is the authoritative fixed-fan-in directed graph | Topology representation |
| `structure.py` | Probe sources are non-edges and structural buffers retain fixed shapes | Buffer shapes |
| Future modalities | A modality injects cell-aligned stimulus and causal stimulation | Replacing `tick` semantics |

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
- Energy currently modulates computation but does not kill or reproduce cells.
- The forced sensory-to-output axons are organ plumbing, not a learned language-specific
  connectome; all synaptic signs and strengths remain trainable.
