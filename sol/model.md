# `model.py`

## Purpose

Implements the first SOL organism: homogeneous recurrent cells with private persistent
state, explicit directed dendrites, streamed character input, character output, metabolic
state, reward-addressable eligibility memory, and bounded fast synaptic efficacy.

## Components

### `SolConfig`
- **Does**: Holds field size, directed topology, recurrence, eligibility, and energy
  constants.
- **Rationale**: All scientific knobs travel with the model rather than hiding in a CLI.

### `FieldState`
- **Does**: Carries hidden state, energy, causal stimulation, eligibility, sensory memory,
  delayed reward, per-edge eligibility, and per-stream fast weights across every
  character and optimizer window.
- **Rationale**: Detaching a graph must not reset the organism.
- **Compatibility**: `state_from_snapshot` initializes additive plasticity fields to
  zero when loading checkpoints made before fast synaptic memory existed.

### `FieldTrace`
- **Does**: Retains token-local hidden states and measured traffic for post-backward
  credit inspection, including edge eligibility and fast-weight saturation.

### `SparseAxonField`
- **Does**: Applies one shared GRU rule at every cell while messages travel only through
  named source-to-target dendrites.
- **Interacts with**: `ContinuousTrainer` in `train.py`.

### `tick`
- **Does**: Consumes one optional streamed character, performs recurrent graph updates,
  tags individual dendrites from local pre/post activity, applies delayed reward to
  previously tagged dendrites, updates energy, and emits next-character logits.
- **Rationale**: `token=None` remains a live interval and proves energy depletion without
  stimulation.
- **Reward contract**: A pending reward is consumed once. `forward_sequence` stores the
  newest score after each prediction, so the final score crosses an optimizer boundary
  exactly once while free generation cannot replay stale prompt reward.

### `forward_sequence`
- **Does**: Runs truncated differentiable windows without resetting field state.
- **Rationale**: Exact BPTT handles within-window credit; scalar reward gates persistent
  eligibility on the following tick, including across truncation boundaries.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train.py` | Tokens and logits use `(batch, time, ...)`; returned state remains attached | Shapes, return arity |
| Tests and diagnostics | `sources[target, slot]` is the authoritative directed graph | Topology representation |
| Future modalities | A modality injects cell-aligned stimulus and causal stimulation | Replacing `tick` semantics |

## Notes

- Fixed topology is intentional for the first falsification. Axon growth comes only after
  forward transport, backward credit, and persistent event memory are demonstrated.
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
