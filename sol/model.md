# `model.py`

## Purpose

Implements the first SOL organism: homogeneous recurrent cells with private persistent
state, explicit directed dendrites, streamed character input, character output, metabolic
state, and reward-addressable eligibility memory.

## Components

### `SolConfig`
- **Does**: Holds field size, directed topology, recurrence, eligibility, and energy
  constants.
- **Rationale**: All scientific knobs travel with the model rather than hiding in a CLI.

### `FieldState`
- **Does**: Carries hidden state, energy, causal stimulation, eligibility, sensory memory,
  and delayed reward across every character and optimizer window.
- **Rationale**: Detaching a graph must not reset the organism.

### `FieldTrace`
- **Does**: Retains token-local hidden states and measured traffic for post-backward
  credit inspection.

### `SparseAxonField`
- **Does**: Applies one shared GRU rule at every cell while messages travel only through
  named source-to-target dendrites.
- **Interacts with**: `ContinuousTrainer` in `train.py`.

### `tick`
- **Does**: Consumes one optional streamed character, performs recurrent graph updates,
  updates eligibility and energy, and emits next-character logits.
- **Rationale**: `token=None` remains a live interval and proves energy depletion without
  stimulation.

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
- Energy currently modulates computation but does not kill or reproduce cells.
- The forced sensory-to-output axons are organ plumbing, not a learned language-specific
  connectome; all synaptic signs and strengths remain trainable.
