# `wiki_semantic_credit.py`

## Purpose

Provides parameter-free, training-only semantic credit at internal wiki-memory tissue.
It aligns paired cellular differences with detached frozen-Qwen contextual differences
without adding an inference route or constraining shared background state.

## Components

### `fixed_semantic_projection`

- **Does**: Reconstructs a seeded Gaussian map from frozen-language width to organism
  hidden width in float32.
- **Interacts with**: `run_training` in `wiki_memory_train.py`.
- **Rationale**: A fixed map supplies compatible dimensions without a learned readout
  that could absorb the task or become an inference dependency.

### `semantic_target_delta`

- **Does**: Projects a detached paired teacher difference and normalizes it to the
  protocol's target RMS.
- **Interacts with**: `paired_tissue_semantic_credit` and CPU gradient contracts.
- **Rationale**: Only the differential semantic axis is prescribed; common cellular
  representation remains unconstrained.

### `paired_tissue_semantic_credit`

- **Does**: Mean-pools matched tissues and applies MSE between their live difference
  and the detached semantic target; reports student separation, cosine alignment, and
  pre-normalization teacher-target RMS.
- **Interacts with**: Post-exposure memory and post-question relay tensors retained by
  `run_wiki_memory_episode`.

### `paired_vector_semantic_credit`

- **Does**: Applies the same detached paired semantic target directly to live
  `[batch, hidden]` vectors without tissue pooling.
- **Interacts with**: The exact final-token content-addressed recall vectors captured by
  `run_wiki_memory_episode` for C1u.
- **Rationale**: Credit reaches recall query/key/value/output before recurrent dilution,
  while still constraining only a counterfactual difference.

### `recall_addressing_credit`

- **Does**: Builds a detached distribution over the newest stored exposure positions
  from cosine similarity between frozen-Qwen token features and the passage-visible
  teacher effect, then minimizes KL from that target to the live pre-recency content
  scores.
- **Reports**: Teacher mass admitted by sparse selection, live attention entropy and
  effective-slot count, and mass on the four newest slots.
- **Rationale**: Query/key addressing receives a direct training surface across all
  valid slots. Recency and hard top-k still govern inference, but cannot hide a bad
  content address from the loss.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `wiki_memory_train.py` | Ordered semantic-credit and addressing telemetry tuples | Return order or reduction |
| `test_wiki_memory.py` | Seed reproducibility, detached targets, opposing endpoint gradients, swap invariance | RNG, detach, or sign convention |
| C1t protocol | Memory seed `307`, relay seed `308`, target RMS `0.25` | Projection construction or normalization |
| C1u protocol | Memory seed `307`, direct recall seed `309`, target RMS `0.25` | Projection construction or normalization |

## Notes

Targets and projections are never registered on the organism, saved in its weights,
or used during development, held-out evaluation, or inference.
