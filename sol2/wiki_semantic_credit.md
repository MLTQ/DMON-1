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

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `wiki_memory_train.py` | Four ordered outputs: loss, student RMS, alignment, raw teacher RMS | Return order or reduction |
| `test_wiki_memory.py` | Seed reproducibility, detached targets, opposing endpoint gradients, swap invariance | RNG, detach, or sign convention |
| C1t protocol | Memory seed `307`, relay seed `308`, target RMS `0.25` | Projection construction or normalization |

## Notes

Targets and projections are never registered on the organism, saved in its weights,
or used during development, held-out evaluation, or inference.
