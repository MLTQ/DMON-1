# `wiki_coherent_value.py`

## Purpose

Provides training-only local credit for coherent answer-value recall and sparse
transport through relay and output tissue. Targets are observations of the organism's
own stored sensory state, never an added inference input or learned readout.

## Components

### `designated_answer_token_mask`

- **Does**: Locates the exact designated-answer tokens in the exposure IDs, preferring
  tokenizer character offsets and using deterministic token-subsequence matching only
  when offsets are unavailable.
- **Interacts with**: Compact counterfactual cards from `wiki_memory_train.py`.

### `answer_memory_slots`

- **Does**: Maps answer token positions into physical circular FIFO slots and rejects
  an answer span that has already been overwritten.
- **Rationale**: Credit must observe the same memory neurons used by live recall.

### `coherent_answer_value_target`

- **Does**: Mean-pools exact answer slots, applies the ordinary bounded recall gain,
  and detaches the resulting same-coordinate target.

### `coherent_recall_value_credit`

- **Does**: Aligns the live final-token recall vector with the detached stored-answer
  target and reports cosine alignment plus student/target RMS.

### `paired_coherent_recall_delta_credit`

- **Does**: Aligns the live incompatible-passage recall difference with the exact
  detached difference between their stored answer values; reports alignment, RMS, and
  the fraction of target separation retained.
- **Rationale**: Prevents the absolute target's large common component from rewarding
  a passage-insensitive recall vector.

### `fixed_rms_paired_targets`

- **Does**: Rescales each detached counterfactual difference to one declared RMS while
  preserving the pair midpoint, direction, branch order, and swap symmetry.
- **Rationale**: Separates learning-signal scale from question-dependent Qwen feature
  magnitude without inventing a new representation or inference input.

### `sparse_coherent_transport_credit`

- **Does**: Soft-selects small relay and output subpopulations by target similarity,
  aligns both pooled live states with the detached answer value, and reports alignment,
  effective-cell counts, and carried-value RMS.
- **Rationale**: Some neurons may remain plastic or unused; the objective asks for a
  useful specialized route without forcing every cell into the same representation.

### `paired_sparse_coherent_transport_credit`

- **Does**: Computes each cell's live counterfactual state difference, selects cells by
  alignment with the exact answer-value delta, and aligns sparse relay/output pools.
- **Rationale**: Differential utility, rather than absolute background similarity,
  determines which cells specialize.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `wiki_memory_train.py` | Exact mask, detached target, ordered credit telemetry | Span semantics or tuple order |
| `test_wiki_memory.py` | Offset/fallback exactness, circular-slot safety, detached targets, sparse gradients | Detach or pooling rule |
| C1y/C1z protocols | No target enters cellular state, LLM input, evaluation, or inference | Adding a runtime teaching route |

## Notes

The output-pool term is downstream of the existing relay-output tract, so an enabled
tract receives developmental gradients without exposing memory directly to the LLM.
Paired losses are invariant to swapping the incompatible branches.
