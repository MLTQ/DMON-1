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

### `sparse_coherent_transport_credit`

- **Does**: Soft-selects small relay and output subpopulations by target similarity,
  aligns both pooled live states with the detached answer value, and reports alignment,
  effective-cell counts, and carried-value RMS.
- **Rationale**: Some neurons may remain plastic or unused; the objective asks for a
  useful specialized route without forcing every cell into the same representation.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `wiki_memory_train.py` | Exact mask, detached target, ordered credit telemetry | Span semantics or tuple order |
| `test_wiki_memory.py` | Offset/fallback exactness, circular-slot safety, detached targets, sparse gradients | Detach or pooling rule |
| C1y protocol | No target enters cellular state, LLM input, evaluation, or inference | Adding a runtime teaching route |

## Notes

The output-pool term is downstream of the existing relay-output tract, so an enabled
tract receives developmental gradients without exposing memory directly to the LLM.
