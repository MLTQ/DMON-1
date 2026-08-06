# `wiki_memory.py`

## Purpose

Defines the provenance-aware data contract, exact multiple-choice surface, and causal
episode operations for L0-C1 held-out wiki memory.

The frozen manifest lives at `experiments/l0c1-wiki-memory-corpus.json`, beside the
preregistered protocol and eventual result artifacts.

## Components

### `WikiMemoryCorpus`, `WikiDocument`, `WikiQuestion`

- Load the frozen JSON manifest into immutable records.
- Enforce four distinct choices, unique question identifiers, valid answers, known
  splits, and source-family disjointness.

### `verify_wiki_sources`

Checks every referenced `/home/m/wiki` source against its preregistered SHA-256 hash.

### `format_wiki_question`

Builds the post-erasure prompt and deterministically permutes answer order from the
question identifier and explicit seed. The passage never appears in this prompt.

### `score_frozen_baseline`

Scores the four one-token answer labels with the frozen language model and no organism
or passage exposure. This is the pre-training admission screen; only the final frozen
feature is sent through the vocabulary head.


### `run_wiki_memory_episode`

- Optionally exposes the continuing organism to a memory card without allocating
  vocabulary logits.
- Supports no-exposure, wrong-passage, reset, stream-memory lesion, exact-zero control,
  and internal freeze/lesion arms through explicit arguments.
- Accepts cached frozen passage/question features so matched causal arms rerun only the
  organism rather than the 8B language encoder.
- In prefix mode, reuses cached features for perception but retains exact question IDs
  for the required second, prefix-conditioned frozen-transformer pass.
- Can subtract a detached no-exposure control reference at the language boundary so
  the scored intervention is only the passage-conditioned delta.
- Processes the question after a fresh language context and computes cross-entropy only
  over the four label logits.
- Closes the stream-memory write gate during the question so the prompt cannot replace
  the passage slots that the intervention is intended to test.
- Retains live control and four-label logit tensors for paired-branch separation
  measurement, optionally retains the live full-vocabulary next-token logits for dense
  teacher credit, captures live memory tissue immediately after exposure and before
  reset/question operations, optionally captures the exact final-question recall
  vector, pre-recency/final scores, and attention, and exposes final relay/output-tissue
  states for training-only causal credit while serializing only detached scalar
  telemetry.

### `summarize_results`

Aggregates exact accuracy, loss/bits, control scale, and internal/memory activity.

## Decisions

- Initial answers are one-token labels rather than generated prose. This makes grading
  exact and keeps the experiment about retrieval rather than surface fluency.
- Answer choices are present after context erasure; only the information needed to
  choose among them was supplied during exposure.
- Memory cards are concise, provenance-linked statements derived from recent wiki
  research summaries. This tests rapid state acquisition before attempting whole-
  article compression.
- Held-out documents change cellular state only. Their content cannot participate in
  optimizer updates.
- Deterministic answer permutation prevents a fixed output-position policy and is part
  of the resumable sample cursor.
- Query tokens continue through input and recurrent tissue but never overwrite the
  passage memory under test.
- Prefix controls are emitted only after the question-perception phase, implementing a
  turn-level perceive-then-express cycle without putting passage text back in context.
- Exposure-memory capture is an observation of the existing state tensor, not another
  write, readout, or inference route. It exists so developmental losses can be
  temporally assigned before question processing.

## Contracts

- Every label ` A` through ` D` must be exactly one tokenizer token.
- Source hashes must match before a result-bearing run.
- Exposure and question use separate frozen LLM calls; no passage token survives in the
  post-erasure language workspace.
- Reset changes state only, never weights.
- Exact-zero control traverses the same exposure and cellular transitions as normal.
- Cached features change performance only; token wrappers remain the training path and
  cached-feature equivalence is covered by the living-language CPU gate.
- Prefix zero-control uses the same repeated first-token anchors and virtual-token
  geometry as normal control; it is the matched floor rather than native no-prefix
  Llama.
- A supplied control reference is detached and affects only language decoding. It does
  not change exposure, cellular dynamics, memory contents, or the continuing state.
- `exposure_memory_state` is `None` without exposure and otherwise refers to live
  post-exposure memory cells before any requested reset or lesion.
- `recalled` is populated only when explicitly requested and refers to the same live
  vector injected into input tissue on the final question token; it is not recomputed.
- Source families cannot cross meta-training, development, and held-out splits.
- The initial implementation runs one lifetime lane per episode; batching variable
  passages is a throughput optimization after the causal pilot works.
