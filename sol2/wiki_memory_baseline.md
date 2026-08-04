# `wiki_memory_baseline.py`

## Purpose

Runs the preregistered frozen-Llama admission screen for all L0-C1 questions before
organism training.

The default corpus is `experiments/l0c1-wiki-memory-corpus.json`.

## Components

### `run_baseline`

- Verifies every wiki source hash.
- Loads the frozen Hugging Face model and validates one-token answer labels.
- Scores each deterministic post-erasure question without passage exposure or organism
  control.
- Marks only incorrect baseline questions as admitted missing-information probes.
- Records split accuracy, exact label log-probabilities, model/device identity, corpus
  hash, and peak CUDA memory.

### CLI

Accepts model, corpus, wiki root, dtype, device, permutation seed, and output path.

## Decisions

- Screening uses one fixed preregistered answer permutation. Later training and
  paraphrase evaluation use different deterministic seeds.
- Baseline failure is an explicit selection criterion. Results may establish a memory
  capability over missing facts but cannot estimate generic question-answering quality.
- A SOL2 shell is constructed only to reuse the exact frozen-backbone interface; no
  organism state or control participates in baseline logits.

## Contracts

- All source hashes pass before the model loads.
- The frozen backbone has zero trainable parameters.
- `admitted` is exactly the negation of baseline correctness.
- The result records the process GPU visibility mask; result-bearing runs must expose
  only the physical RTX 4090 UUID.

## Example

```bash
CUDA_VISIBLE_DEVICES=GPU-21d45575-7ece-a97c-35a0-294f7bce9c39 \
python -m sol2.wiki_memory_baseline \
  --model NousResearch/Meta-Llama-3-8B-Instruct --local-files-only \
  --out data/dmon-l0/l0c1-baseline.json
```
