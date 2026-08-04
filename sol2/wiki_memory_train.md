# `wiki_memory_train.py`

## Purpose

Runs checkpointed L0-C1 meta-training on variable wiki-linked bindings and evaluates
unchanged source-family-disjoint development and held-out facts under causal controls.

## Components

### `WikiMemoryTrainConfig`

Defines update count, optimizer strength, evaluation/checkpoint intervals, token limits,
permutation seed, and reproducibility seed.

### `counterfactual_training_record`

Appends an episode-specific temporary erratum to a meta-training memory card and changes
the target choice deterministically each epoch. Static Llama knowledge cannot solve the
varying binding.

### `save_wiki_memory_checkpoint` / `load_wiki_memory_checkpoint`

Round-trip organism and attached-graft weights, optimizer moments, continuing lifetime
state, update/sample cursor, history, evaluations, corpus identity, and CPU/CUDA RNG.
The 8B frozen backbone is referenced by name and never duplicated in the checkpoint.

### `evaluate_wiki_memory`

Evaluates normal, no-exposure, zero-control, reset, wrong-passage, stream-memory lesion,
internal lesion, and paraphrase arms from matched fresh states. Held-out content causes
state transitions only and never optimizer updates.

### `run_training`

- Continues one lifetime lane across meta-training updates while truncating autograd at
  each optimizer boundary.
- Cycles deterministically across all meta-training questions and counterfactual values.
- Evaluates development data at frozen intervals and saves atomic checkpoints.
- Evaluates held-out data once after the final update and writes complete JSON telemetry.

## Decisions

- Counterfactual annotations were added only after the committed baseline showed that
  static Llama already solved 75% of the original meta-training questions. The
  amendment was frozen before any optimizer update.
- Development and held-out memory cards remain natural, unmodified wiki-derived facts.
  Transfer to them is deliberately harder than memorizing the counterfactual syntax.
- One persistent lane is a closer model of a continuing creature than independent
  minibatches. It also creates interference pressure that a useful memory operation
  must learn to manage.
- Evaluation starts each question from a matched fresh cellular state. This isolates
  rapid exposure memory from incidental history accumulated by the training lane.
- Classification loss is restricted to four frozen-head label logits. The backbone
  stays fully frozen and is never serialized into organism checkpoints.

## Contracts

- Source hashes and corpus SHA-256 match before loading or resuming.
- Schedule position is exactly the checkpoint update count.
- Held-out evaluation performs no optimizer step.
- Backbone trainable parameters and gradient tensors remain zero.
- All result-bearing GPU commands expose only the physical RTX 4090 UUID.
- Checkpoint writes are atomic and resume the exact continuing cellular state.

## Example

```bash
CUDA_VISIBLE_DEVICES=GPU-21d45575-7ece-a97c-35a0-294f7bce9c39 \
python -m sol2.wiki_memory_train \
  --model NousResearch/Meta-Llama-3-8B-Instruct --local-files-only \
  --baseline data/dmon-l0/l0c1-baseline.json \
  --updates 300 --out-dir data/dmon-l0/l0c1-train
```

