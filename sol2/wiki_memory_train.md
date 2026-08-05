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

### `counterfactual_training_pair`

Builds two incompatible temporary bindings for an identical question and choice order.
With `--paired-counterfactual`, both branches start from the same organism state and
contribute equally to one update, so a question-only or lifetime-position policy cannot
satisfy both targets.

### `paired_binding_margin_loss`

Jointly compares the two live four-label distributions. Each branch must prefer its
own passage-bound answer over its mate's incompatible answer by the configured margin.
This prevents averaged cross-entropy gradients from cancelling into a static half-
solution while leaving all internal representation and routing choices unconstrained.

### `organism_gradient_groups`

Reports gradient RMS, participating element count, and tensor count separately for the
language sensor, recall head, connectome, tissue dynamics, cell identity, and effector.
This distinguishes a weak reward from a route that receives no learning signal.

### `organism_optimizer_groups`

Partitions every organism parameter exactly once and assigns explicit learning rates
to sensor, recall, and effector groups while leaving connectome, tissue, identity, and
other parameters at the base rate. This turns measured plasticity imbalance into an
auditable experimental treatment rather than implicit gradient manipulation.

### `save_wiki_memory_checkpoint` / `load_wiki_memory_checkpoint`

Round-trip organism and attached-graft weights, optimizer moments, continuing lifetime
state, update/sample cursor, history, evaluations, corpus identity, and CPU/CUDA RNG.
The 8B frozen backbone is referenced by name and never duplicated in the checkpoint.

### `evaluate_wiki_memory`

Evaluates normal, no-exposure, zero-control, reset, wrong-passage, stream-memory lesion,
internal lesion, and paraphrase arms from matched fresh states. Held-out content causes
state transitions only and never optimizer updates.

Frozen passage and question features are cached once per evaluation. Every arm still
performs independent cellular transitions and language-head control.

### `run_training`

- Continues one lifetime lane across meta-training updates while truncating autograd at
  each optimizer boundary.
- Cycles deterministically across all meta-training questions and counterfactual values.
- Optionally accumulates gradients from a matched incompatible pair while carrying only
  the primary branch forward as the continuing lifetime lane.
- Builds both paired graphs before one backward pass and combines ordinary task loss
  with the explicit causal binding margin.
- Can train first on compact question-to-answer memory cards before restoring the full
  wiki paragraph as distractor context.
- Records RMS separation between paired control banks and paired four-label logits.
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
- The paired branch is a controlled counterfactual used only for credit assignment;
  the primary branch remains the single physical continuation after each update.
- Query-phase memory gating was added after the 50-update diagnostic showed that the
  multiple-choice prompt completely overwrote the 16-slot exposure FIFO. The question
  still drives input and recurrent tissue, but cannot destroy the passage slots.
- Evaluation starts each question from a matched fresh cellular state. This isolates
  rapid exposure memory from incidental history accumulated by the training lane.
- Classification loss is restricted to four frozen-head label logits. The backbone
  stays fully frozen and is never serialized into organism checkpoints.
- Causal evaluation caches detached frozen features because recomputing identical Llama
  representations for eight arms dominated the two-update pilot runtime.
- L0-C1c uses paired separation as an early routing gate: aggregate accuracy is not
  treated as memory evidence when incompatible passages emit indistinguishable controls.

## Contracts

- Source hashes and corpus SHA-256 match before loading or resuming.
- Schedule position is exactly the checkpoint update count.
- Held-out evaluation performs no optimizer step.
- Backbone trainable parameters and gradient tensors remain zero.
- All result-bearing GPU commands expose only the physical RTX 4090 UUID.
- Checkpoint writes are atomic and resume the exact continuing cellular state.
- A paired update reports exact control/logit separation; zero separation identifies a
  passage-insensitive shortcut even if one branch or held-out accuracy improves.
- Binding preference, margin loss, task loss, and combined objective are checkpointed
  as separate telemetry; the reported task loss remains comparable to earlier runs.
- Compact bindings alter meta-training exposure only. Development and held-out wiki
  cards remain frozen, natural, and source-family disjoint.
- Optimizer groups preserve full parameter coverage and serialize their names, rates,
  parameter counts, and moments in checkpoints/results.

## Example

```bash
CUDA_VISIBLE_DEVICES=GPU-21d45575-7ece-a97c-35a0-294f7bce9c39 \
python -m sol2.wiki_memory_train \
  --model NousResearch/Meta-Llama-3-8B-Instruct --local-files-only \
  --baseline data/dmon-l0/l0c1-baseline.json \
  --paired-counterfactual --updates 300 --out-dir data/dmon-l0/l0c1-train
```
