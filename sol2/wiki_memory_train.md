# `wiki_memory_train.py`

## Purpose

Runs checkpointed L0-C1 meta-training on variable wiki-linked bindings and evaluates
unchanged source-family-disjoint development and held-out facts under causal controls.

## Components

### `WikiMemoryTrainConfig`

Defines update count, optimizer strength, evaluation/checkpoint intervals, token limits,
permutation seed, reproducibility seed, and the explicit bounded control gain used by
the language graft, including serialized late-residual/prefix mode, dense-teacher,
passage-effect, control-energy, and reference-centering treatments.
Local memory/relay semantic-credit weights, fixed projection seed, and target RMS are
also serialized so developmental scaffolding cannot drift across resume. The lifetime
lane policy explicitly distinguishes continuous state inheritance from fresh episodic
training populations. Coherent-recall mode, residual gain, sparse top-k, and recency
bias are likewise serialized architectural treatments.

### `training_episode_start_state`

Selects the cellular start for one optimizer example. `continuous` returns the supplied
lane; `fresh_episode` creates zero hidden state and cursor while preserving batch,
device, dtype, anatomy, and the current weight-version marker.

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

### Causal passage contrast

When explicitly enabled, reruns the identical question without exposure and rewards
each paired passage only for raising its own target log probability above both that
no-exposure baseline and the incompatible passage. Baseline logits are detached, so
the objective cannot win by damaging a control arm. `--task-weight 0` makes this the
only behavioral reward rather than an auxiliary to ordinary answer fitting.

### Frozen-effector developmental intervention

`--freeze-effector-updates` holds the already calibrated language output organ and
control basis fixed while gradients continue through them into output and internal
tissue. This prevents further question-to-label learning in the head without blocking
internal tissue from learning to drive the existing interface.

### Fixed output-tissue credit

When explicitly enabled, scores final output-cell state against a deterministic
parameter-free four-label codebook. This delivers paired passage-target credit at the
anatomical bottleneck without adding any evaluation or inference path.

### Eligibility-gated differential transport

When explicitly enabled, aligns the paired output-state difference with the fixed
incompatible answer-code axis. Detached relay-state separation gates the objective, so
credit is present only when presynaptic tissue carries a passage distinction. The
signed paired projection gives identical output endpoints opposing nonzero gradients,
avoiding L0-C1h's shared-mixture solution.

### Optional attentive relay tract

Builds the organism with a per-output-cell gated cross-attention tract over relay tissue
when requested. Gate magnitude and a separate `transport` gradient group are reported
so opening a new anatomical path cannot be inferred from task loss alone.
The serialized initial gate may be nonzero for explicitly preregistered developmental
treatments; zero remains the backward-compatible default.

### Pre-transformer prefix treatment

`--language-control-mode prefix` preserves detached Llama perception but performs a
second differentiable frozen forward with the final organism control bank prepended to
the erased-context question. This lets every transformer layer reason over creature
output while keeping the ordinary late residual as the backward-compatible default.

### Passage-visible teacher distillation

When explicitly enabled, a detached frozen-Llama teacher sees the temporary passage and
the identical permuted question in one context. The passage-erased DMON student retains
its full next-token vocabulary logits and minimizes temperature-scaled reverse KL via
`wiki_distillation.py`. Teacher label accuracy and paired full-logit separation make an
uninformative teacher visible before a result is interpreted.

`passage_visible_teacher_logits` returns one detached `[vocabulary]` vector for the
final prompt position, matching each retained student episode vector exactly.

### Passage-effect distillation

When explicitly enabled, `question_only_teacher_logits` scores the identical formatted
question without any passage. The detached teacher passage-minus-question effect is
added to the detached student zero-prefix logits, and only the live exposed student
minimizes reverse KL to that target. This cancels the teacher's shared language prior
and makes zero control optimal only when the hidden passage has no effect.

Telemetry separately reports teacher/student effect RMS and paired effect separation.
The question-only teacher and zero-prefix student baselines never receive gradients or
enter organism state.

### Differential local semantic credit

When explicitly enabled, `wiki_semantic_credit.py` aligns two live paired tissue
differences with detached frozen-Qwen contextual differences:

- post-exposure memory tissue matches the final contextual difference between the two
  incompatible passages before any question is processed;
- post-question relay tissue matches the difference between the two passage-visible
  teacher effects relative to the same question-only baseline.

Both use deterministic parameter-free Qwen-width projections and constrain only the
paired delta, not either branch's absolute representation. The memory capture occurs
before reset/lesion/question operations. Teacher vectors, projection axes, targets,
losses, and telemetry have no evaluation or inference path.

### Direct recall semantic credit

When explicitly enabled, the trainer captures the actual content-addressed vector used
on the final question token and aligns its paired difference with the detached
passage-visible Qwen effect relative to the question-only baseline. Projection seed is
`semantic_credit_seed + 2`. This credit reaches recall query/key/value/output and live
memory before recurrent or relay dilution; it has no effector or inference bypass.

### Coherent sparse recall

The language graft may preserve stored sensory coordinates with identity-plus-residual
value/output paths, reorder circular memory chronologically, bias selection toward recent
slots, and apply top-k before softmax. Defaults retain the legacy dense learned-only
path. These are inference anatomy, not teacher shortcuts.

### Reference-centered control delta

When explicitly enabled, a detached no-exposure organism pass supplies the question's
homeostatic output-organ reference. Exposed, wrong-passage, reset, and lesion arms inject
only their control minus that reference; zero-scale and the reported no-exposure arm
remain exact zero. An optional mean-square energy term prices the expressed delta. This is
experimental common-mode rejection, not a final single-pass anatomy.

### `organism_gradient_groups`

Reports gradient RMS, participating element count, and tensor count separately for the
language sensor, recall head, connectome, tissue dynamics, cell identity, and effector.
This distinguishes a weak reward from a route that receives no learning signal.

`recall_gradient_components` additionally reports query, key, value, and output RMS
separately so aggregate recall activity cannot hide a dead addressing component.

### `require_finite_organism_gradients`

Rejects any non-finite organism gradient before clipping, optimizer mutation, lifetime
state advancement, or checkpointing. Aggregate gradient norm receives a second finite
check after clipping computes it and before the optimizer step.

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

- Either continues one lifetime lane across meta-training updates or starts each
  optimizer example from a fresh cellular lifetime, while always preserving continuous
  passage-to-question state inside an episode.
- Cycles deterministically across all meta-training questions and counterfactual values.
- Optionally accumulates gradients from a matched incompatible pair while carrying only
  the primary branch forward as the continuing lifetime lane.
- Builds both paired graphs before one backward pass and combines ordinary task loss
  with the explicit causal binding margin.
- Can train first on compact question-to-answer memory cards before restoring the full
  wiki paragraph as distractor context.
- Records RMS separation between paired control banks and paired four-label logits.
- Records training-only output-code loss/accuracy and paired output-state separation.
- Records eligibility loss, answer-axis projection, relay separation, output/relay
  transport ratio, and eligibility-gate strength.
- Records attentive-tract gate magnitude when the optional architecture is enabled.
- Records four passage-causal advantages, their positive fraction, the matched
  no-exposure task outcome, and the exact count of frozen effector parameters.
- Records full-vocabulary teacher KL, teacher label accuracy, teacher-pair separation,
  and effective-delta energy beside the existing causal metrics.
- Records passage-effect KL, teacher/student effect RMS, and paired effect separation
  when the effect-only treatment is enabled.
- Records memory/relay semantic losses, live paired separation, target cosine
  alignment, and pre-normalization teacher projection RMS when local credit is enabled.
- Records the corresponding direct-recall loss/separation/alignment and all four recall
  component gradient RMS values when recall credit is enabled.
- Records start/end memory cursors so lane-policy behavior and saturation are auditable.
- Evaluates development data at frozen intervals and saves atomic checkpoints.
- Evaluates held-out data once after the final update and writes complete JSON telemetry.

## Decisions

- Counterfactual annotations were added only after the committed baseline showed that
  static Llama already solved 75% of the original meta-training questions. The
  amendment was frozen before any optimizer update.
- Development and held-out memory cards remain natural, unmodified wiki-derived facts.
  Transfer to them is deliberately harder than memorizing the counterfactual syntax.
- Continuous and fresh-episode lanes answer different questions. Continuous state tests
  one creature under interference; fresh episodes train shared developmental rules over
  a population without carrying cellular contents across changing weight versions.
- The paired branch is a controlled counterfactual used only for credit assignment;
  the primary branch remains the single physical continuation after each update.
- Query-phase memory gating was added after the 50-update diagnostic showed that the
  multiple-choice prompt completely overwrote the 16-slot exposure FIFO. The question
  still drives input and recurrent tissue, but cannot destroy the passage slots.
- Evaluation starts each question from a matched fresh cellular state. This isolates
  rapid exposure memory from incidental history accumulated by the training lane.
- Ordinary classification loss is restricted to four frozen-head label logits;
  optional teacher credit compares the detached full vocabulary at the same position.
  The backbone stays fully frozen and is never serialized into organism checkpoints.
- Causal evaluation caches detached frozen features because recomputing identical Llama
  representations for eight arms dominated the two-update pilot runtime.
- L0-C1c uses paired separation as an early routing gate: aggregate accuracy is not
  treated as memory evidence when incompatible passages emit indistinguishable controls.
- Control gain is part of the serialized training protocol. This matters at the BF16
  boundary: a differentiable but sub-ULP residual can receive gradients while producing
  exactly identical frozen-head logits in the forward pass.
- Output credit defaults to zero for exact legacy behavior. Its codebook seed, scale,
  and weight are serialized; evaluation never constructs or consults auxiliary scores.
- Differential eligibility also defaults to zero. It reuses the fixed codebook only to
  define a paired direction; no target code or teaching drive enters cellular state.
- Dense teacher credit is a developmental scaffold: the teacher sees the passage only
  to produce detached training targets. The student and every result-bearing inference
  arm retain erased Llama context.
- Reference-centering was introduced after C1o grew a large passage-independent prefix
  that harmed the frozen-language floor. It is deliberately evaluated before attempting
  to internalize homeostasis in one organism pass.
- C1t local credit follows C1s's nonzero output-effect/vanishing-upstream result. It
  keeps Qwen contextual states as sensory input and uses differential developmental
  losses to assign credit at the time and tissue where storage and recall occur.

## Contracts

- Source hashes and corpus SHA-256 match before loading or resuming.
- Schedule position is exactly the checkpoint update count.
- Lifetime lane policy is serialized. Old checkpoints default to `continuous`; resume
  rejects a requested policy that differs from the checkpoint.
- Held-out evaluation performs no optimizer step.
- Backbone trainable parameters and gradient tensors remain zero.
- All result-bearing GPU commands expose only the physical RTX 4090 UUID.
- Checkpoint writes are atomic and preserve the last completed episode state. The
  `continuous` policy resumes it; `fresh_episode` deliberately creates a new cellular
  state at the next update while restoring weights, optimizer, RNG, and schedule.
- A paired update reports exact control/logit separation; zero separation identifies a
  passage-insensitive shortcut even if one branch or held-out accuracy improves.
- Binding preference, margin loss, task loss, and combined objective are checkpointed
  as separate telemetry; the reported task loss remains comparable to earlier runs.
- Compact bindings alter meta-training exposure only. Development and held-out wiki
  cards remain frozen, natural, and source-family disjoint.
- Optimizer groups preserve full parameter coverage and serialize their names, rates,
  parameter counts, and moments in checkpoints/results.
- Positive control gain is passed unchanged into graft construction and serialized in
  `train_config`; it does not change topology, parameter count, or backbone weights.
- Coherent recall, residual gain, top-k, and recency bias are serialized and passed
  unchanged into graft construction; sparse selection sees valid chronological memory
  only and recalled drive still enters input tissue.
- Output credit has no trainable parameters and cannot create a memory-to-Llama bypass.
- Eligibility credit has no trainable parameters, detaches its relay gate, and is absent
  from development, held-out evaluation, and inference.
- Causal passage contrast requires paired counterfactuals and has no inference path.
- Teacher logits are detached, never serialized into the organism, and never enter
  cellular state; full-vocabulary KL sends gradients only through student logits.
- Passage-effect targets detach the student baseline and both teacher paths; only the
  conditioned exposed student receives effect-loss gradients.
- Semantic projections are deterministic non-parameters. Teacher contextual features
  and target axes detach; local losses can update only live paired organism states and
  require paired counterfactual training.
- Memory semantic credit observes post-exposure tissue before question processing;
  relay semantic credit observes post-question relay tissue while the teacher's
  passage remains absent from the student language context.
- Recall semantic credit observes the exact final-question recall tensor from the
  ordinary step graph; teacher effects and projection axes remain detached, and the
  capture is absent from evaluation and inference unless explicitly requested.
- Reference controls are computed from the identical starting state and question,
  detached before subtraction, and serialized as a treatment flag rather than state.
- Delta energy prices effective language intervention after reference subtraction, not
  the organism's raw passage-independent output.
- Language control mode is checkpointed; prefix evaluation uses matched zero-prefix
  geometry and never restores the exposed passage to Llama context.
- Frozen-effector mode removes gradients only from the attached language effector;
  gradients still traverse its fixed operations into cellular output tissue.
- A non-finite gradient can never reach `optimizer.step`; the failed update leaves the
  continuing lane and checkpoint cursor unchanged.

## Example

```bash
CUDA_VISIBLE_DEVICES=GPU-21d45575-7ece-a97c-35a0-294f7bce9c39 \
python -m sol2.wiki_memory_train \
  --model NousResearch/Meta-Llama-3-8B-Instruct --local-files-only \
  --baseline data/dmon-l0/l0c1-baseline.json \
  --paired-counterfactual --updates 300 --out-dir data/dmon-l0/l0c1-train
```
