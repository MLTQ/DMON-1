# L0-C1t: local semantic credit through memory and relay

Status: protocol frozen 2026-08-05 before implementation tests or optimizer updates.

## Question

Can training-only local semantic credit keep the passage-to-memory-to-relay route
plastic long enough for C1s passage-effect distillation to teach a useful Qwen prefix?
C1s created a measurable passage-dependent vocabulary effect, but recall and sensor
gradients collapsed while the effector continued learning. Correct passage consequently
underperformed the erased-context Qwen floor.

## Preserved inference pathway

The frozen Qwen language organ remains on both sides of the organism:

```text
passage text -> frozen Qwen contextual states -> sensors -> memory
question text -> frozen Qwen contextual states -> recall -> relay -> output cells
output cells -> continuous prefix -> frozen Qwen transformer/head -> answer
```

Passage text and its Qwen activations are absent from the answer-time Qwen context.
Only cellular state crosses that boundary. Every semantic target below exists only in
the training loss and is never written into organism state or supplied at inference.

## Differential local targets

Use two deterministic parameter-free Gaussian projections from Qwen width to organism
hidden width. Projection seeds are `307` for memory and `308` for relay. Projected
targets are RMS-normalized to `0.25`; only their paired difference is constrained, so
common/background cellular representation remains free.

### Exposure-local memory credit

For the two incompatible compact passages, retain each branch's final frozen-Qwen
contextual feature and its mean-pooled memory-cell state immediately after exposure,
before the question:

```text
teacher_memory_delta = project(qwen_passage_left - qwen_passage_right)
student_memory_delta = mean(memory_left) - mean(memory_right)
memory_loss = MSE(student_memory_delta, normalize(teacher_memory_delta, RMS=0.25))
```

This target cannot encode the question or answer-time logits because it is applied
before the question is observed. Swapping paired branches negates both deltas and must
leave the loss unchanged.

### Query-local relay credit

The frozen passage-visible teacher supplies final contextual features for the combined
passage and identical permuted question. Subtracting the question-only teacher defines
each passage effect; the common baseline cancels in the paired delta. Match that target
to mean-pooled relay tissue after the passage-erased organism has processed the
question:

```text
teacher_query_effect(P,Q) = qwen_feature(P + Q) - qwen_feature(Q)
teacher_relay_delta = project(effect(left,Q) - effect(right,Q))
student_relay_delta = mean(relay_left) - mean(relay_right)
relay_loss = MSE(student_relay_delta, normalize(teacher_relay_delta, RMS=0.25))
```

The organism does not receive the teacher effect. It can satisfy relay credit only by
combining stored cellular state with the Qwen-encoded question.

## Fixed treatment

- Fresh seed 7 organism and optimizer; do not resume C1s weights.
- Frozen `Qwen/Qwen3.5-4B`, BF16, local files only, on physical RTX 4090 UUID
  `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39` only.
- Preserve C1s 8/96/64/16/8 cells, hidden 96, 12/8 dendrites, three microsteps, four
  rank-8 anchored prefix controls, attentive relay tract, and initial tract gate 0.25.
- Preserve compact paired counterfactuals, permutation seed 101, LR `0.001`, clip 1,
  and recall/sensor/effector LR multipliers 20/4/1.
- Preserve reference-centered prefix controls.
- Objective weights: passage-effect reverse KL 1, causal contrast 1, memory semantic
  credit 4, relay semantic credit 4. Absolute task NLL, absolute teacher KL, control
  energy, binding, output-code, and eligibility losses remain 0.
- Semantic target RMS is `0.25`; semantic seed is `307`. These are serialized protocol
  fields, not trainable parameters.
- One-update preflight first. A pass licenses one fresh 25-update representation run,
  with development and held-out evaluation only at update 25.

## Gates

### CPU and one-update preflight

- Fixed projections reproduce exactly from their seed and have no gradients.
- Teacher features and target axes are detached. Identical student endpoints receive
  nonzero opposing gradients, swapping the pair is invariant, and aligned endpoints
  produce zero loss within tolerance.
- Exposure memory state is captured before question/reset/lesion operations and has no
  evaluation-only teaching path.
- Both teacher target axes are finite and nonzero; every intended organism gradient
  group participates; Qwen has zero trainable parameters and zero gradient tensors.
- All parameters, optimizer state, continuing state, metrics, and checkpoints remain
  finite. Peak allocation leaves 4090 headroom and the 2070S remains untouched.

### Update-25 representation gate

- Recurrence-matched memory and relay semantic losses improve in aggregate; trailing
  mean target alignment exceeds `0.25` at both stages.
- Paired memory and relay separation remain materially nonzero rather than collapsing
  at the output boundary.
- Late recall, sensor, connectome, tissue, and transport gradients remain finite and
  nonzero; interpretation reports their early/late ratios rather than hiding scale
  behind the auxiliary weights.
- All four late paired causal advantages are positive more often than not.
- Correct passage beats no exposure, zero control, reset, wrong passage, memory lesion,
  and internal lesion on target likelihood. Accuracy is descriptive on eight held-out
  questions and cannot override the continuous causal ordering.

A failure rejects this fixed local-credit cascade as the immediate scale-up treatment,
not Qwen contextual perception, organismal differentiation, or local developmental
scaffolding in general. A pass licenses an update-100 persistence test and subsequent
annealing of the local losses; it does not yet license a language-capability claim.
