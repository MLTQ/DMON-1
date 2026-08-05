# L0-C1u: direct semantic credit at content-addressed recall

Status: preregistered before implementation and optimizer initialization on 2026-08-05.

## Question

Can question-conditioned semantic credit preserve learning in the organism's actual
content-addressed recall vector, before recurrent and relay dilution? C1t weakly shaped
post-exposure memory but its relay target did not teach retrieval, and recall gradient
RMS collapsed by roughly 60x from the early to late matched window.

## Preserved inference pathway

```text
passage text -> frozen Qwen contextual states -> sensors -> stream memory
question text -> frozen Qwen contextual states -> content-addressed recall
recall -> input/recurrent/relay/output tissue -> continuous prefix
continuous prefix -> frozen Qwen transformer/head -> answer
```

The new observation is the exact live recall vector already injected into input tissue
on the final question token. Capturing it must not recompute recall, alter state, add a
readout, or expose a teacher at inference. The passage remains erased from Qwen's answer
context; only organism state crosses that boundary.

## Differential targets

Retain C1t's post-exposure memory target with fixed projection seed `307` and target
RMS `0.25`. Replace the post-question relay target with a direct recall target using
projection seed `309`:

```text
teacher_effect(P,Q) = qwen_feature(P + Q) - qwen_feature(Q)
teacher_recall_delta = project_309(effect(left,Q) - effect(right,Q))
student_recall_delta = recalled(left,Q) - recalled(right,Q)
recall_loss = MSE(student_recall_delta,
                  normalize(teacher_recall_delta, RMS=0.25))
```

The teacher features and projected axis are detached. Only the paired difference is
constrained. The final-token recall vectors remain connected to the existing recall
query/key/value/output parameters and to the exposure memory cells; the target never
enters recurrent state or the frozen language model.

## Fixed treatment

- Fresh seed 7 organism and optimizer; do not resume C1t weights.
- Frozen `Qwen/Qwen3.5-4B`, BF16, local files only, on physical RTX 4090 UUID
  `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39` only. The 2070S is excluded.
- Preserve C1t's 8/96/64/16/8 cells, hidden 96, 12/8 dendrites, three microsteps,
  four rank-8 anchored prefix controls, attentive relay tract, and initial tract gate
  `0.25`.
- Preserve compact paired counterfactuals, permutation seed 101, LR `0.001`, clip 1,
  and recall/sensor/effector LR multipliers 20/4/1.
- Preserve reference-centered prefix controls.
- Objective weights: passage-effect reverse KL 1, causal contrast 1, memory semantic
  credit 4, direct recall semantic credit 4. Relay semantic credit, absolute task NLL,
  absolute teacher KL, control energy, binding, output-code, and eligibility losses are
  0.
- Semantic target RMS is `0.25`; base semantic seed is `307`, making direct recall seed
  `309`.
- Run all CPU contracts, then a fresh one-update GPU preflight. A pass licenses one
  fresh 25-update run. Checkpoint and log every update; evaluate development and heldout
  only at update 25.

## Gates

### Mechanics and one-update preflight

- Recall capture is observational: enabled and disabled scoring produce exactly equal
  controls, logits, state, and health; capture returns the exact vector injected by the
  existing recall path and is absent when recall is not used.
- Fixed vector credit is deterministic, target-detached, branch-swap invariant, sends
  opposing gradients to paired recall vectors, and is zero for an aligned difference.
- A direct recall loss sends finite nonzero gradients to each of recall query, key,
  value, and output parameters and to live exposure memory; it does not recruit the
  effector through a shortcut.
- Qwen has zero trainable parameters and zero gradient tensors. All optimizer state,
  continuing state, metrics, and checkpoints remain finite.

### Update-25 representation gate

- Across recurrence-matched positions 1-12 versus 13-24, both memory and direct recall
  semantic losses improve in aggregate and their trailing mean alignments exceed
  `0.25`.
- Paired memory and recall separation remain nonzero. Recall query/key/value/output
  gradients remain finite and nonzero in the late window; early/late ratios are
  reported rather than replaced by aggregate gradient norm.
- All four late paired causal advantages are positive more often than not.
- Correct passage beats no exposure, zero control, reset, wrong passage, memory lesion,
  and internal lesion on held-out target likelihood. Eight-question accuracy remains
  descriptive and cannot override this continuous causal ordering.

Failure at recall alignment means the content-addressed operation or its memory-slot
representation needs redesign before scale. Recall alignment without downstream causal
control instead isolates recurrent transport as the next bottleneck. Passing both
licenses a longer persistence run and annealing of developmental credit; it is not yet
a general language-capability claim.
