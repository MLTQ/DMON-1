# L0-C1s: passage-effect distillation

Status: protocol frozen 2026-08-05 before implementation tests or optimizer updates.

## Question

Can the organism learn a causal memory-to-language route when dense credit contains
only information added by the hidden passage? C1r showed that matching an absolute
teacher distribution lets a capable frozen Qwen prior dominate while DMON's effective
control collapses toward zero.

## Effect target

For the identical permuted question `Q` and temporary passage `P`, compute detached
frozen-teacher logits:

```text
teacher_effect(P, Q) = teacher_logits(P + Q) - teacher_logits(Q)
student_target(P, Q) = detach(student_zero_logits(Q)) + teacher_effect(P, Q)
```

The exposed, passage-erased student minimizes full-vocabulary reverse KL to this
target. Adding the teacher effect to the student's own matched zero-prefix baseline
preserves the student's language prior while transferring only the passage-induced
change. Softmax makes vocabulary-wide additive logit offsets irrelevant.

## Fixed pilot

- Fresh seed 7 organism; no C1r checkpoint resume.
- Frozen `Qwen/Qwen3.5-4B`, BF16, local files only, on physical RTX 4090 UUID
  `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39` only.
- Same 8/96/64/16/8 typed cells, hidden 96, 12/8 dendrites, three microsteps, four
  rank-8 anchored prefix controls, attentive relay tract, and initial tract gate 0.25.
- Same compact paired counterfactual schedule, permutation seed 101, LR 0.001, clip 1,
  and recall/sensor/effector LR multipliers 20/4/1.
- Reference-centered prefix controls remain enabled.
- Objective weights: passage-effect reverse KL 1, paired causal contrast 1; absolute
  task NLL 0, absolute teacher KL 0, control energy 0, and all binding/output auxiliary
  terms 0. Effect temperature and causal margin remain 1 and 0.1.
- One-update preflight first. A passing preflight licenses a fresh 25-update
  representation run with evaluation only at update 25.

## Gates

### Implementation and preflight

- A common teacher prior cancels within floating tolerance; zero teacher effect targets
  the detached student zero-prefix distribution exactly.
- Only conditioned student logits receive effect-loss gradients. Student baseline and
  both teacher paths remain detached.
- Teacher question-only inputs contain no passage annotation and use the identical
  formatted question and permutation as passage-visible inputs.
- Teacher-effect RMS and paired teacher-effect separation are finite and nonzero.
- Qwen has zero trainable parameters and zero gradient tensors; organism gradient and
  checkpoint are finite; the 2070S remains untouched.

### Update-25 representation gate

- Student-effect RMS and paired student-effect separation grow materially from zero.
- Student-effect KL improves on recurrence-matched examples rather than merely across
  different schedule positions.
- All four late paired causal advantages are positive more often than not.
- Correct passage beats zero/no exposure and wrong passage on target likelihood.
- Recall, sensor, connectome, and tissue gradients remain measurably engaged rather
  than collapsing while only the effector learns.

Failure rejects this effect-target construction. It does not justify enlarging the
organism until the causal credit route itself remains active.
