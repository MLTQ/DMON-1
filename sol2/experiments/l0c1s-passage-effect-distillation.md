# L0-C1s: passage-effect distillation

Status: completed at update 25 on 2026-08-05; representation gate failed with a
partial output-effect result.

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

## Result

The one-update preflight passed. The frozen Qwen backbone retained zero trainable
parameters and zero gradient tensors, all organism state and gradients were finite,
and peak CUDA allocation/reservation was 14.35/16.72 GiB on the 4090. The 25-update
run then completed the frozen protocol without touching the 2070S. Its raw artifact
is `l0c1s-effect-s7-u25-result.json` with SHA-256
`a084c86a00e931e3dc20d6eec415fe047bf6f41b2980e6919781a5cf0d9c6ae3`.

Effect-only distillation did prevent the immediate *vocabulary-effect* silence seen
under C1r's absolute target. Over updates 16--25, student-effect RMS was `0.02914` and
paired student-effect separation was `0.02922`, versus exact zero at initialization.
Eight of twelve recurrence-matched questions reduced effect KL on their second
appearance. This establishes that a very small organism prefix can produce a
measurable, passage-dependent change in Qwen's vocabulary distribution.

It did not pass the causal representation gate:

- mean recurrence-matched effect KL was `0.9822` on first appearances and `0.9969`
  on second appearances because four regressions outweighed the eight improvements;
- late correct-passage advantages over no exposure and the incompatible passage were
  positive on only 20--40% of updates, with mean advantages
  `[0.01669, 0.00937, -0.00518, 0.00518]`;
- effective control RMS fell from `6.18e-4` over updates 1--10 to `9.77e-7` over
  updates 16--25, and paired control separation fell from `2.53e-5` to `2.38e-7`;
- recall gradient RMS fell from `8.19e-6` early to `3.13e-8` late and sensor gradient
  from `1.53e-4` to `7.29e-7`, while effector gradient increased from `0.0364` to
  `0.0574`;
- held-out correct-passage accuracy was 75%, below the 87.5% no-exposure and
  zero-control baselines. Correct passage had mean loss `0.5938` versus `0.5511` for
  either baseline, and beat the wrong passage on loss for only 3/8 questions.

The nonzero vocabulary effect alongside microscopic late controls is consistent with
Qwen amplifying a tiny prefix perturbation. It is not evidence that the internal
memory route remained useful: zero/reset baselines were better, and memory/internal
lesions were not harmful. The next treatment should therefore add direct local credit
to the memory-to-relay route and test whether a compact, question-conditioned semantic
effect reaches the output organ. Merely increasing organism size would add depth and
capacity to a route whose upstream gradients are already disappearing.

This single bounded failure rejects C1s as the immediate scale-up recipe; it does not
show that passage-effect distillation is intrinsically useless. Its successful
passage-dependent output effect is a component worth retaining in a structurally
better-credited treatment.
