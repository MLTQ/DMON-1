# L0-C1p: reference-centered memory delta distillation

Status: protocol frozen 2026-08-05 before implementation tests or optimizer updates.

## Question

Can the anchored pre-transformer interface learn passage-specific control when dense
credit describes the desired frozen-Llama distribution and the organism is forbidden
from expressing its passage-independent common mode?

This is a test of credit geometry, not a claim that distillation is the final learning
mechanism for a living organism. Episodic reinforcement learning remains a later tool
for nondifferentiable, delayed decisions after a causal memory-to-language path exists.

## Treatment

For every paired counterfactual training record:

1. Run frozen Llama with the temporary passage and erased-context question together.
   Retain its detached full next-token distribution as the teacher.
2. Run SOL2 from the same lifetime state without passage exposure to obtain a detached
   homeostatic output-organ reference for that question.
3. Expose SOL2 to the passage, erase it from Llama's workspace, process the identical
   question, and inject only `exposed_control - homeostatic_control` as the anchored
   prefix residual.
4. Train SOL2 with full-vocabulary reverse KL to the passage-visible teacher, absolute
   answer-label NLL, paired passage-causal contrast, and an RMS energy price on the
   expressed delta.

The reference subtraction is an experimental common-mode rejection mechanism. It
requires a matched counterfactual organism pass and is not presented as final anatomy.
If it succeeds, a later treatment may internalize the reference as a learned
homeostatic/gating circuit.

## Fixed pilot

- Fresh seed 7 organism; do not resume C1o's common-mode checkpoint.
- Frozen `NousResearch/Meta-Llama-3-8B-Instruct`, BF16, local files only.
- Physical RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39` only.
- C1o topology: 8/96/64/16/8 typed cells, hidden 96, 12/8 dendrites, three
  microsteps, four rank-8 prefix controls, and attentive relay-output tract at gate
  0.25.
- Compact paired counterfactual memory cards; 256-token passage and question limits.
- Base LR 0.001; recall/sensor/effector multipliers 20/4/1; clip norm 1.
- Objective weights: teacher distillation 1, absolute task NLL 1, causal contrast 1,
  binding/output-code/eligibility 0, and delta energy 0.1. Distillation temperature 1.
- One-update finite preflight first. Only a finite pass may continue to update 50.

## Gates

### Implementation

- CPU tests prove full-vocabulary reverse KL is zero for matching distributions and
  sends gradients only into the student.
- Reference-centering makes a recomputed matched no-exposure prefix numerically zero,
  while the reported zero-scaled no-exposure arm is exact zero and exposed controls
  retain gradients.
- Passage-visible teacher inputs and passage-erased student inputs are structurally
  distinct and use the identical deterministic question permutation.
- Configuration, telemetry, checkpoint resume, and non-finite pre-step rejection cover
  all new objective terms.

### One-update GPU preflight

- Every organism gradient, parameter, optimizer state, and continuing state is finite.
- Llama has zero trainable parameters and zero gradient tensors.
- The run fits the 4090 with safe allocator headroom and leaves the 2070S untouched.
- Teacher distributions differ across the paired incompatible passages and teacher
  label accuracy is reported; a failed teacher makes the treatment uninterpretable.

### Update-50 representation gate

- The late ten-update mean normal delta has material nonzero RMS without becoming an
  undifferentiated common control.
- Full-vocabulary KL and absolute answer NLL improve over the early window.
- All four late-window causal advantages are positive more often than not, including
  both correct-passage-over-wrong-passage terms.
- Correct-passage target likelihood beats matched zero delta, no exposure, and wrong
  passage. Discrete accuracy is secondary at this small sample size.

Passing this gate licenses a longer before-mode curriculum. Failing it diagnoses this
specific reference/distillation construction; it does not reject pre-transformer or
multi-depth organism control.

## Result

Status: stopped at the one-update gate on 2026-08-05; no 50-update extension.

The corrected BF16 preflight completed on the RTX 4090 with a finite checkpoint,
14.258 clipped pre-step gradient norm, zero trainable or gradient-bearing Llama
parameters, and safe peak allocation. The initial reference-centered control was
exactly zero and the passage-visible teacher distributions were distinct (0.4817 RMS
full-logit separation), so the implementation mechanics passed.

The teacher-validity gate failed. Frozen Llama selected the designated label for only
one of the two incompatible temporary bindings (50%). After the single update,
held-out normal, no-exposure, zero-control, and wrong-passage accuracy were all 62.5%.
Normal target loss was 1.1143 versus 1.0520 for the matched zero/no-exposure floor;
wrong-passage loss was 1.1417. The expressed delta reached only 0.001447 RMS and did
not establish passage identity.

This result rejects promotion of this particular Llama teacher treatment. It does not
reject dense distributional credit: an oracle that contradicts the supervised answer
is not a valid target. Before repairing the teacher, C1q audits newer, smaller language
organs and explicitly compares foundation and post-trained checkpoints.
