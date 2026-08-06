# L0-C1r: Qwen3.5-4B reference-centered memory delta

Status: protocol frozen 2026-08-05 before optimizer updates.

## Question

Does C1p's passage-erased organism learn a causal memory-conditioned prefix when its
dense passage-visible teacher is label-consistent? C1p established finite credit flow
but stopped because Llama obeyed only one of two temporary bindings. C1q qualified
Qwen3.5-4B on all 24 compact incompatible bindings. C1r changes only the frozen
language model needed to repair that failed validity gate.

## Fixed treatment

- Fresh seed 7 organism; no C1p/C1o organism checkpoint is resumed.
- Frozen `Qwen/Qwen3.5-4B`, BF16, local files only, used as both passage-visible
  teacher and passage-erased DMON-controlled language organ.
- Physical RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39` only.
- Same 8/96/64/16/8 typed-cell organism, hidden 96, 12/8 dendrites, three
  microsteps, four rank-8 anchored prefix controls, and relay-output tract gate 0.25.
- Same compact paired counterfactual schedule, fixed question permutation 101, and
  256-token passage/question limits.
- Same LR 0.001, clip norm 1, recall/sensor/effector multipliers 20/4/1.
- Same objective: task NLL 1, full-vocabulary reverse KL 1 at temperature 1, causal
  contrast 1 at margin 0.1, delta energy 0.1; binding/output-code/eligibility terms 0.
- Same detached no-exposure reference; only the exposed-minus-reference delta enters
  the anchored prefix.

The Qwen hybrid linear/full-attention implementation may use its installed PyTorch
fallback kernels. Kernel installation is a throughput optimization, not a treatment
change.

## Gates

### One-update preflight

- Both paired passage-visible teachers select their designated temporary labels and
  have nonzero full-logit separation.
- Every organism gradient, parameter, optimizer state, continuing state, and metric is
  finite; Qwen has zero trainable parameters and zero gradient tensors.
- Initial reference-centered delta is exact zero, then the checkpoint remains finite.
- Peak allocator use leaves safe 4090 headroom and the 2070S remains untouched.

### Update-50 representation gate

- Teacher KL and absolute target NLL improve between early and late windows.
- Passage-conditioned control separation becomes materially nonzero rather than a
  shared common mode.
- Correct-passage target likelihood beats matched zero, no exposure, and wrong
  passage; all four late paired causal advantages are positive more often than not.
- Lesions and reset distinguish stored organism state from ordinary frozen-model prior.

A pass licenses a larger-organism curriculum on Qwen3.5-4B. A failure diagnoses the
reference-centered prefix/credit construction despite a valid teacher; it does not
impugn Qwen or dense teacher credit in isolation.

## Result

Status: stopped at update 50 on 2026-08-05; representation gate failed.

The run and checkpoint were finite, Qwen retained zero trainable and gradient-bearing
parameters, and the teacher was label-consistent on 99% of branches. The organism did
not learn a causal passage delta:

- effective control RMS fell from `6.43e-4` over updates 1–10 to `8.66e-7` over
  updates 41–50;
- paired control separation fell from `2.82e-5` to `4.15e-7`;
- all four late mean causal advantages were negative, with only 10–30% positive
  frequency;
- held-out normal, zero-control, and no-exposure accuracy were all 87.5%, while normal
  loss 0.5822 was worse than the zero/no-exposure floor 0.5511;
- recall gradient RMS collapsed from `1.25e-5` early to `1.87e-8` late and sensor
  gradient from `2.28e-4` to `8.83e-7`, while effector gradient remained near 0.075.

Internal and memory activity became large, but lesions did not reveal useful language
control. Absolute teacher KL and task NLL rewarded the already-capable frozen Qwen
prior, so the easiest solution was to suppress the reference-centered intervention.
C1s therefore teaches only the *change caused by passage visibility* and removes the
absolute objectives that admit this zero-control shortcut.
