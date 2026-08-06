# L0-C1z: paired differential coherent transport

Status: preregistered after C1y and before any C1z optimizer update.

## Question

C1y aligned recall and relay tissue with exact stored answer values, but live recall
preserved only about one tenth of the available paired target difference. Absolute
similarity rewarded a large passage-common component, broadened participation, and
failed correct-vs-wrong passage control. Does making counterfactual difference the
local unit of credit preserve passage identity through the same anatomy?

## Frozen continuation

- Resume the exact C1y seed-7 update-125 checkpoint, optimizer moments, RNG, schedule,
  geometry, frozen Qwen 3.5 4B, coherent top-16 recall, reference-centered prefix,
  compact paired counterfactuals, and fresh-episode policy. Use only the RTX 4090.
- Set absolute coherent value/transport weights to zero. Add paired coherent recall-
  delta credit at weight `32` and paired differential relay/output transport credit at
  weight `32`, using the existing temperature `0.05`.
- The detached target is exactly `left stored answer value - right stored answer value`.
  Recall loss compares live branch difference with it. Transport pools select shared
  physical cell indices by cosine alignment of their live branch-state difference with
  that target delta; passage-common state therefore has zero utility.
- Preserve addressing KL `1`, passage-effect KL `1`, causal contrast `1`, LR `0.001`,
  recall/sensor/effector multipliers `20/4/1`, and all other zero-weight objectives.
- Continue for 25 updates through 150, checkpoint every update, and run development and
  held-out causal suites once at update 150. No growth or new inference structure.

## Gates

- Target deltas remain detached, nonzero, and exactly swap-invariant. Qwen remains
  frozen; every causal organism subsystem receives finite gradients.
- Recall-delta retention rises from C1y's roughly `0.10` to at least `0.50`, with
  positive delta cosine alignment and no addressing regression.
- Differential relay/output alignment becomes positive, and effective differential
  pools use fewer than 32/8 of the 64/16 available cells. Unused neurons are allowed.
- Paired control separation and student-effect separation grow materially above C1y;
  late causal satisfaction exceeds C1y's `0.417` late block and C1w's `0.542` if the
  signal reaches language usefully.
- Correct passage beats wrong passage, floor, memory lesion, and internal lesion in
  held-out mean likelihood and a majority of strict per-question comparisons.

A pass licenses retention/growth work. Preserved internal delta without behavioral
control licenses a more explicit differentiated output-organ interface. Failure to
preserve delta with live gradients licenses a fresh-weight ablation before rejecting
the paired coherent approach.
