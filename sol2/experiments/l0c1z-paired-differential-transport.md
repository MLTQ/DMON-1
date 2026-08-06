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

## Result

C1z completed update 150 with Qwen at zero trainable parameters and zero gradient
tensors. Peak allocation/reservation was 21.94/22.45 GiB. The raw artifact is
`l0c1z-paired-delta-s7-u150-result.json`, SHA-256
`ff950efc7e6bba9d232faf5a32d7937944a56178e01cadb00c47bb07703095b3`.

Differential credit was mechanically valid but too weak and inconsistent to pass its
retention gate. Across updates 126-137 versus 139-150:

- addressing stayed healthy at KL `0.0285 -> 0.0308` and selected mass
  `0.534 -> 0.539`;
- recall-delta cosine improved `0.183 -> 0.216`, but live delta RMS remained
  `0.00316 -> 0.00304` while schedule target RMS increased `0.0259 -> 0.0355`;
  retention therefore fell `0.137 -> 0.093` instead of exceeding `0.50`;
- differential relay alignment was `0.131 -> 0.116`, output alignment
  `0.024 -> 0.038`, and effective pools `25.6 -> 30.6` relay and `10.0 -> 12.8`
  output cells. Individual examples sometimes concentrated to 6-19 relay cells, but
  shared weights did not stabilize one sparse route across the schedule;
- every causal subsystem retained finite nonzero gradients. Late control separation
  rose 2.7x from `1.71e-5` to `4.55e-5`, label separation `0.0330 -> 0.0592`, and
  causal satisfaction `0.146 -> 0.458`, but all regressed in the final five updates.

The behavioral direction improved without reaching useful control. Development normal
loss was `0.28955`, better than wrong passage `0.29224` but worse than floor `0.27222`,
memory lesion `0.28247`, and internal lesion `0.28760`. Held out, normal was
75%/`0.57495`: marginally better than floor `0.57526` and meaningfully better than
wrong passage `0.58014` and internal lesion `0.59705`, but worse than memory lesion
`0.56726`. Strict normal wins were 2/8 versus floor, 1/8 versus wrong passage, 1/8
versus memory lesion, and 3/8 versus internal lesion (with 1/4/2/1 exact ties).

C1z fails the preregistered retention and held-out gates, but unlike C1y it establishes
the correct-vs-wrong passage direction in both development and held-out mean loss.
The raw exact delta varies by question and contributes a much smaller gradient scale
than the frozen-language effect objective. The next clean treatment should normalize
the exact coherent delta direction to a fixed developmental RMS and form the local
differential route in a short stage before reintroducing language-effect and causal
credit. More capacity would not address the measured scale/common-route failure.
