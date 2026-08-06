# L0-C1x: coherent juvenile maturation to update 100

Status: preregistered after C1w update 25 and before resuming its optimizer.

## Question

C1w learned sparse addressing, recruited every causal organism subsystem after the
effector opened, and improved causal-advantage satisfaction from 0.354 early to 0.542
late, but still expressed mostly harmful held-out controls. Was update 25 merely too
early—two passes through a 12-example developmental schedule—for value transport and
language control to mature?

## Frozen continuation

- Resume the exact C1w seed-7 update-25 checkpoint, optimizer moments, RNG, schedule
  position, evaluation history, and fresh-episode lane policy. Do not restart weights.
- Change only the terminal update from 25 to 100. Preserve the 736-cell
  16/384/256/64/16 geometry, hidden 128, three microsteps, dendrites 16/12, eight
  rank-16 controls, coherent residual gain 0.1, top-k 16, and recency bias 0.08.
- Preserve frozen Qwen 3.5 4B, reference-centered prefix control, compact paired
  counterfactuals, LR/clip and group multipliers, and every C1w objective weight.
- Checkpoint every update and evaluate development at updates 50, 75, and 100. Run the
  held-out causal suite once at update 100. Use only the physical RTX 4090.
- Do not add answer-span targets, sharper attention, utility protection, allocation,
  or growth during this continuation. Those are separate treatments if maturation
  fails.

## Gates

- Qwen remains zero-trainable/zero-gradient and all optimizer, state, loss, and
  gradient telemetry remains finite.
- Addressing KL remains below the C1w early mean `0.1318`; selected teacher mass does
  not regress below `0.50`; query/key gradients remain nonzero.
- Connectome, tissue, transport, recall, sensor, and effector groups remain recruited.
- Late causal satisfaction exceeds C1w's `0.5417` and paired control/logit separation
  grows rather than collapsing.
- Direct recall/value alignment or separation must improve; otherwise time is not
  repairing value semantics even if the address stays good.
- At update 100, correct passage must beat no exposure, reset, wrong passage, memory
  lesion, and internal lesion in mean target likelihood and in a majority of strict
  per-question comparisons. Accuracy alone cannot pass.

A pass licenses continuous-lifetime training followed by utility protection and
demand-driven growth. An addressing-only pass licenses exact answer-span/coherent-value
credit. Failure with gradient collapse licenses lower LR or staged optimizer treatment;
failure with live gradients rejects training duration as the missing ingredient.

## Result

C1x completed update 100 with Qwen at zero trainable parameters and zero gradient
tensors. Peak allocation/reservation was 21.65/21.84 GiB. The raw artifact is
`l0c1x-coherent-juvenile-s7-u100-result.json`, SHA-256
`0e9ba2aac473f73478da464ef1374734a4e772303ca28e62a74142b47cf39b82`.

Duration improved the already successful address while collapsing expressed control.
Across the four 25-update blocks:

- addressing KL fell `0.1083 -> 0.0616 -> 0.0444 -> 0.0337`, effective slots sharpened
  `14.25 -> 13.57 -> 12.98 -> 12.61`, and selected teacher mass remained above `0.52`;
- direct-recall alignment was `0.0170 -> 0.00635 -> -0.0116 -> 0.0264`, while recall
  separation monotonically fell `0.00565 -> 0.00478 -> 0.00375 -> 0.00352`;
- causal satisfaction fell `0.460 -> 0.310 -> 0.340 -> 0.280` rather than exceeding
  C1w's late `0.542`;
- control RMS fell `0.01147 -> 0.00561 -> 0.000764 -> 0.000423`, student-effect RMS
  `0.474 -> 0.238 -> 0.0328 -> 0.0321`, and label separation `0.0978 -> 0.0793 ->
  0.0506 -> 0.0409`;
- connectome/tissue/transport gradients remained finite but fell by roughly 478x/763x/
  296x from the first to final block. Query/key remained healthier at `3.55e-5`/
  `1.08e-4`, while value/output fell to `3.64e-7`/`4.13e-7`.

Development normal loss moved from `0.4312` at update 25 to `0.2864`, `0.2876`, and
`0.2922` at updates 50/75/100, chiefly by silencing the harmful intervention. The
update-100 floor was `0.2722`, wrong passage `0.2798`, memory lesion `0.2766`, and
internal lesion `0.2380`; correct passage therefore lost every mean causal comparison.

Held out, normal reached 87.5%/`0.56555`, slightly better in mean loss than the frozen
floor at 75%/`0.57526` and memory lesion at 87.5%/`0.56702`. It remained worse than
wrong passage at 87.5%/`0.55945` and internal lesion at 87.5%/`0.56445`. Strictly,
normal beat the floor and memory lesion on only 2/8 questions, wrong passage on 0/8,
and internal lesion on 4/8. This is a small near-neutral common effect, not correct-
passage-specific memory.

C1x rejects training duration as the missing ingredient. The organism learned where
to look but not a stable value representation, then minimized damage by becoming
silent. The next treatment must identify the designated answer span during training,
define the recall target in the sensor/memory coordinate itself rather than a random
projection, and locally require sparse relay tissue to transport that live value. More
time, capacity, utility protection, and growth are deferred until that value route can
beat wrong-passage and lesion controls.
