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
