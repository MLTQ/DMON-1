# L0-C1o: anchored pre-transformer creature prefix

Status: complete 2026-08-05. The numerical/interface gate passed; natural memory did
not. This result does not reject pre-transformer control.

## Question

Does pre-transformer creature control have a finite, learnable BF16 backward path when
continuous controls perturb ordinary token-like prefix embeddings rather than entering
Llama as pathological all-zero virtual tokens?

## Frozen treatment

Retain C1n's two-pass perceive-then-express interface, fresh seed-7 organism, causal
controls, geometry, optimizer, compact paired bindings, and 4090 UUID. For each prefix
slot, repeat the detached embedding of the question's first ordinary token and add one
bounded SOL2 control vector. The matched zero-control arm uses the identical anchors
with an exact-zero creature residual. Passage text remains absent from the second Llama
workspace.

Before clipping or stepping, reject the update if any organism gradient tensor or the
aggregate gradient norm is non-finite. The rejected update must not change parameters,
optimizer moments, continuing state, or checkpoint cursor.

All other pilot constants remain those frozen for C1n: 8/96/64/16/8 typed cells,
hidden 96, 12/8 dendrites, three microsteps, tract gate 0.25, four rank-8 controls,
control/recall gain 1, base LR 0.002, recall/sensor/effector multipliers 20/4/1,
causal-only weight 4 and margin 0.1, and 256-token limits.

## Gates

1. CPU tests must prove anchor-plus-residual construction, deterministic matched zero
   control, gradients into controls through frozen layers, and pre-step non-finite
   rejection without mutation.
2. Repeat one 8B BF16 update on only physical RTX 4090 UUID
   `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. Require finite gradients, parameters,
   and states; Llama 0 trainable / 0 gradient tensors; and safe peak memory.
3. Only a finite pass may continue fresh seed 7 to update 25. Require nonzero
   passage-conditioned control and label-logit separation and positive causal
   advantages over both no exposure and the incompatible passage.
4. Natural memory still requires normal to beat wrong/no/reset/lesion controls; a
   compact-binding training effect alone is interface evidence, not memory evidence.

## Implementation and preflight

`language-control-mode=prefix` now performs a two-pass turn. A frozen first Llama pass
provides detached contextual features to SOL2. SOL2 advances over the question and its
output tissue emits four bounded rank-8 controls. A second frozen Llama pass sees the
exact question IDs preceded by four continuous anchor-plus-residual prefix vectors.
Logits returned to the task exclude the virtual prefix positions. The Llama parameters
remain frozen while gradients traverse its layers into the organism.

The exact-zero C1n prefix was numerically pathological in BF16: its forward pass was
finite, but the first organism backward produced a non-finite gradient norm. C1o's
ordinary-token anchor removed that failure. The one-update 8B BF16 preflight passed on
only the physical RTX 4090:

- peak allocated/reserved memory was 19,152,308,736 / 19,274,924,032 bytes;
- the organism gradient norm was finite at 26.755 before clipping;
- Llama had zero trainable parameters and zero gradient tensors;
- every saved organism and optimizer tensor was finite; and
- the second update sent nonzero gradients into sensor, identity, connectome, tissue,
  effector, transport, and recall parameter groups. The first update touched only the
  zero-initialized effector path, as expected from the decoder's zero-up construction.

CPU gates also cover anchor construction, matched zero controls, gradient transport
through frozen Llama layers, and rejection of non-finite gradients before any optimizer
mutation.

## Update-25 result

The fresh seed-7 arm continued from its finite two-update preflight to update 25. It
used 845,066 trainable organism parameters and zero trainable or gradient-bearing Llama
parameters. Peak allocated/reserved CUDA memory was 19,571,956,224 / 19,671,285,760
bytes. All 239 tensor leaves in the update-25 checkpoint, including optimizer state,
were finite. The 4090 returned to idle after evaluation; the 2070S was not used.

The late update-16--25 window had mean causal advantages, in the frozen order
`left>none`, `right>none`, `left>wrong`, `right>wrong`, of:

```text
+0.00694, +0.00198, +0.01121, +0.01379
```

Their positive-update fractions were only `0.50, 0.40, 0.50, 0.60`; mean causal
accuracy was 0.50 and mean causal loss was 0.09152 against the 0.1 margin. Mean
passage-conditioned control separation was nonzero but extremely small at
`8.31e-7` RMS, while mean label-logit separation was `0.05264` RMS. Thus the narrow
continuous representation gate passed in sign, but it was weak and unstable rather
than a robust learned contrast.

Natural held-out behavior failed the memory gate:

| intervention | accuracy | mean loss | control RMS |
| --- | ---: | ---: | ---: |
| normal | 0.250 | 1.5571 | 0.08797 |
| no exposure | 0.250 | 1.5088 | 0.08549 |
| matched zero control | 0.625 | 1.0520 | 0 |
| reset after exposure | 0.250 | 1.5088 | 0.08549 |
| wrong passage | 0.250 | 1.5562 | 0.08797 |
| memory lesion | 0.250 | 1.5161 | 0.08549 |
| internal lesion | 0.250 | 1.4302 | 0.07079 |
| question paraphrase | 0.250 | 1.4937 | 0.08797 |

Development normal and wrong-passage arms were likewise identical at 0/4 and loss
1.6348; matched zero control scored 3/4 at loss 0.3697. Small samples make the discrete
accuracies noisy, but the normal/wrong equality and the much better zero-control arm
are directionally unambiguous.

## Interpretation

The important positive result is mechanistic: a living SOL2 state can produce bounded
controls *before* the frozen transformer stack, receive finite credit through all Llama
layers, and recruit the internal organism. This is a substantially cleaner interface
than treating the creature as a late logit bias.

The current objective, however, mostly grew a question-conditioned common-mode control.
Control magnitude rose to about 0.088 RMS while incompatible passages differed by only
about one part in 100,000. That shared push degraded the useful frozen-language floor;
the loss supplied no absolute task term or energy price to prevent it. The small
late-window contrast is therefore interface evidence, not evidence that wiki content is
stored and causally decoded.

The next before-mode experiment should keep this interface and change the credit
geometry before spending on a long run:

1. Decompose each output organ's prefix into a no-op/common component and an explicitly
   memory-gated delta, initialized at exact zero.
2. Penalize control energy or movement from the matched zero-control floor, so a generic
   prefix is costly and only answer-improving intervention survives.
3. Combine absolute target likelihood with the paired causal contrast. This prevents a
   relative-margin solution that harms both exposed branches.
4. Measure the delta directly against no-exposure and wrong-passage states, and require
   the normal arm to beat matched zero control before extending the run.

Only after this before-mode control learns stable content-conditioned deltas should the
same organism controls be offered at several depths. That later experiment can test
whether layer-specific organs add iterative reasoning bandwidth rather than compensating
for a faulty memory-to-control objective.
