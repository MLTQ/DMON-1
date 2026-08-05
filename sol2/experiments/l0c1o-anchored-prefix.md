# L0-C1o: anchored pre-transformer creature prefix

Status: protocol frozen 2026-08-05 before implementation tests or optimizer updates.

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
