# L0-C1n: pre-transformer creature prefix

Status: protocol frozen 2026-08-05 before implementation tests or optimizer updates.

## Question

Can a frozen Llama reason over passage-dependent creature state when SOL2 emits latent
tokens before the transformer, rather than adding a residual after all transformer
layers have already run?

## Interface treatment

Preserve the existing perception pass: frozen Llama converts each exposure and
question token into detached final contextual features, and SOL2 evolves over those
features. After the question has been perceived, the dedicated output cells emit the
same bounded low-rank control-token bank used by the late-residual graft.

In the treatment arm, rerun the erased-context question through frozen Llama with the
creature controls prepended as continuous input embeddings. Every transformer layer
may therefore attend to the creature state before the ordinary frozen vocabulary head
produces the answer. A matched zero-control arm uses the same number of exact-zero
prefix embeddings, separating creature information from the structural null-prefix
effect. No passage tokens enter this second Llama workspace.

The initial implementation is a two-pass turn-level organ, not token-level biological
timing: first perceive the complete user turn, then express a response conditioned on
the resulting organism state. Generated language can return as later sensory input.

## Frozen boundaries

- Llama weights remain frozen and excluded from checkpoints and the optimizer.
- Gradients may traverse the frozen transformer into prefix controls and SOL2.
- The prefix originates only from dedicated output tissue through the existing
  bounded language effector; it cannot read targets, logits, or memory directly.
- Late-residual mode remains the exact backward-compatible default.
- Development and held-out passages alter state only and never update weights.
- Normal, no-exposure, null-prefix, reset, wrong-passage, memory-lesion, and
  internal-lesion arms share the same question tokens and prefix geometry.

## Gates

1. CPU contracts must prove shapes, deterministic null prefixes, prefix gradients,
   frozen-backbone ownership, cached-perception equivalence, and unchanged late mode.
2. Run a one-update 8B BF16 preflight on only physical RTX 4090 UUID
   `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`. Require finite loss and gradients,
   Llama 0 trainable / 0 gradient tensors, and peak reserved memory below the card.
3. If viable, continue a fresh seed-7 paired causal-only arm to update 25. Require
   nonzero prefix-control and four-label separation plus nonzero causal advantages over
   no exposure and the incompatible passage.
4. Only a representation pass may extend. Memory evidence still requires normal to
   beat no-exposure, wrong-passage, reset, and relevant lesions on natural held-out
   passages. Eight-question accuracy is descriptive; causal ordering is primary.

## Interpretation

- OOM or unstable gradients: use checkpointed/segmented frozen-transformer backward
  before considering smaller scientific models.
- Prefix logits separate but passage controls do not: the language interface works;
  internal retention/routing remains the bottleneck.
- Paired compact bindings separate but natural controls do not: improve transfer from
  compact operations to ordinary text rather than scaling updates blindly.
- Natural causal controls order: replicate, then extend the same interface to
  generated conversation and only afterward test several-layer injection.
