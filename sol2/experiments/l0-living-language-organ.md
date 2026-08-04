# DMON-L0: a living organism with a detachable language organ

Status: architecture contract implemented 2026-08-04; production backbone and
result-bearing training protocol remain exploratory until the 4090 smoke measurements.

## Goal

Create a small persistent cellular creature that can converse through a pretrained
language model without becoming merely another fine-tuned LLM. The language model is
replaceable Broca-like machinery. SOL2 must own lifetime state, individual continuity,
development, adaptive behavior, and eventually growth.

## Graft

For each new user or generated token:

1. a frozen causal language model supplies its final contextual hidden state;
2. a bounded trainable sensor writes that state into SOL2 sensory and input tissue;
3. the continuing organism takes one cellular transition;
4. an attentive organ reads dedicated output tissue only;
5. a zero-up low-rank decoder emits continuous control tokens;
6. the frozen final hidden state attends over those controls, receives their residual,
   and passes through the frozen vocabulary head; and
7. the selected token returns as the creature's next sensory observation.

The first production injection point is immediately before the frozen language head.
It needs only one no-gradient transformer pass per token and retains exact zero-control
behavior. Multi-layer zero-residual control is a later expressivity experiment, not a
precondition for testing organism dependence.

## Non-negotiable invariants

- Every language-backbone parameter has `requires_grad=False` and remains byte-identical.
- A newly attached or explicitly zeroed control graft exactly reproduces adapter
  baseline logits within the selected dtype's numeric tolerance.
- The language organ reads no internal tensor except the output tissue supplied by
  `Sol2.step`.
- Output tissue is cleared each token; persistent information must use cellular routes.
- Clearing the LLM context never clears `OrganismState` unless the reset arm requests it.
- Generated tokens feed back through the same continuing state.
- Backbone-specific native-logit parity is measured before training; architectures
  with extra logit transforms receive specialized adapters.

## Stage L0-A: closed-loop contract

The deterministic CPU fixture must demonstrate:

- exact zero-control language parity;
- first-step recruitment of the zero-up decoder and later gradient flow into sensor
  and organism;
- no backbone gradients or parameter changes;
- state continuity across context erasure and generated-token feedback; and
- paired context-erasure accuracy of 100%, with zero-control, reset, and internal
  lesion at or below 50%.

This stage is implemented in `test_living_language.py` and is a prerequisite for every
GPU run.

## Stage L0-B: pretrained 4090 smoke

Choose a decoder-only open-weight instruct model whose unquantized or mixed-precision
inference, final hidden state, and output head fit comfortably on the physical RTX 4090.
The model choice and license are recorded in the result directory rather than hardcoded
into SOL2.

Before any training:

- measure native-logit parity, peak allocated/reserved VRAM, tokens/second, and
  backbone parameter digest;
- verify exact zero-control text and logits on fixed prompts;
- measure one forward/backward update into the organism only; and
- compare full-prefix execution with a future cached adapter to establish the throughput
  target.

No 2070S process is launched. A larger rented device is unnecessary unless production
backbone memory or sequence length—not SOL2 capacity—becomes the measured bottleneck.

## Stage L0-C: lifetime language curriculum

Training uses a few continuing lifetime lanes rather than a wide hyperparameter sweep.
Each lane receives interleaved ordinary language and paired requirements that cannot be
solved from its visible LLM context:

- names, preferences, relationships, and commitments established before context erase;
- identical visible prompts requiring organism-specific answers;
- delayed procedural instructions instantiated with new subject matter;
- self-consistency requirements involving the creature's own earlier generated text;
  and
- organ-detachment intervals where cellular state continues without language output.

The local wiki corpus supplies ordinary research prose and argument patterns, but does
not by itself establish organism dependence. Synthetic and curated paired episodes are
mixed with it so lifetime state is causally useful.

## Measurements

- ordinary held-out language loss and generation quality as the frozen-language floor;
- paired lifetime accuracy after 0, 1, and several context erasures;
- normal minus zero-control and normal minus reset loss/accuracy;
- targeted input, internal, output, memory, and newly grown tissue lesions;
- per-tissue activity, gradient participation, differentiation, attention allocation,
  control RMS, and connectome traffic;
- state survival, parameter digest, and deterministic checkpoint resume;
- organism-specific answers under identical visible prompts; and
- later, growth triggers, new-cell recruitment, and removal cost.

## Interpretation boundary

A fluent sample is not success; the frozen LLM is already fluent. Success begins when
normal operation exhibits stable individual continuity or learned procedure that the
same frozen LLM loses under zero control, reset, or relevant tissue lesion. Conversely,
a failed model, injection depth, training strength, or single seed does not disprove the
living-organism architecture.
