# `test_living_language.py`

## Purpose

Provides the CPU gate for the DMON-L0 frozen-language graft before any pretrained
backbone or GPU training is attempted.

## Coverage

- A newly grafted continuous organ emits exact zeros and exactly preserves frozen
  language logits.
- Prefix mode matches its deterministic first-token-anchor plus exact-zero-residual
  floor and sends gradients through all frozen language layers into controls, never
  model parameters.
- A raw control minus its detached homeostatic reference is numerically a prefix no-op
  while retaining a live gradient into the recomputed student control; explicit zero
  scale remains exact zero under reference-centering.
- Contextual language features enter stream-written organism memory through the
  bounded sensor.
- Stream-memory writes preserve delayed gradients into the sensor, and content recall
  is bounded, masks unwritten slots, and backpropagates through valid memory only.
- Language loss first recruits the zero-up control decoder and subsequently reaches
  the sensor and deeper closed loop.
- One-pass vectorized teacher forcing exactly matches the causal repeated-prefix
  reference loss, controls, and continuing cellular state.
- Exposure-only observation, final-position scoring, and masked token loss preserve
  exact causal state evolution while avoiding unnecessary vocabulary logits.
- Cached frozen-feature entry points exactly match their token-encoding wrappers.
- Cached prefix scoring matches token-based perception when supplied the same original
  IDs required for the second pre-transformer pass.
- Query-phase write gating preserves memory cells and cursor while token and cached-
  feature scoring remain exactly equivalent.
- Frozen language parameters receive no gradients and remain byte-for-byte unchanged.
- Generation feeds prompt and generated tokens into one continuing organism state.
- Erasing the visible language context does not erase cellular state.
- A specialized language organ detaches and reattaches as the same module object.
- The generic Hugging Face adapter freezes an exposed causal model, preserves exact
  zero-control logits, passes gradients only into controls, and audits native parity.
- The graft factory is deterministic from a private seed and leaves global RNG state
  untouched.
- Explicit recall gain is preserved by deterministic graft construction without
  changing parameter initialization or RNG neutrality.
- Mixed-precision backbones and FP32 organism state cross the graft with explicit dtype
  conversion while retaining exact zero-control logits.
- A paired context-erasure task reaches perfect accuracy, while exact-zero control,
  state reset, and internal-tissue lesion remain at or below chance.

## Decisions

- The deterministic toy backbone tests interface causality only. It is not included as
  a scientific comparator for the organism's intended emergent capabilities.
- Recruitment is tested across two optimizer steps because exact-zero output adapters
  intentionally block upstream gradients on their first step.
- Greedy generation makes the state-transition count auditable and removes sampling
  noise from the contract gate.
- The small deterministic learning gate is an architectural proof: identical visible
  queries become separable only through the continuing cellular route.
- Recall is tested separately from language quality and the graph invariant verifies
  that it does not create a direct memory-to-output effector bypass.

## Run

```bash
.venv/bin/python -m sol2.test_living_language
```
