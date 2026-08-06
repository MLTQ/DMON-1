# backbone.py

## Purpose

Frozen causal LMs that accept bounded control residuals at several preregistered
decoder depths — the C1aa mandate's "several bounded, output-tissue-only injection
sites", implemented so that the no-control floor is the bare model, bitwise.

## Components

- `resolve_depths(fractions, n_layers)` — fractions in (0,1] → distinct post-block
  layer indices; collisions raise.
- `inject_control_residual(hidden, controls)` — additive cross-attention residual:
  softmax(h·cᵀ)·c added to the hidden stream. Convex combination ⇒ componentwise
  bound equals the control bound. fp32 softmax under reduced-precision backbones.
- `MultiDepthBackbone(HuggingFaceFrozenBackbone)` — real HF decoder; injection via
  forward hooks on `model.model.layers[d]`; `bare_logits()` is the true floor
  (no hooks, no graph); inherits `encode` (no-grad final hidden) and
  `native_logit_error` from the sol2 adapter.
- `ToyMultiDepthBackbone` — deterministic frozen GRU-residual stack with the same
  injection surface, for CPU contracts. Not a performance baseline.

## Decisions

- **Scaffold-free by construction.** The C1n continuous prefix's "no-control floor"
  still carried prepended embeddings, so floor and bare model were different
  computations. Here zero controls short-circuit to the untouched hidden stream —
  but only when the controls carry no gradient graph. A live graph never
  short-circuits: the zero-initialized graft recruits through exactly that gradient
  path, and severing it would freeze the organ at birth (caught in review before
  first run).
- **Hooks, not model surgery.** The backbone stays a stock frozen checkpoint;
  handles are registered per call and removed in `finally`, so a raised exception
  cannot leak an injection site into later calls.

## Contracts

- Zero/absent controls ⇒ logits equal to `bare_logits` (contract-tested on the live
  gradient path, where the short-circuit cannot fire).
- Residual componentwise magnitude ≤ max |control| (contract-tested).
- `bare_logits` and `encode` never build a training graph; injected forwards leave
  no gradient tensors on backbone parameters.
