# organ.py

## Purpose

The Broca organ: one sensor from frozen language features into the organism, one
attentive trunk over dedicated output tissue, and K zero-initialized low-rank heads
that translate the shared trunk summary into one bounded control bank per injection
depth. This is the "differentiated language organ" the 2026-08-06 program review
promotes — representational transformation is learned, nothing is copied
coordinate-for-coordinate.

## Components

- `sense(features)` — `EffectiveLinear(language_width → hidden)` + per-input-cell
  bounded gain/bias; identical pattern to `sol2.language_organs`.
- `read(output_cells)` — trunk (`AttentiveOutputOrgan`) → `tanh` → per-depth linear
  heads → per-depth orthogonal bases → `control_gain · tanh(·)`. Returns
  `[batch, n_depths, tokens, language_width]`.
- `operator_norms()` — sensor, trunk, per-head, per-basis spectral norms.

## Decisions

- **Zero-init heads, live trunk.** A fresh graft is exactly silent; the first
  optimizer step recruits the heads, after which trunk/sensor/organism gradients
  open up (two-stage recruitment is contract- and audit-gated).
- **Boundedness lives at exactly one place — the control level.** An earlier draft
  also bounded the coefficient stage with its own tanh over the *unbounded* trunk
  summary (entries observed at ±6.6). Under moderate head growth that stage
  saturated bit-exact: arm differentials fell below fp32 resolution while the
  static control kept inflating — the C1x collapse-to-silence and C1y common-mode
  failures reproduced in one mechanism on the toy curriculum. The summary is now
  bounded once (`tanh`) before linear heads, and the only saturating nonlinearity
  is the final control bound.
- **Per-depth heads, shared trunk.** Heads are independently lesionable (per-depth
  attribution) and independently zero-initialized; the trunk gives them a common
  read of output tissue without multiplying parameters by K.

## Contracts

- Fresh organ ⇒ exactly zero controls (bitwise silent graft).
- `read` input is only ever the output-tissue slice (enforced by
  `Sol2._read_output`; boundary contract-tested).
- Every control component lies in `(-control_gain, control_gain)`.
