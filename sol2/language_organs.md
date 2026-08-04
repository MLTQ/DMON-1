# `language_organs.py`

## Purpose

Defines the detachable continuous interface between a frozen language backbone and
the persistent SOL2 organism. Language remains a replaceable organ; identity,
adaptation, and lifetime state remain in the cellular substrate.

## Components

### `ContinuousLanguageOrgan`

- Projects one frozen-backbone feature into SOL2's bounded sensory width.
- Writes that bounded feature through the ordinary stream-memory contract.
- Reads only dedicated output tissue with the existing attentive organ reader.
- Emits a small bank of continuous language-width control tokens through a shared
  low-rank basis instead of decoding vocabulary logits.
- Reports sensor, output-reader, and control-basis norms through normal health
  telemetry.

## Decisions

- The language backbone is not stored here and is never trained by this module.
- Controls are continuous so the organism can modulate an LLM's representative
  substrate without learning a second vocabulary decoder.
- The coefficient decoder is exactly zero-initialized. Attaching an untrained organ
  is therefore a behavioral no-op, and a zero-control causal comparison is exact.
- The low-rank basis is shared across control-token positions. Each position gets its
  own coefficients, preserving distinct signals without a dense
  `output_tissue -> control_tokens * language_width` matrix.
- The sensor uses the same bounded drive and per-input-cell gain/bias convention as
  discrete organs, keeping internal anatomy causally relevant.

## Contracts

- Sensory input has shape `[batch, language_width]`.
- Effector output has shape
  `[batch, n_control_tokens, language_width]` and lies in
  `[-control_gain, control_gain]`.
- At initialization, every control value is exactly zero.
- The organ cannot read input, memory, compute, or relay tissue directly.
- `control_rank` is positive and no larger than `language_width`.

