# `model.py`

## Purpose

Assembles typed tissues, the directed connectome, stream-written sensory memory,
private cell identity, and the fixed text-output organ into one persistent organism.

## Components

### `StepHealth`

Records hidden/message/logit scale, tissue update rates, and the largest effective
operator norm beside behavioral loss.

### `Sol2`

- Builds explicit input, memory, compute, relay, and output index buffers.
- Writes bounded token embeddings into a detached FIFO sensory-memory tissue.
- Applies one rule per mutable tissue type.
- Optionally gives every cell bounded private gain/bias expression.
- Reads a fixed output organ through a concat head; internal growth never resizes it.
- Supports zero-and-freeze tissue interventions without changing the training path.

### `count_parameters`

Counts only trainable parameters for matched controls.

## Decisions

- Typed rules are the SOL2 substrate, not an experimental toggle. The prior universal
  shared-rule constraint was accidental.
- Cell identity is bounded and zero-initialized, so its enabled arm starts behaviorally
  identical to its disabled counterpart when seeded identically.
- Token embeddings are bounded before entering persistent state in every arm.
- The output organ is fixed and position-specific; growth extends internal tissue only.
- Relay tissue is physically last in the index layout, so relay growth is append-only
  and reconstructible from configuration plus checkpoint buffers.

## Contracts

- Mutable state remains in `[-1,1]` from a zero initial state.
- Only the external stream writes memory cells.
- Output cells are sinks in the connectome.
- `weight_version` is preserved through ticks.
- All tissue access uses index buffers; growth may append cells after the original
  output block without relying on contiguity.
