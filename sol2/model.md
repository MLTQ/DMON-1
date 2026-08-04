# `model.py`

## Purpose

Assembles typed tissues, the directed connectome, stream-written sensory memory,
private cell identity, and the attentive text-output organ into one persistent organism.
The legacy A organ remains checkpoint-compatible while additional sensor/effector
bundles can be explicitly attached, detached, and selected.

## Components

### `StepHealth`

Records hidden/message/logit scale, tissue update rates, organ-attention allocation,
and the largest effective operator norm beside behavioral loss.

### `Sol2`

- Builds explicit input, memory, compute, relay, and output index buffers.
- Writes bounded token embeddings into a detached FIFO sensory-memory tissue.
- Applies one rule per mutable tissue type.
- Optionally gives every mutable cell bounded private target/time-constant expression.
- Optionally gives every mutable cell a private low-rank transition residual over its
  state, incoming message, and drive.
- Clears output tissue at each token boundary, preventing it from becoming an
  autonomous recurrent shortcut.
- Reads dedicated output tissue through learned organ queries; the organ never sees
  input, memory, compute, or relay tensors directly.
- Retains legacy A parameter names and stores additional physical ports in an initially
  empty `attached_organs` module dictionary.
- Selects one organ explicitly per step and exposes exact attach/detach/reattach and
  organ-parameter iteration operations.
- Supports zero-and-freeze tissue interventions without changing the training path.

### `count_parameters`

Counts only trainable parameters for matched controls.

## Decisions

- Typed rules are the SOL2 substrate, not an experimental toggle. The prior universal
  shared-rule constraint was accidental.
- Cell identity is bounded and zero-initialized, so its enabled arm starts behaviorally
  identical to its disabled counterpart when seeded identically.
- Private adapters use a seed-derived private random down projection and exactly zero
  up projection, making their enabled initialization behaviorally identical to rank
  zero while allowing gradients to recruit cell-local computation. The private RNG
  also leaves all matched downstream organ initialization unchanged.
- Token embeddings are bounded before entering persistent state in every arm.
- The output organ has fixed query slots but is not tied to concatenated cell positions.
  Growth extends internal tissue only in this first kernel.
- An empty attached-organ registry adds no state-dict keys, so pre-attachment
  acquisition checkpoints continue to load strictly.
- Relay tissue is physically last in the index layout, so relay growth is append-only
  and reconstructible from configuration plus checkpoint buffers.
- Private-expression rows map only to mutable cells. Stream-written memory contributes
  no dead parameters to the identity treatment or matched-control budget.
- Adapter rows use the same mutable-cell mapping and cannot create a memory or decoder
  bypass; their bounded residual enters only the typed tissue target.

## Contracts

- Mutable state remains in `[-1,1]` from a zero initial state.
- Only the external stream writes memory cells.
- Output cells are sinks in the connectome.
- Output cells receive only from compute/relay tissue and retain state only within a
  token's microsteps.
- `weight_version` is preserved through ticks.
- All tissue access uses index buffers; growth may append cells after the original
  output block without relying on contiguity.
- Attaching an organ preserves global RNG state and cannot alter A behavior until that
  organ is explicitly selected.
