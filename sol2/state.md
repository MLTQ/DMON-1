# `state.py`

## Purpose

Defines the minimum persistent electrical state of a SOL2 organism independently from
the model and trainer.

## Components

### `OrganismState`

- `hidden`: per-lane, per-cell persistent state.
- `memory_cursor`: next stream-written sensory-memory slot.
- `weight_version`: records which parameter version most recently advanced the state.
- `detach()` preserves experience while truncating autograd history.
- `clone_detached()` creates an immutable learning snapshot for asynchronous work.

## Contracts

- Detaching never resets hidden state or the memory cursor.
- Growth migrators append cells without changing existing state values.
- The version is telemetry; changing it must not change model behavior.
