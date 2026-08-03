# `checkpoint.py`

## Purpose

Persists the complete continuously learning process so resume does not silently create
a new organism or a new stream.

## Components

- `pack_state` / `unpack_state` — round-trip organism and baseline state.
- `save_checkpoint` — atomically writes model, optimizer, state, stream cursor,
  configuration, counters, histories, and RNG state.
- `load_checkpoint` — schema-checked, device-aware load.

## Contracts

- Existing recurrent state and sensory-memory cursor survive resume.
- Stream positions resume at the exact next character.
- A temporary file is replaced only after serialization succeeds.
- Schema changes require an explicit migration.
