# `checkpoint.py`

## Purpose

Persists the complete continuously learning process so resume does not silently create
a new organism or a new stream.

## Components

- `pack_state` / `unpack_state` — round-trip organism and baseline state.
- `same_tensor_values` — compares invariant tensor values after normalizing away
  `map_location` device changes.
- `restore_rng_state` — restores CPU and CUDA byte-state tensors on the devices expected
  by PyTorch's RNG APIs.
- `save_checkpoint` — atomically writes model, optimizer, state, stream cursor,
  configuration, counters, histories, and RNG state.
- `load_checkpoint` — schema-checked, device-aware load.

## Contracts

- Existing recurrent state and sensory-memory cursor survive resume.
- Stream positions resume at the exact next character.
- A temporary file is replaced only after serialization succeeds.
- Schema changes require an explicit migration.
- CUDA resume remains valid when `torch.load(map_location=...)` remaps every serialized
  tensor, including tensors that PyTorch requires on CPU during restoration.
