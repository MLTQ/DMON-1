# model.py

## Purpose
S0 creature: streaming char-level field with input ports, mirror memory, dendritic attention, continuous state.

## Components

### `CreatureState`
- **Does**: Carries `h` and mirror ring cursor; `detach` for truncated BPTT

### `StreamingCreature`
- **Does**: Embed → mirror write → recurrent dendrite microsteps → mean-pool readout
- **Interacts with**: `DendriteGraph`, `SharedGRURule`, `TrainConfig`

### Mirror contract
- Stream writes detached embeddings into rotating mirror slot
- `mutable_mask` blocks rule overwrite; dendrites may still read mirrors

### Readout
- Mean-pool over output cells + LayerNorm + linear (scales with `n_output` better than concat)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train.py` | `step(ids[B], state) → logits[B,V], state`; no hidden reset | Signature / reset semantics |
| Port buffers | indices move with `.to(device)` via register_buffer | Manual CPU-only indices |
