# cell.py

## Purpose
Shared local update rule. One parameter set applied at every mutable cell; private hidden state lives outside this module.

## Components

### `SharedGRURule`
- **Does**: GRUCell over concat(message, drive)
- **Interacts with**: `model.py`
- **Rationale**: GRU is petridish's preserved baseline; gated state retention is required for any cross-token integration

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `StreamingCreature` | `forward(h, messages, drive) → h'` all `[B,N,H]` | Rank or channel mismatch |

## Notes
- LSTM / temporal-transformer cell variants can mirror petridish `sequence_cells` later as controlled ablations.
