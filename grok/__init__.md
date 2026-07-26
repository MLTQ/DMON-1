# __init__.py

## Purpose
Package entry for the grok/ S0 streaming substrate. Exports the public surface used by trainers and experiments.

## Components

### `TrainConfig`, `StreamingCreature`
- **Does**: Re-export core types
- **Interacts with**: `config.py`, `model.py`

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CLI / notebooks | `from grok import StreamingCreature` works | Rename or move exports |
