# `test_sol.py`

## Purpose

Protects the scientific contracts of the first SOL prototype rather than only checking
tensor shapes.

## Components

### Topology tests
- **Does**: Proves the graph is sparse, directed, and connects sensory cells to outputs.

### Stream and persistence tests
- **Does**: Proves optimizer windows are adjacent and prior characters affect later
  predictions.

### Metabolism and eligibility tests
- **Does**: Proves unstimulated energy depletion and delayed reward dependence on a
  remembered event trace.

### Credit-path test
- **Does**: Proves exact loss gradients reach retained cell states and directed synapses.

### Learning smoke test
- **Does**: Requires substantial loss reduction on a deterministic character stream
  without resetting organism state.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| SOL development | All causal, metabolic, and learning paths remain real | Removing assertions or replacing measured state with synthetic diagnostics |
