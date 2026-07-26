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

### Checkpoint test
- **Does**: Proves save/load preserves the exact next update, stream cursor, live field
  state, optimizer, vocabulary, and metadata.
- **Does**: Proves a frozen-connectome control neither changes edges nor becomes
  trainable after resume.

### Evaluation and control tests
- **Does**: Exercises persistent/reset/shuffled held-out policies and verifies the GRU
  and causal-transformer controls are stateful and genuinely parameter matched.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| SOL development | All causal, metabolic, learning, resume, and control paths remain real | Removing assertions or replacing measured state with synthetic diagnostics |
