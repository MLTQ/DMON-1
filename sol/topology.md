# `topology.py`

## Purpose

Measures directed sensory-to-cell and sensory-to-output reachability on SOL's named
dendrite table. These metrics are the transport invariant for fixed and growing
connectomes.

## Components

### `TopologyMetrics`
- **Does**: Records edge/self-edge counts, reachable cells/outputs, fractions, and mean
  shortest sensory-to-output distance.

### `analyze_topology`
- **Does**: Converts target-owned source slots into forward adjacency and performs a
  multi-source breadth-first search from sensory cells.
- **Interacts with**: `SparseAxonField.sources` in `model.py` and `benchmark.py`.
- **Rationale**: A growth or pruning rule cannot claim useful morphology if it silently
  disconnects the output organ.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `benchmark.py` | Metrics are JSON-serializable and distances follow message direction | Field names or direction |
| Future axon growth | Output reachability remains one and path length is measurable | Reachability semantics |
