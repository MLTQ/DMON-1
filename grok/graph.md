# graph.py

## Purpose
Directed dendrite connectome. Replaces traditional NCA neighborhoods with per-cell source slots — axons to specific partners. Aggregation is local attention over those real dendrites only.

## Components

### `DendriteGraph`
- **Does**: Buffer of discrete source indices + learnable signed weights; Q/K/V projections; `aggregate(h)` → messages
- **Interacts with**: `model.py` step loop
- **Rationale**: petridish insight — neurons talk over specific dendrites, not grid adjacency; attention stays *local to owned slots* (no all-pairs)

### `_wire`
- **Does**: Role-aware initial topology; prefers unique sources when pool allows
- **Rationale**: Traversable input→output path without a hand-designed stack

### `aggregate`
- **Does**: Softmax attention over K sources; signed synapse bias on logits; optional non-attention weighted sum via `use_attention=False`

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `StreamingCreature.step` | `aggregate` returns `[B,N,H]` | Signature / indexing |
| Constructor | requires `hidden` and role index tensors | Added required `hidden` arg |

## Notes
- Topology frozen during differentiable streaming. Growth/pruning is later.
- `use_attention=False` is the ablation control for "is attention earning its keep."
