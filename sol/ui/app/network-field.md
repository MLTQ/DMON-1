# `network-field.tsx`

## Purpose

Renders the checkpoint's actual sparse directed connectome. Positions are an abstract
readable layout; node identity, roles, target-owned dendrites, signed weights, activity,
energy, and flow all come from the live organism.

## Components

### `NetworkField`
- **Does**: Draws every checkpoint cell and dendrite on canvas and supports cell
  selection.
- **Does**: Maps latest measured per-edge message coefficient magnitude to edge
  brightness/width and maps retained stimulation to cell brightness.
- **Rationale**: Avoids decorative packets that imply measurements the model did not
  report.

### `makeNodes`
- **Does**: Places real sensory cells at left, output cells at right, and interior cells
  on a deterministic abstract grid.
- **Rationale**: SOL has learned connectivity but no physical coordinate state yet.

### `makeEdges`
- **Does**: Preserves the model's `(target, dendrite slot) -> source` ownership exactly,
  including self-edges.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `page.tsx` | Selected-cell callback receives the real checkpoint cell id | Selection semantics |
| `organism-types.ts` | Flow/weight matrices align with `sources[target][slot]` | Matrix indexing |
| Users | Motion is never fabricated; clock updates replace measured snapshots | Synthetic activity |
