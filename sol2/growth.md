# `growth.py`

## Purpose

Implements the first SOL2 developmental operation: append transport/relay tissue
without deleting anatomy, resetting optimizer moments, or moving existing cells.

## Components

### `grow_relay_tissue`

- Appends relay cells at the end of the physical layout.
- Gives every new cell an active sensory path and zero initial state.
- Activates one weak dormant dendrite in existing tissue to read each new cell.
- Preserves all installed sources, edge parameters, private identities, and Adam moments.
- Returns a state migrator and an auditable graft ledger.

### `_replace_resized_parameter`

Replaces one row-resized parameter inside optimizer groups while preserving old state
and zero-initializing new optimizer rows.

## Decisions

- SOL2 initially grows only relay tissue. Because relay is physically last, checkpoints
  reconstruct the layout from configuration without a hidden placement manifest.
- Dormant capacity makes growth additive. Fable F1 had to donate working slots, which
  introduced an avoidable alternative cause.
- Grafts begin weak rather than claiming immediate usefulness; recruitment must be
  learned and measured later.

## Contracts

- No previously active source or slot changes.
- Every new cell is sensory-reachable and read by existing tissue.
- Output cells remain sinks.
- Existing state and optimizer rows are bitwise preserved.
- New cell state, identity, and optimizer moments begin at zero.
