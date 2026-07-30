# `page.tsx`

## Purpose

Provides the interactive SOL character-organ console: exact checkpoint state,
stimulus composer, autonomous output stream, clock control, and organism telemetry.

## Components

### `Home`
- **Does**: Loads the real bridge snapshot, advances genuine no-input ticks at 4 Hz,
  appends sampled output, and freezes/resumes the organism without resetting state.
- **Interacts with**: `NetworkField`, `GET/POST /api/organism`, and
  `POST /api/generate`.
- **Rationale**: Training update, organism clock tick, and rendered animation are kept
  distinct so the UI does not imply false activity.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Users | Output continues during run and stops without state reset when frozen | Clock semantics |
| Browser tests | `network-canvas` and `response-output` test IDs remain stable | Removing test IDs |
| `network-field.tsx` | Receives only bridge-provided topology and flow | Synthetic fallback |
| Live SOL bridge | JSON follows `OrganismPayload` | API response shape |
