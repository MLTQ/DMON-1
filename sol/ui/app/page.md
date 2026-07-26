# `page.tsx`

## Purpose

Provides the interactive SOL character-organ console: a directed-field visualization,
prompt composer, streamed response, and organism telemetry.

## Components

### `NetworkField`
- **Does**: Renders cells, directed axons, measured-role colors, traffic pulses, and
  selectable cells on an HTML canvas.
- **Rationale**: Canvas keeps the network a functional visualization rather than a
  decorative illustration.

### `Home`
- **Does**: Owns prompt, response streaming, selected-cell state, and live telemetry.
- **Interacts with**: `POST /api/generate` in `api/generate/route.ts`.
- **Rationale**: The endpoint is replaceable by the local PyTorch bridge without changing
  the interface.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Users | Prompt submission streams a visible response and network activity | Control labels and response semantics |
| Browser tests | `network-canvas` and `response-output` test IDs remain stable | Removing test IDs |
| Live SOL bridge | JSON response contains `output`, `mode`, and `metrics` | API response shape |
