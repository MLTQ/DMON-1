# `serve.py`

## Purpose

Runs a local-only HTTP bridge between one live SOL checkpoint and the existing console.
It binds to `127.0.0.1` by default and has no hosting or external-service dependency.

## Components

### `LiveOrganism`
- **Does**: Owns one persistent checkpoint lane across requests and serializes access.
- **Interacts with**: `load_organism` in `checkpoint.py`.

### `LiveOrganism.generate`
- **Does**: Streams the human prompt through the ordinary field, backpropagates prompt
  next-character loss without updating weights, reports measured cell/edge credit, then
  samples through the same output path.
- **Rationale**: UI credit telemetry comes from a real reverse signal rather than a
  fabricated animation metric.

### `LiveOrganism.snapshot`
- **Does**: Returns energy, stimulation, eligibility, directed sources, and signed edge
  weights for future live visualization.

### HTTP handler
- **Does**: Serves `GET /health`, `GET /snapshot`, and `POST /generate`.
- **Rationale**: The Python standard library is sufficient for a single-user local bridge.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/ui/app/api/generate/route.ts` | `/generate` returns `output`, `mode`, and legacy metric keys | Response shape |
| Local operator | Default host is loopback-only | Changing bind default |
| UI telemetry | Cell and edge credit are measured prompt gradients | Replacing backward pass with proxy values |
