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
  next-character loss without updating weights, reports measured cell/edge credit and
  fast synaptic efficacy, causal-probe traffic, persistent backward credit, and
  structural rewrite count, then samples through the same output path.
- **Does**: Adds viability, quiescence, external energy input, metabolic spending, and
  transport drift to local-only checkpoint telemetry.
- **Rationale**: UI credit telemetry comes from a real reverse signal rather than a
  fabricated animation metric.

### `LiveOrganism.snapshot`
- **Does**: Returns energy, stimulation, cell/edge eligibility, directed sources, slow
  signed edge weights, per-stream fast weights, candidate sources, retained structural
  credit, backward-credit magnitude, rewrite count, and reward surprise baseline for
  live visualization.
- **Does**: Derives current viability and quiescent fraction from the checkpoint lane's
  actual cell energies without mutating it.

### HTTP handler
- **Does**: Serves `GET /health`, `GET /snapshot`, and `POST /generate`.
- **Rationale**: The Python standard library is sufficient for a single-user local bridge.

### CLI
- **Does**: Loads `sol/runs/live.pt` by default or an explicit `--checkpoint`.
- **Interacts with**: `promote.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/ui/app/api/generate/route.ts` | `/generate` returns `output`, `mode`, legacy metric keys, and additive `fastWeight` | Response shape |
| Local operator | Default host is loopback-only | Changing bind default |
| `promote.py` | Default checkpoint path is `sol/runs/live.pt` | Path mismatch |
| UI telemetry | Cell and edge credit are measured prompt gradients | Replacing backward pass with proxy values |
