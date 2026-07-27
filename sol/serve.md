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
  structural rewrite count, then samples through the same output path. Scalar reverse
  credit and channel-shaped output-error credit are exposed separately.
- **Does**: Adds viability, quiescence, external energy input, metabolic spending, and
  transport drift to local-only checkpoint telemetry.
- **Rationale**: UI credit telemetry comes from a real reverse signal rather than a
  fabricated animation metric.

### `LiveOrganism.advance_silence`
- **Does**: Advances one or more genuine `token=None` intervals, samples the output
  organ without feeding a sensory token, and retains the resulting state.
- **Does**: Reports zero novelty/input energy plus the real per-dendrite message flow
  produced by each silent tick.
- **Rationale**: The local run/pause clock must exercise the organism rather than animate
  a static checkpoint.

### `LiveOrganism.snapshot`
- **Does**: Returns exact cell/dendrite counts, energy, stimulation, cell/edge
  eligibility, directed sources, slow signed weights, per-stream fast weights, latest
  measured edge/probe flow, per-cell activity/energy/viability, role indices, clock
  state, structural credit, backward-credit magnitude, and rewrite count.
- **Does**: Exposes the actual mean magnitude of checkpointed output-error credit;
  unstimulated generation can only transport and decay existing credit because it has
  no observed target from which to create a new decoder correction.
- **Does**: Derives current viability and quiescent fraction from the checkpoint lane's
  actual cell energies without mutating it.
- **Does**: Returns the exact active-slot mask so dormant capacity is visually and
  semantically distinct from a live dendrite.

### HTTP handler
- **Does**: Serves `GET /health`, `GET /snapshot`, `POST /generate`, and
  `POST /advance`.
- **Rationale**: The Python standard library is sufficient for a single-user local bridge.

### CLI
- **Does**: Loads `sol/runs/live.pt` by default or an explicit `--checkpoint`.
- **Interacts with**: `promote.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/ui/app/api/generate/route.ts` | `/generate` returns output plus the complete live snapshot | Response shape |
| `sol/ui/app/api/organism/route.ts` | `/snapshot` and `/advance` expose exact topology and no-input ticks | Endpoint or payload shape |
| Local operator | Default host is loopback-only | Changing bind default |
| `promote.py` | Default checkpoint path is `sol/runs/live.pt` | Path mismatch |
| UI telemetry | Credit, activity, and flow are measured; geometric positions alone are illustrative | Replacing model values with animation proxies |
