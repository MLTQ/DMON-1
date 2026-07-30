# `organism-types.ts`

## Purpose

Defines the local checkpoint payload consumed by the character console. The types mirror
real bridge state and deliberately contain no synthetic topology fallback.

## Components

### `OrganismPayload`
- **Does**: Carries checkpoint identity, organism clock, telemetry, exact topology, and
  optional output characters.
- **Interacts with**: `/api/organism`, `NetworkField`, and `Home`.

### `OrganismTopology`
- **Does**: Names every target-owned dendrite and its latest measured flow, slow/fast
  weight, cell activity, energy, viability, and biological role.

### `EMPTY_METRICS`
- **Does**: Represents an unavailable bridge without fabricating organism measurements.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `network-field.tsx` | Matrix rows align by target cell and dendrite slot | Shape or indexing convention |
| `page.tsx` | Clock ticks are live bridge ticks, not training updates | Clock semantics |
| `sol/serve.py` | JSON keys use camel-case UI names | Payload field names |
