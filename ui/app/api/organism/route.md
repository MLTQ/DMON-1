# `route.ts`

## Purpose

Exposes the loopback organism snapshot and genuine no-input clock tick to the local UI.
Unlike the generation demo fallback, this route never fabricates topology or activity.

## Components

### `GET`
- **Does**: Proxies the live bridge `/snapshot` payload containing exact cells,
  dendrites, state, and latest measured flow.

### `POST`
- **Does**: Validates a bounded tick count and temperature, then advances `/advance`
  through genuine `token=None` intervals.
- **Rationale**: The browser's run/pause control becomes the organism clock without a
  public service or background process.

### `proxy`
- **Does**: Preserves bridge status/payload and returns an explicit offline response when
  loopback is unavailable.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `app/page.tsx` | GET returns `OrganismPayload`; POST adds output characters | Payload shape |
| `sol/serve.py` | Serves `/snapshot` and `/advance` on loopback | Endpoint paths |
| Users | Offline mode contains no synthetic connectome | Adding fallback measurements |
