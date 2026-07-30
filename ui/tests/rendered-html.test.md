# `rendered-html.test.mjs`

## Purpose

Protects the local SOL console and its checkpoint-proxy fallback at the rendered worker
boundary.

## Components

### Console render test
- **Does**: Requires product metadata, primary interaction surfaces, and the connectome
  canvas with truthful exact-topology copy while excluding all starter-preview
  artifacts.

### Generation route test
- **Does**: Requires prompt validation, output length, and telemetry fields in either
  explicit demo or live-checkpoint mode.

### Organism clock route test
- **Does**: Rejects tick counts outside the bridge's bounded no-input interval before
  attempting a loopback proxy.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local runtime | Worker renders the console and serves `/api/generate` plus `/api/organism` | Route or product-copy changes |
