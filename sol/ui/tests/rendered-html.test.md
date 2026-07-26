# `rendered-html.test.mjs`

## Purpose

Protects the local SOL console and its checkpoint-proxy fallback at the rendered worker
boundary.

## Components

### Console render test
- **Does**: Requires product metadata, primary interaction surfaces, and the connectome
  canvas while excluding all starter-preview artifacts.

### Generation route test
- **Does**: Requires prompt validation, deterministic output length, and telemetry fields.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local runtime | Worker renders the console and serves `/api/generate` | Route or product-copy changes |
