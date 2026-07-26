# `rendered-html.test.mjs`

## Purpose

Protects the deployed SOL console and its hosted demonstration endpoint at the rendered
worker boundary.

## Components

### Console render test
- **Does**: Requires product metadata, primary interaction surfaces, and the connectome
  canvas while excluding all starter-preview artifacts.

### Generation route test
- **Does**: Requires prompt validation, deterministic output length, and telemetry fields.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Site deployment | Worker renders the console and serves `/api/generate` | Route or product-copy changes |
