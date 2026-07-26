# `sites-vite-plugin.ts`

## Purpose

Provides the bundled Sites build integration that turns the vinext application into a
Cloudflare Worker-compatible deployment.

## Components

### Sites Vite plugin
- **Does**: Stages hosting metadata and validates the deployment output.
- **Interacts with**: `vite.config.ts` and `.openai/hosting.json`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Sites packaging | Hosting metadata reaches `dist/` | Plugin export or staging behavior |
