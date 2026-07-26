# `vite.config.ts`

## Purpose

Composes vinext, React, Cloudflare, and Sites plugins for development and production.

## Components

### Vite configuration
- **Does**: Builds the application and worker using hosting declarations.
- **Interacts with**: `build/sites-vite-plugin.ts` and `worker/index.ts`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Development and Sites | `npm run dev` and `npm run build` use the same app shape | Plugin ordering |
