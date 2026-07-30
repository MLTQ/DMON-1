# `vite.config.ts`

## Purpose

Composes vinext and the local Cloudflare worker emulator for development and
production-like local builds.

## Components

### Vite configuration
- **Does**: Builds and runs the application worker entirely on the local machine.
- **Interacts with**: `worker/index.ts`.
- **Does not**: Bind the project to a hosting provider or package deployment metadata.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local development | `npm run dev` and `npm run build` use the same app shape | Plugin ordering |
