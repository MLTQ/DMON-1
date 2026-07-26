# `route.ts`

## Purpose

Provides a deterministic character-level browser demonstration for the hosted SOL console.
It is an explicit integration seam for a later live PyTorch inference bridge.

## Components

### `POST`
- **Does**: Validates a prompt, generates a bounded character continuation, and returns
  demonstration telemetry.
- **Rationale**: The hosted UI remains interactive without pretending that a Python
  checkpoint runs inside the Cloudflare worker.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `app/page.tsx` | Returns `{ output, mode, metrics }` | Response fields or metric names |
| Tests | Empty prompts return 400; valid prompts return deterministic text | Validation semantics |
