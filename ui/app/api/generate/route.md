# `route.ts`

## Purpose

Proxies the local console to the loopback SOL checkpoint bridge and falls back to an
explicit deterministic demonstration when the Python process is not running.

## Components

### `POST`
- **Does**: Validates a prompt, calls `http://127.0.0.1:8765/generate` (or
  `SOL_BACKEND_URL`), and preserves the live checkpoint response.
- **Does**: Uses a deterministic local demonstration only when the bridge is unreachable
  or has an internal failure.
- **Rationale**: The UI stays usable during frontend work while clearly identifying
  whether output came from PyTorch.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `app/page.tsx` | Returns `{ output, mode, metrics }` | Response fields or metric names |
| Tests | Empty prompts return 400; valid prompts return deterministic text | Validation semantics |
| `sol/serve.py` | Local `/generate` accepts prompt and length | Bridge path or response shape |
