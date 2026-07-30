# `index.ts`

## Purpose

Provides the Worker-shaped entrypoint used by Vinext's local runtime.

## Components

### Worker export
- **Does**: Delegates incoming requests to the generated application handler.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local Vinext runtime | Default export is a Worker-compatible fetch handler | Export shape |
