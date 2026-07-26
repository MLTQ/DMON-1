# `index.ts`

## Purpose

Provides the Cloudflare Worker entrypoint emitted by the vinext build.

## Components

### Worker export
- **Does**: Delegates incoming requests to the generated application handler.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Sites runtime | Default export is a Worker-compatible fetch handler | Export shape |
