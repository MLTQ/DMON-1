# `postcss.config.mjs`

## Purpose

Enables Tailwind’s PostCSS transform for the global stylesheet.

## Components

### PostCSS configuration
- **Does**: Registers the Tailwind PostCSS plugin used by `app/globals.css`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Vite CSS build | Tailwind imports compile successfully | Plugin configuration |
