# `eslint.config.mjs`

## Purpose

Defines TypeScript and React quality rules for the console source.

## Components

### ESLint configuration
- **Does**: Applies the Next.js flat configuration while excluding generated output.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `npm run lint` | Source files are checked without generated artifacts | Ignore or rule changes |
