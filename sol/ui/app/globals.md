# `globals.css`

## Purpose

Defines the visual system and responsive layout for the SOL scientific instrument,
including signed-edge legend and continuous output states.

## Components

### Tokens and global shell
- **Does**: Establishes the dark biological palette, typography, focus treatment, and
  ambient background.

### Instrument panels
- **Does**: Styles the connectome canvas, measured signed-edge legend, telemetry,
  energy gauge, expanded cell inspector, composer, and scrolling output stream.

### Responsive behavior
- **Does**: Reflows telemetry and prompt/output panels for tablet and mobile widths.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `page.tsx` | Component class names map to the instrument layout | Renaming selectors |
| Accessibility | Focus indicators and reduced-motion behavior remain available | Removing focus or motion media rules |
