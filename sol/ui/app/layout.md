# `layout.tsx`

## Purpose

Defines the SOL site shell, typography, metadata, and icons.

## Components

### `generateMetadata`
- **Does**: Supplies the title, description, and host-correct Open Graph/X social
  preview URL.
- **Interacts with**: `public/og.png`.

### `RootLayout`
- **Does**: Applies Geist text and monospaced typefaces to every route.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| All routes | Global stylesheet and font variables are present | Removing imports or body classes |
