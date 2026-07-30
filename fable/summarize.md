# summarize.py

## Purpose

Merge per-arm run JSONs under a runs root into one LADDER.md — the artifact a
verdict is read from.

## Components

- `summarize(root)` — walks arm directories, pairs `creature.json` /
  `gru.json` / `bypass.json` / `growth.json`, emits the table + mean gap.

## Decisions

- The table carries `Skipped` (non-finite-gradient updates) and the three
  ablation deltas next to every BPC so a number cannot look healthy while its
  run was sick (the log must not be able to lie).
- Gap is `creature − gru` on final held-out BPC, the same convention as grok's
  ladders, so tables are comparable across spikes.

## Contracts

- Missing files render as "—", never as fabricated zeros; NaN BPCs print as
  nan and are excluded from the mean gap (their presence is visible in the
  per-arm row and Skipped column).
