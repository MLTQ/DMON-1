# baselines.py

## Purpose

The parameter-matched GRU null model. "The substrate is doing something" is a
measurement against this, on the identical stream, under the identical
schedule — never an assumption (HANDOFF discipline).

## Components

- `MatchedGRU` — Embedding → 1-layer `nn.GRU` → Linear; `step` and
  `forward_chunk` mirror the creature's interface so train.py cannot treat
  the arms differently.
- `gru_param_count` — closed form `6h² + (2V+6)h + V` (grok instantiated 125
  full models to count parameters, debt #5).
- `match_hidden` — argmin |params − target| over integer h.

## Decisions

- Matching targets the creature's **true trainable count** — fable has no dead
  parameter blocks to inflate it (see cell.md).
- The transformer control stays in sol/ where it ran; grok carried dead
  transformer-matching code it never called (debt #17).

## Contracts

- Match tolerance asserted < 2% in the smoke test; both counts and the chosen
  hidden size are recorded in the run JSON.
