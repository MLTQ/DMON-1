# stream.py

## Purpose

Continuous multi-lane character stream with a contiguous 90/10 train/holdout
split. No episodes, no resets — lanes wrap modulo the training region.

## Components

- `Corpus` / `load_corpus` — char vocabulary, contiguous split.
- `LaneStream` — `n_lanes` wrapping cursors; `next_chunk(L)` returns
  `[lanes, L]` inputs/targets and advances.

## Decisions

- **Lane starts are evenly spaced + seed-jittered.** grok's lanes were purely
  deterministic (`arange(B) * n//B`), so every same-B run saw the identical
  token order and cross-seed variance under-measured data variance (debt #28).
  The jitter is seeded, so a given seed remains exactly reproducible.

## Contracts

- Targets are inputs shifted by one within the same region; the last training
  token wraps to the first (one impure pair per lane per epoch, same as grok).
- Holdout is never streamed during training; evaluate.py reads it directly.
