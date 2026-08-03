# `stream.py`

## Purpose

Provides the identical never-reset character stream used by SOL2 and its controls.

## Components

- `Corpus` — vocabulary plus contiguous 90/10 train/holdout tensors.
- `load_corpus` — deterministic character encoding.
- `LaneStream` — evenly spaced wrapping cursors with exact checkpoint resume.

## Contracts

- Inputs and targets are adjacent positions on the same contiguous stream.
- Advancing by a chunk never creates an episode reset.
- Cursor state round-trips without storing or duplicating the corpus.
- Controls receive a separately constructed stream with the same seed and geometry.
