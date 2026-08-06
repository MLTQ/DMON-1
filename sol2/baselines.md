# `baselines.py`

## Purpose

Defines conventional closed-world controls that quantify SOL2's sequence-modeling
opportunity cost. They do not test growth, grafting, topology, or continuous evolution
and therefore do not determine whether the substrate succeeds overall.

## Components

- `MatchedGRU` — persistent recurrent control on the identical stream.
- `MatchedTransformer` — causal, stateless-within-chunk control.
- Closed-form parameter counters and nearest-width matchers.

## Decisions

- The transformer trains and evaluates at the same context length; untrained position
  embeddings cannot contaminate the comparison.
- Controls share stream, token budget, optimizer family, and schedule with SOL2.
- Initialization uses an independent deterministic seed in `train.py`.

## Contracts

- Parameter matching is based on trainable parameters only.
- GRU state persists across chunk boundaries.
- Transformer context does not silently persist across training chunks.
