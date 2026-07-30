# adapt.py

## Purpose

F2's machinery: the regime-cycling stream, the online trainer with per-token
raw loss records, and the savings/interference analysis. Measures
*adaptability* — lifetime properties of the learner — where F0 measures
compression.

## Components

- `sample_permutation` — vocabulary permutation with < vocab/8 fixed points
  (exp1's guard against near-identity permutations).
- `RegimeStream` — corpus tiled into A/B blocks; region truncated to a
  multiple of 2·block so wrapping preserves parity; lanes spaced 2·block so
  every lane is always in the same regime.
- `run_arm` — online training (creature or GRU) with per-token
  (position-in-block, regime, loss) records saved raw to `raw.pt`.
- `analyze` — second-half-only: log-binned adaptation curves, per-visit
  savings (early excess over that block's own steady state), steady-state
  BPCs for the interference comparison.
- CLI: `python -m fable.adapt --kind {creature,gru} --stream {cycled,aonly}`.

## Decisions

- **Lane alignment is the design's load-bearing wall.** Staggered lanes (the
  F0 stream) would feed the weights a stationary A+B mixture — no coherent
  regime switch would ever reach the learner and the experiment would
  silently measure nothing.
- **Savings are scored against the same visit's own final quarter**, not a
  global baseline, so drift in overall capability cannot masquerade as
  (anti-)adaptation.
- **Raw records are kept** (exp1 lost an arm to discarded raws). ~4 MB/arm.
- Partial blocks at the scoring boundary are dropped, not padded.
- The trainer duplicates train.py's guard/clip/schedule logic rather than
  importing its loop, because per-token unreduced losses change the inner
  loop's shape; the shared pieces (build_model/optimizer/schedule) are
  imported, not copied.

## Contracts

- `next_chunk` returns (inputs, targets, pos, parity) all `[lanes, L]`;
  targets are encoded under their own position's regime.
- Analysis only ever reads `raw.pt` — it can be rerun offline with different
  binning without touching GPU state.
