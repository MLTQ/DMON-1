# retention.py

## Purpose

M1a: the retention curve — how long an acquired mode survives neutral traffic
streaming through the organism. Measurement only, on a frozen checkpoint; the
preregistered prediction and reading rules live in
`experiments/m1a-retention-curve.md`.

## Components

- `build_filler_stream` — one fixed English filler (meta_train prompts,
  repeated to the ladder maximum, no French/German content), encoded once;
  every N uses a nested prefix, so ladder points differ only in traffic length.
- `scored_with_filler` — a mode arm with N filler tokens observed (memory
  writes on, exactly as lived experience) between exposure and scoring; the
  filler is identical in every arm, so the demonstration language remains the
  only difference.
- `run_retention` — the ladder over heldout direction-episodes; records the
  wrong-mode differential, raw and floor-anchored mode margins (the M0
  instrument upgrade: the no-exposure+filler arm at the same N is the anchor),
  strict wins, and the named quantities N½, N₀, residual-at-max.
- `render_curve` (`--plot`) — two panels with the FIFO span marked; run
  locally on the JSON.

## Decisions

- **Filler is observed, not skipped**: writes go to memory because the question
  is precisely whether lived time evicts the mode. A no-write filler would
  measure nothing.
- The no-exposure arm also receives the filler, so the anchor moves with any
  filler-induced drift and the anchored margin isolates the demonstrations'
  residue.
- No pass/fail gates: the deliverable is the curve; the only illegitimate
  outcome is not publishing it.

## Contracts

- `scored_with_filler` at N=0 reproduces `run_mode_arm`'s likelihoods exactly
  (same seed, same arms); rejects unknown arms (contract-tested).
