# `train.py`

## Purpose

Runs the same resumable continuous training loop for the SOL2 organism, matched GRU,
and matched transformer.

## Components

- `build_model` — deterministic organism/control construction and parameter matching.
- `run` — persistent stream, chunked BPTT, guarded updates, health logging, held-out
  interventions, atomic checkpoints, and exact resume.
- CLI/config helpers — reproducible one-arm launch surface.

## Decisions

- S0-R defaults to constant `1e-3` for 24k updates. A finite cosine may be selected as
  a disclosed control but is not the primary indefinite-plasticity regime.
- Transformer chunk length equals its context length and batch is rescaled to preserve
  tokens per update.
- Module gradient norms, per-cell material/reserve gradient fractions, organ attention,
  and effective operator norms are logged alongside BPC.
- A rejected update advances lived state and stream position but cannot poison weights.

## Contracts

- State detaches at optimizer boundaries but never resets.
- Evaluation never mutates training state.
- Checkpoints include the exact next stream position and optimizer moments.
- Baselines receive identical token, optimizer, and schedule budgets.
- Two hundred consecutive rejected gradients abort loudly instead of producing a zombie
  run.
- Reserve cells are reported rather than penalized; the gate requires nontrivial
  internal participation, not universal utilization.
