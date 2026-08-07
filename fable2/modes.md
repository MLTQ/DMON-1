# modes.py / train_modes.py

## Purpose

The M-line: mode acquisition through the Broca graft. A mode is a
low-dimensional behavioral policy (M0: response language, French vs German)
acquired from demonstrations streamed through tissue and expressed as a
continuation-likelihood shift on held-out prompts. Preregistration:
`experiments/m0-acquired-language-mode.md`. This line exists because three
G-line experiments falsified content transport while repeatedly demonstrating
modulation transport.

## Components

- Corpus (`build_mode_corpus`): Qwen-generated translation twins over neutral
  English prompt topics; gates are deterministic and behavioral — stopword
  language identification, twin length parity, demo-format length ≤ 56 tokens
  (so K=4 demos fit the 256-slot FIFO), and per-item **bare-indifference**
  (the floor must not already prefer a language, or margins measure the prior).
- `ModeBank` — cached demo/twin features and token spans; deterministic
  sha-seeded demonstration sampling (resampling replication varies these).
- `run_mode_arm` — the G-line arm battery with mode semantics (`wrong_mode` =
  other-language demonstrations); scores *both* twins per episode.
- `paired_mode_loss` — G2's three hinges over twin log-likelihoods: live-wrong
  differential, no-harm vs detached neutral, anti-sabotage on the wrong arm.
- `train_modes.py` — mirrors `train.py` (same telemetry keys, same
  checkpointing incl. per-eval retention, same guards); episodes alternate
  exposure direction so a fixed-language bias cannot masquerade as a mode.

## Decisions

- **Twins control content by construction**: both continuations say the same
  thing, so likelihood margins isolate language, not meaning.
- **Evaluation counts item×direction episodes** and `mean_loss` is the negative
  exposed-twin log-likelihood, so `fable2.curves`, strict wins, and trend
  verdicts work unchanged (`curves.ARM_ALIASES` maps `wrong_mode` into the
  counterfactual panel slot).
- Telemetry key `advantage_vs_wrong_passage` carries the wrong-*mode*
  differential in mode runs — kept for instrument compatibility; the result
  JSON's `experiment` field disambiguates.

## Contracts

Language gates, twin parity, zero-init floor identity, arm distinctness and
determinism, loss arithmetic incl. the sabotage configuration, and a 60-update
toy learnability smoke — all in `test_fable2.py`.
