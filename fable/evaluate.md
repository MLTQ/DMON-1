# evaluate.py

## Purpose

Prequential held-out evaluation with causal ablations that separate *which*
state is doing the work.

## Components

- `_prequential` — B=1 walk over the holdout: `warmup` unscored, `scored`
  counted; optional per-token `mutate` hook, `reset`, `frozen_idx`.
- `evaluate_model` — normal pass (any arm).
- `evaluate_with_ablations` — creature: normal, reset-each-token,
  shuffle-internal, mirror-zero, plus the three deltas.

## Decisions

- **No label feedback at eval.** grok's eval called `observe_prediction` with
  the true target, mutating persistent credit state — an information pathway
  its GRU control lacked. Fable has no such pathway anywhere.
- **Shuffle permutes internal cells only.** grok permuted every cell including
  ports and the readout correspondence, which an untrained model also fails —
  too strong to distinguish trained substrate from wiring (debt #26).
- **Mirror-zero is its own ablation.** The reset delta bundles the mirror
  ring's verbatim n-gram window with distributed internal state (debt #23);
  `mirror_delta_bpc` prices the ring alone, so `reset − mirror` bounds the
  distributed-state share.

## Contracts

- All passes run on the same holdout window with the same warmup, so deltas
  are directly comparable.
- The shuffle permutation is fixed (generator seed 1729) and re-applied every
  token.
