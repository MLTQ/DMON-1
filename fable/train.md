# train.py

## Purpose

One training loop for every arm — creature, matched GRU, head-only bypass — so
"matched" is enforced by code sharing, not by discipline.

## Components

- `build_schedule` — linear warmup → cosine to `lr_min` over `updates`.
- `build_optimizer` — AdamW with decay/no-decay param groups.
- `build_model` — seeded constructors; `gru` sizes itself against the
  creature's true parameter count via closed form; `bypass` is the creature
  with everything frozen except `readout` + `out_norm` (an echo-state /
  reservoir control: if a head on an *untrained* substrate approaches the full
  creature, the substrate's training added nothing).
- `run` — the loop; writes `<kind>.json` (config, params, history, evals) and
  `<kind>.pt`. `on_update` hook is how grow.py injects the F1 graft.
- CLI: `python -m fable.train --model {creature,gru,bypass} ...`.

## Decisions

- **Warmup + cosine for every arm.** sol S12 found late decay was its largest
  gain (−0.114 best BPC) and never mirrored it into a control; grok used
  constant LR everywhere and NaN'd twice. Both arms read the same schedule
  fields here by construction.
- **Non-finite gradient guard, not just clipping.** 128 sequential GRU
  applications can overflow fp32 *before* clipping, and `clip_grad_norm_` of
  an inf norm multiplies every parameter by NaN — one poisoned step and the
  run is dead (the likely chase-1 mechanism: NaN at u~5200 with clip=1.0
  active). Overflowing chunks are skipped and counted (`skipped_total`,
  reported in the LADDER; the log must not be able to lie).
- **Weight decay only on genuine weight matrices** — not biases, norms,
  embeddings, edge logits, or the sensory affine (grok decayed all of them,
  debts #9/#27).
- **Checkpoints carry `model.cfg`, not the run's initial config** — a grown
  model is bigger than the config it started from.
- GRU init is seeded (`seed + 1000`) and reproducible; grok's control
  inherited leftover RNG state (debt #16).

## Contracts

- History entries carry `bpc, lr, tokens_per_s, grad_norm, skipped_total` and
  (creature arms) `h_max, msg_rms, logit_absmax`.
- Eval cadence: every `eval_every` and at the final update; creature evals
  include the three ablations, GRU evals are `normal` only.
- A skipped update still advances the LR schedule (schedule is a function of
  update count only).
