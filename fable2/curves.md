# curves.py

## Purpose

Train-time diagnostics for every fable2 run: rendered loss/margin/health curves
plus a *computed* verdict on whether training duration is influencing the result.
Added 2026-08-06 at operator direction — under-training and over-training have
both burned this project (sol S12's schedule artifact; C1aa's tail collapse;
fable F0's "holdout still descending at stop").

## Components

- `classify_series(updates, values, better)` — one metric trajectory →
  `still_improving_at_stop` (undertraining signal), `degrading_tail`
  (overtraining signal, with `best_update`), or `plateau` (with
  `plateau_since_update`). Epsilon is 2% of series range, floored — flatter than
  that is noise, not trend.
- `compute_trends(result)` — pure, render-free: smoothed train loss, per-split
  normal loss, margins versus wrong/floor/no-exposure, and a silence-collapse
  flag (per-depth control RMS falling below 10% of its peak — the C1x failure).
- `render_curves(result, trends, out_dir)` — two PNGs:
  `train-dynamics.png` (paired loss, advantages, per-depth control RMS, grad
  norm) and `eval-curves.png` (per-arm mean loss and margins-over-normal with
  depth lesions, both splits). Verdicts are printed into panel titles so the
  picture carries its own conclusion.
- CLI: `python -m fable2.curves --result <result.json> --out-dir <dir>` — writes
  `trend.json` + figures; runs locally on synced artifacts (no matplotlib needed
  on Aine).

## Decisions

- **A verdict is recorded, not inferred from a picture.** The trend JSON is part
  of the run artifact; preregistration gates may cite it (`g1` requires it before
  interpretation).
- **Fixed identity→hue assignment** (validated categorical palette): an arm keeps
  its color in every panel and every future figure; the floor is neutral dashed.
  One axis per panel; margins are drawn as control − normal so "positive = normal
  better" reads the same everywhere.
- Margins versus `memory_lesion`/`internal_lesion` are visible in the arm panel
  but not verdict-classified — the classified set is the three the G1 gates cite.

## Contracts

- `classify_series` verdicts are contract-tested on synthetic descending,
  V-shaped, and flat trajectories; `compute_trends` runs on a synthetic result
  and raises on empty telemetry/evaluations.
