# episodes.py

## Purpose

Deterministic paired wiki episodes over passage prose, the arm battery, and the
paired causal-contrast objective. This file is where the C1 instrument defects are
fixed structurally.

## Components

- `EpisodeBank` — encodes every exposure and formatted question exactly once
  through the frozen backbone; serves items by split; deterministic mate pairing.
- `run_arm(system, bank, item, arm)` — fresh-episode scoring under one of
  `EVAL_ARMS`: `normal`, `wrong_passage`, `no_exposure`, `memory_lesion`,
  `internal_lesion`, `bare_floor`.
- `run_depth_lesion` — the normal arm with one injection depth silenced.
- `paired_contrast_loss(exposed, wrong, neutral, margin)` — the G1 objective.

## Decisions

- **Exposure is the passage prose itself.** The C1 line trained on synthetic
  `Designated answer: <choice>` cards and evaluated on prose — a train/eval
  distribution shift its instruments never flagged. `FORBIDDEN_EXPOSURE_MARKERS`
  makes the bank raise on any such card.
- **Arms are distinct computations or they are not arms.** `zero_control` is
  scaffold-identical to the floor by design — asserted as an invariant, not
  reported. `reset_after_exposure` is computation-identical to `no_exposure` in
  fresh episodes — so it is simply not an arm here; it returns as a real arm only
  with continuous lifetimes, where pre-exposure state is nonzero.
- **The objective's engine is the live-wrong differential.** The wrong-passage arm
  is live, so content-independent inflation cancels exactly; the no-exposure
  baseline is detached (per `sol2.wiki_causal_contrast`) and demoted to a
  zero-margin no-harm hinge. Both alternatives were tried and failed measurably on
  the toy curriculum first: a detached wrong arm produced the C1y ratchet
  (baseline re-tracks, advantage decays, weights inflate), and a margined neutral
  term drove static inflation into coefficient saturation that erased the
  differential bit-exact. Whether exposure *helps* over no exposure is measured by
  the evaluation gates, not trained.
- **Mate constraints**: same split, different source family (the mate passage
  cannot support the question), different formatted correct label (margins compare
  distinguishable targets). Selection is deterministic; an unsatisfiable corpus
  raises at bank construction.

## Contracts

- Bank construction raises on: forbidden exposure markers, empty splits,
  unsatisfiable mate constraints, over-length texts.
- `paired_contrast_loss` raises on non-positive margin or mismatched questions;
  gradients cannot reach the neutral baseline.
