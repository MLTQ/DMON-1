# G1: paired behavior through the multi-depth Broca graft

Status: preregistered 2026-08-06, before any G1 optimizer update. Launches only
after G0 records `gates_all_pass: true`.

## Question

Can a passage alter the living substrate, survive context erasure, and later
change frozen-Qwen behavior *for the right reason* — through a differentiated,
bounded, output-tissue-only language organ injecting at four transformer depths,
with learned representational transformation?

This is the treatment the 2026-08-06 program review promotes after C1aa retired
exact same-coordinate transport. It changes the interface architecture; it does
not change the assay's causal standards.

## What is deliberately not here

Per the review's rejection list: no scale-up of tissue, no duration escalation, no
absolute auxiliary value/transport/semantic credits, no reference-centered
controls, no designated-answer cards, no coherent-recall scaffolds. The loss has
two preregistered terms (live-wrong differential at margin 0.1; zero-margin
no-harm versus detached no-exposure) and `task_weight` is 0.

## Setup

- Backbone, GPU, corpus: as G0. Fresh organism, seed 7, 408 cells
  (8 input / 256 memory / 96 compute / 32 relay / 16 output, hidden 96),
  private expression + rank-4 adapters on.
- Exposure is passage prose; questions are the C1 formatted four-choice prompts;
  episodes are fresh-lifetime (a developmental population — explicitly **not**
  lifetime-persistence evidence, per the review's fresh-episode caveat).
- 300 updates, AdamW, lr 1e-3 both groups, eval every 25 on development and
  heldout. Command: `bash fable2/launch_g1.sh <result-root> 300 25`.

## Primary metric

Held-out mean correct-answer log-likelihood (fp32) per arm, with per-question
strict wins. Four-choice accuracy is reported as plumbing only (8 heldout
questions = 12.5-point resolution; one question is not a result).

## Pass gates (all, on heldout, at the final evaluation)

1. `normal` beats `wrong_passage`, `bare_floor`, `no_exposure`, `memory_lesion`,
   and `internal_lesion` in mean loss.
2. Strict per-question wins are a majority (≥ 5/8) against `wrong_passage` and
   against `bare_floor`.
3. At least one depth lesion hurts on a majority of questions (the organ's depths
   are causally used, not decorative).
4. `memory_lesion` and `internal_lesion` each cost measurable loss (> 0.01 mean),
   so the effect runs through tissue, not through the effector alone.

## Train-time influence (amendment 1, added at update ~100, before any gate
## interpretation)

Every run must produce `python -m fable2.curves` output — figures plus
`trend.json` — before its gates are interpreted, and the recorded verdicts join
the result:

- If `heldout_margin_vs_wrong_passage` is `still_improving_at_stop`, a gate fail
  is recorded as **undertrained-fail**: the preregistered follow-up is one
  continuation of the same checkpoint to 600 updates (a resume, not a new
  treatment), exactly once. This is the single sanctioned duration extension.
- If the verdict is `degrading_tail`, the result is read at `best_update`'s
  checkpoint and the tail is reported as overtraining waste; no extension.
- A `silenced_depth_indices` warning is reported alongside any pass (a passing
  run that silenced depths is a narrower pass than it looks).

Operator direction 2026-08-06: both under-training and over-training have cost
this project real time; duration influence must be measured, not argued.

## Fail / stop rules

- Any gate fails at update 300 → record, stop, no extension of updates, no capacity
  increase, no loss-weight escalation. The next move is analysis, not scale.
- Gradient rejection or non-finite telemetry → the run dies loudly and the artifact
  records where.
- A development-split pass with a heldout fail is a fail (the C1 series' most
  common shape; it gets named, not excused).

## If it passes

Per HANDOFF.md order: withdraw developmental scaffolds, restore interleaved
continuous lifetimes (where `reset_after_exposure` returns as a real arm), test
delayed retention, then return to mode/thought-pattern curricula. Growth and
anchors come only after that, and only recruited growth counts.

## Result

Pending.
