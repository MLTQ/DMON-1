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

Complete 2026-08-06, 300 updates on the physical RTX 4090 (122 min, peak 15.29
GiB, zero gradient rejections). Artifacts: `g1-broca-paired-result.json`
(SHA-256 `6857d3eb59c65ee2b393c0ad32007ecd345d57a616f679c203bd88bc8538cd3f`),
`g1-broca-paired-trend.json`, `figures/g1-*.png`.

**Verdict: FAIL on the content-differential gates, with real subsidiary
findings.** Not an undertrained fail — the train-time rule reads the run at its
best updates and reports the tail as overtraining waste; no resume is
sanctioned.

Gate outcomes (heldout):

1. Mean loss at u300: `normal` (0.4201) beats `bare_floor` (+0.126),
   `no_exposure` (+0.180), `memory_lesion` (+0.180), `internal_lesion` (+0.073)
   — but **loses to `wrong_passage` by −0.060**. Gate 1 fails.
2. Strict wins at u300: 7/8 vs no-exposure and both tissue lesions, but 2/8 vs
   wrong passage and 4/8 vs floor. Gate 2 fails. At the wrong-margin's best
   update (u75) the mean margin was positive (+0.026) but strict wins were still
   3/8 — the differential never had a majority at any read point.
3. Depth attribution passes: at u150 all four depth lesions hurt (7/7/4/4 of 8);
   at u300 depth-7 lesion still hurts 6/8 (+0.121).
4. Tissue-causation passes: memory and internal lesions cost well above the 0.01
   bar at every read point.

Train-time verdicts (`trend.json`): heldout normal loss `degrading_tail`
best@150 (0.2312 there vs 0.4201 at stop — the last 150 updates gave back nearly
half the improvement); heldout margin-vs-wrong `degrading_tail` best@75, ending
negative; train loss `plateau` since ~u74. A transient instability spiked
heldout normal loss to 1.32 at u175 (recovered by u200) without tripping the
gradient guard — visible only in the curves, which is exactly why they are now a
required artifact.

What the run established:

- **A strong, content-nonspecific exposure effect.** Any passage carried through
  the organism improves frozen-Qwen answers dramatically over floor and
  no-exposure (u150: +0.32 vs floor, 7/8; u75: 8/8). The organism transports
  *something* real across context erasure through causally necessary tissue and
  injection sites. What it transports is not the passage's distinguishing
  content: on held-out questions the mate passage helps as much or more.
- **Content discrimination trains but does not generalize.** Train advantages
  vs wrong passage reached 0.5–0.7 log-prob on meta_train while the heldout
  differential decayed — consistent with binding the 12 training pairs rather
  than learning a general read-from-memory operation. The suspect bottleneck is
  corpus size (12 meta_train questions), not organism capacity or duration —
  both already isolated by the C1 series and prohibited as responses.
- **All cross-erasure influence lives in the memory FIFO.** `memory_lesion` and
  `no_exposure` are numerically identical at every evaluation: the recurrent
  internal state carries nothing across the erasure boundary (contractive
  dynamics wash it out within a question's length). Retention-in-tissue claims
  for this architecture are claims about stream-written memory plus its readout,
  not about persistent recurrent state.
- **The overtraining tail has an anatomical signature**: by u300 the depth-15
  lesion *helps* (−0.053) — mid-depth sites drifted harmful while depth 7
  remained load-bearing.

Per the stop rules: no update extension, no capacity increase, no loss-weight
escalation. The next moves are analytical: (a) expand the paired corpus (more
questions, more families — data, which nothing in the review prohibits) before
concluding the differential cannot generalize; (b) treat the generic exposure
effect as the currently-real capability and characterize what is being
transported; (c) retain per-eval checkpoints in future runs so best-update
models survive their own tails.
