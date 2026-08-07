# G2a: relevance-gradient transport probes

Status: preregistered 2026-08-06, after the G1 verdict and before any probe
execution. No training; runs on the frozen G1 u300 checkpoint.

## Question

G1 showed that *any* passage carried through the organism improves held-out
frozen-Qwen answers (+0.13–0.32 mean loss vs floor). What survives the
FIFO→injection route: prose, domain vocabulary, token soup, or mere traffic?

## Setup

Seven conditions per heldout question, fresh episode each, G1 checkpoint
(u300 — the u150 best was not retained; a known limitation recorded in the G1
result): `bare` (organism-free), `none`, `own`, `mate`, `off_domain`
(different-family meta_train passage), `shuffled_own` (order destroyed,
unigrams kept), `pool_random` (corpus-unigram soup, length-matched). fp32
scoring throughout; deterministic per-question generators.

Command: `python -m fable2.probes --checkpoint runs/fable2-g1/broca.pt
--local-files-only --out runs/fable2-g2a/probes.json`.

## Preregistered readings

The gradient differences (positive = the more-structured exposure is better):

1. **Content world**: `own < shuffled_own < pool_random < none` with a material
   prose-vs-shuffled step — passage structure survives transport. The G2 data
   hypothesis keeps its mechanism; proceed to G2 as planned.
2. **Vocabulary world**: `own ≈ shuffled_own < pool_random` — unigram identity
   survives, order does not. G2 proceeds, but its prediction is downgraded:
   a differential would have to ride on vocabulary overlap, so the G2 corpus
   must control lexical overlap between mates (recorded as a G2 design input).
3. **Traffic world**: `own ≈ shuffled_own ≈ pool_random`, all ≪ `none` — the
   transport is an activation/calibration artifact of tokens having passed
   through. G2 still runs (it is the licensed data test and this probe is one
   checkpoint, not a law), but a G2 null then indicts the interface, and the
   next treatment is architectural (question-conditioned recall at scoring
   time), not more data.

`off_domain` vs `mate` separates "wiki-domain prior" from "counterfactual
passage" effects; `bare` vs `none` measures what an untrained-content organism
pass costs or buys by itself.

This is a characterization on 8 questions at one checkpoint — patterns near the
per-question noise scale (the G1 eval-to-eval wobble, ~0.05 mean loss) are
reported as inconclusive, not narrated into a world.

## Result

Complete 2026-08-06 on the G1 u300 checkpoint, heldout split. Artifact:
`runs/fable2-g2a/probes.json` on Aine (copied to
`g2a-transport-probes-result.json`).

Mean losses: bare 0.5465 · none 0.5997 · own 0.4201 · mate 0.3600 ·
shuffled_own 0.3832 · pool_random 0.3867 · **off_domain 0.8413**.

**Reading: traffic world (preregistered reading 3), plus a sharper finding the
preregistration did not anticipate.**

- The generic benefit is content-indifferent: corpus-vocabulary token soup
  reproduces it in full (pool_vs_none +0.213; shuffled_vs_pool +0.003), and
  intact prose is no better than its own shuffle (prose_vs_shuffled −0.037).
- An exposure-less organism pass actively hurts (none 0.60 vs bare 0.55): the
  trained injection is miscalibrated until ~250 tokens of anything have passed
  through.
- **Trained passages are poison off-pair.** `off_domain` (a meta_train passage
  on a heldout question) is catastrically worse than everything, including no
  exposure (+0.24 over none). The organism learned specific responses to its 12
  training passages that push hard toward the wrong pair's answer.

Mechanistic synthesis for G1's "differential that didn't generalize": the paired
objective's live-wrong term can be satisfied by **sabotaging mismatched pairs**
(make any recognized-passage/question mismatch worse) instead of using matched
content. The no-harm hinge guards only the exposed arm, not the wrong arm. A
sabotage solution produces exactly what was observed: large train-time
"differential", generic benefit on novel passages, no heldout differential, and
off-domain catastrophe on trained passages.

Consequence, per the preregistered boundary plus this addendum: G2 proceeds as
the licensed data test, but running it with an objective whose degenerate
optimum is now *demonstrated* would be preregistered waste. G2's treatment
therefore includes an anti-sabotage hinge (wrong-passage arm must not fall below
the detached no-exposure baseline), recorded as a pre-execution amendment in
`g2-corpus-scaling.md` and justified by this probe. A G2 null after both
changes indicts the interface architecture, not the data.
