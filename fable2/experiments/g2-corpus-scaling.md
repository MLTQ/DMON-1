# G2: paired-corpus scaling with a sabotage-guarded differential

Status: preregistered 2026-08-06, after G1's verdict and the G2a probes, before
any G2 corpus generation or optimizer update.

## Question

G1's content differential trained (0.5–0.7 log-prob on 12 meta_train questions)
but did not transfer to heldout. Capacity and duration are already falsified as
remedies (C1 series) and prohibited. Data was never moved. Does a 6× training
corpus, under an objective whose demonstrated degenerate optimum is closed,
produce a held-out content differential?

## Two treatment changes from G1, both preregistered here before execution

1. **Corpus at 4.5× documents / 6× meta_train questions**: 36/6/12 documents
   (72/12/24 questions) built by `fable2/corpus.py` from the wiki vault, with
   the behavioral admission gate (bare Qwen fails closed-book AND passes
   open-book, fp32, permutation seed 101). 24 heldout questions bring strict-win
   resolution from 12.5 to ~4 points.
2. **Anti-sabotage hinge** in the paired objective: the wrong-passage arm may
   not fall below the detached no-exposure baseline. Justification: the G2a
   probes showed G1's differential was substantially *sabotage* — trained
   passages were poison off-pair (0.84 vs 0.60 no-exposure) while token soup
   reproduced the full generic benefit. Running G2 with the known degenerate
   optimum open would be preregistered waste. The change is one zero-margin
   hinge (`episodes.py`), contract-tested against the sabotage configuration.

Everything else is unchanged from G1: organism (408 cells, seed 7), depths
resolved from fractions (0.25/0.5/0.75/1.0 → layers 7/15/23/31), control shape,
lr 1e-3, 300 updates, eval every 25, fresh-episode policy and its caveat.
Per-eval checkpoints are retained this time (`broca-u<N>.pt`), closing G1's
lost-best-model gap.

## Amendment 2 (pre-training, during corpus generation)

The first build attempt admitted ~3% of documents under a 2-of-2 rule (both
drafted questions must pass both gates), projecting under-quota on the full
vault. Questions are now gated individually and quotas count questions
(72/12/24), not documents; a document contributes its admissible questions
(one or two) and documents never straddle splits. The per-question gate itself
is unchanged — the 2-of-2 rule was discarding admissible questions for their
siblings' failures, which added strictness without adding rigor. Per-question
gate outcomes (closed-book/open-book) are now logged for the record. No G2
training had run at amendment time.

## Attribution note

G2 moves two factors relative to G1 (data, objective guard). This is accepted
and recorded: G2a demonstrated the unguarded objective's optimum is degenerate,
so "6× data under the broken objective" is not an arm worth a GPU-run; the
guard is a fix, not a treatment. If G2 passes, the differential's existence is
established; attributing it between data and guard is a follow-up ablation
(guarded objective on the C1-size corpus), run only if that attribution matters
to the next step.

## Primary metric and gates

Identical to G1 (heldout mean correct-answer log-likelihood per arm; strict
wins now out of 24): normal must beat wrong passage, bare floor, no-exposure,
memory lesion, and internal lesion in mean loss; strict-win majority (≥ 13/24)
vs wrong passage and floor; at least one depth lesion hurts a majority; tissue
lesions cost > 0.01. Train-time artifact (`fable2.curves`) required before
interpretation; read-at-best rule and the single sanctioned resume-to-600 on a
`still_improving_at_stop` fail both carry over.

Additional G2 gate, from the G2a finding: at the read point, the
**wrong-passage arm must not sit below no-exposure by more than 0.02 mean loss**
— a differential achieved by re-learning sabotage under the hinge (e.g., via
the margin term overpowering it) does not count.

## Stop rules

- Corpus generation failing its quotas (fewer than 54 admissible documents) is
  a recorded infrastructure result, not a license to weaken the admission gate.
- A gate fail at the read point: stop; no second data scaling, no capacity or
  duration moves. The residual suspect becomes the interface design
  (final-state-only controls), and the next treatment is architectural
  (question-conditioned recall at scoring time), preregistered separately.
- A pass: proceed per HANDOFF order — scaffold withdrawal, interleaved
  continuous lifetimes, delayed retention — before any growth or anchors.

## Result

Complete 2026-08-07, 300 updates on the physical RTX 4090 (~6.7 h). Artifacts:
`g2-corpus-scaling-result.json`, `g2-corpus-scaling-trend.json`,
`g2-corpus.json`, `g2-u75-permutation-verify.json`, `figures/g2-*.png`.

**Verdict: FAIL — no held-out content differential. Corpus scale is exonerated
alongside capacity and duration; per the preregistered decision boundary the
residual suspect is the interface design, and the next treatment is
architectural.**

The gate battery at the read point technically passed, and that pass did not
survive verification — recording both halves in full:

- Every heldout trend series reads `degrading_tail` best@75. At u75, all
  preregistered gates passed: margins vs wrong (+0.039, 13/24 + 3 ties), floor
  (+0.041, 15/24), no-exposure (+0.064, 18/24), both tissue lesions > 0.05,
  depth-7 lesion hurting 13/24, and the anti-sabotage check comfortably clear
  (the wrong arm sat *above* no-exposure).
- A post-hoc robustness probe (recorded as post-hoc) re-scored the retained u75
  checkpoint under two fresh label-permutation seeds. The differential
  inverted: wrong-passage margins −0.033 (9/24) and −0.044 (8/24); no-exposure
  margins also negative. Real content use is invariant to answer-choice order;
  this was not. **The u75 pass was selection on noise** — read-at-best across
  12 eval rounds plus threshold-exact wins is a multiple-comparisons machine,
  and it manufactured a pass the permutation probe immediately falsified.

**Instrument amendment for all future runs**: a read-at-best pass counts only
if it survives re-evaluation of the retained checkpoint under at least two
fresh permutation seeds. Filed as a standing protocol change.

What G2 established beyond the verdict:

- **G1's generic transport effect was Qwen-prior-mediated.** On this corpus —
  where the admission gate pins the bare floor at chance (1.428 ≈ ln 4) — the
  generic any-passage benefit vanishes: at the best checkpoint, normal ≈ floor
  under fresh permutations. Priming helps only where the prior already knows.
- **The control policy overfits regardless of 6× data.** Development floor
  margin grew to +0.40 while heldout margins collapsed; from u150 the heldout
  injection turned actively toxic — by u250–300 *lesioning tissue helped*
  (organism arms 2.2–2.6 vs lesioned 2.2 vs floor 1.43). Train loss kept
  improving to best@208 through all of it. Textbook policy memorization, now
  demonstrated at two corpus scales.
- **The anti-sabotage hinge worked as designed**: the wrong arm tracked
  at-or-above no-exposure throughout (train `wrong_vs_noexp` climbed from
  −0.05 toward 0; heldout wrong ≈ normal at every round). The G2a poison
  pattern did not recur. Closing the degenerate optimum did not rescue
  generalization — the failure is deeper than the objective.
- `memory_lesion ≡ no_exposure` held exactly at all 12 evaluations, again.
- Train-time verdicts: `degrading_tail` on every heldout series; ~225 of 300
  updates were overtraining waste (~3.5 h GPU) — measured, per protocol, not
  argued.

Per the stop rules: no second data scaling, no capacity or duration moves. The
next treatment (G3, preregistered separately) is the architectural one the
decision boundary names: question-conditioned read-out — the organism's memory
must be *queried at scoring time* rather than summarized into final-state
control banks that a question never conditions.
