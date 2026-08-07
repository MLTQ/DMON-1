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

Pending.
