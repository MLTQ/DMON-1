# test_fable2.py

## Purpose

The CPU contract gate. Runs with `python -m fable2.test_fable2`, no pytest, no
downloads. This gate must pass before any GPU time.

## Components

- `TinyTokenizer` — deterministic (sha-based) whitespace tokenizer satisfying the
  wiki-memory tokenizer surface; labels encode to single distinct tokens.
- `toy_corpus` — nine three-split documents with one question each, distinct
  source families.
- Twelve contracts: config round-trip and guards; depth resolution; injection
  bounds + floor identity + live-zero gradient; bitwise-silent fresh graft and the
  zero-control≡floor invariant; output-tissue read boundary; pairwise arm
  distinctness after head recruitment (including a depth lesion); two-stage
  gradient recruitment with a frozen backbone; paired-loss arithmetic and
  detachment; answer-card leakage lint and empty-split guards; mate constraints;
  bitwise checkpoint round-trip; and a 60-update learnability smoke that must
  descend and end with correct-passage > wrong-passage.

## Decisions

- **The smoke is a dynamics regression test, not a benchmark.** It is the exact
  harness that exposed the common-mode ratchet and the coefficient-saturation
  collapse before either could reach a GPU; if the objective or organ regress into
  a static-control optimum, this test fails first.
- Checkpoint tests run in eval mode because training-mode forwards advance the
  spectral power-iteration buffer (order-dependent state).
- The toy bank retries permutation seeds until mate constraints are satisfiable —
  determinism per seed is preserved; an unsatisfiable corpus still raises.

## Contracts

Meta: every assertion message states what broke and why it matters; a vacuous
check (one that cannot fail) is treated as a defect — the fable line shipped one
(`smoke_test.py:73`, `or True`) and it was found only in review.
