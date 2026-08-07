# M1a: the retention curve — does the mode survive lived time?

Status: preregistered 2026-08-07, after the M0 qualified pass and before any
probe execution. Measurement only: no training, frozen M0 u300 checkpoint.

## Question

M0 proved acquisition: demonstrations → tissue → behavior across erasure. A
creature's mode must also survive *life continuing* — new experience streaming
through the same tissue. How long does the acquired mode persist as neutral
traffic displaces the demonstrations, and where does it die?

## Prediction (preregistered, from the program's own record)

`memory_lesion ≡ no_exposure` held exactly in all four experiments: everything
the organism carries across erasure lives in the 256-slot FIFO tape, and the
demonstrations occupy ~200–224 of those slots. The FIFO is a conveyor: N filler
tokens evict the oldest N demonstration tokens. Therefore:

- decay onset near N ≈ 32–64 (first demonstrations evicted),
- differential ≈ 0 by N ≈ 256 (demonstrations fully evicted),
- **any** reliable differential surviving N ≥ 384 would contradict the
  memory-lesion invariant and be evidence of recurrent-state carry — the most
  interesting possible outcome, and the least expected.

The measured cliff location is the creature's memory horizon; M1b's delayed-
expression training then makes moving that horizon the thing gradient descent
must solve.

## Protocol

Frozen M0 u300 checkpoint (`runs/fable2-m0/train/broca.pt`), heldout split, all
32 direction-episodes, floor-anchored margins reported alongside raw (the M0
instrument upgrade). For each N in {0, 32, 64, 128, 192, 256, 384, 512}:

1. Fresh episode state; observe demonstration stream (matching or wrong mode —
   both arms), memory writes on.
2. Observe the first N tokens of one fixed neutral English filler stream
   (meta_train prompts, no French/German content), memory writes on — the
   filler is identical in every arm, so the only difference remains the
   demonstration language.
3. Score both twins; record the wrong-mode differential, the mode margin, and
   the no-exposure+filler baseline at the same N (the floor anchor for that N).

One fixed demonstration sample seed per episode (the M0 replication already
established sample robustness); `python -m fable2.retention` writes the JSON,
`--plot` renders the curve.

## Reading rules

This is a measurement; the recorded output is the curve plus three named
quantities: N½ (first N where the differential falls below half its N=0
value), N₀ (first N where strict wins fall below majority), and the residual
at N=512. No pass/fail; the only illegitimate outcome is not publishing the
curve. M1b's design consumes these numbers.

## Result

Pending.
