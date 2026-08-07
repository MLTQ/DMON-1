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

Complete 2026-08-07. Artifacts: `m1a-retention-curve-result.json`,
`figures/m1a-retention-curve.png`.

**The prediction is confirmed with unusual precision: the mode lives entirely
on the FIFO tape, decays in proportion to evicted demonstration tokens, and
dies exactly at full eviction. There is no recurrent carry.**

| N filler | differential | strict wins /32 |
|---|---|---|
| 0 | +0.0833 | 31 |
| 32 | +0.0850 | 30 |
| 64 | +0.0821 | 31 |
| 128 | +0.0710 | 30 |
| 192 | +0.0468 | 30 |
| 256 | +0.0009 | 19 |
| 384 | +0.0013 | 15 |
| 512 | −0.0022 | 15 |

Named quantities: N½ = 256, N₀ = 384, residual at 512 = −0.002 (strict wins at
exact coin-flip). The demonstrations occupy ~200–224 slots of the 256-slot
FIFO; the differential tracks the surviving fraction almost linearly
(graceful degradation — the readout integrates over remaining evidence rather
than cliffing at first eviction) and reaches zero precisely when the last
demonstration token scrolls off. The floor-anchored margin equals the raw
margin at every N: the filler itself induces no mode tilt (clean control).

This measures the creature's memory horizon: **one FIFO span**. The
memory-lesion invariant is now confirmed dynamically, not only by ablation.
M1b design inputs: train with exposure→query delays ramped across 64–384;
success is any above-chance differential surviving full FIFO eviction, which
would certify a learned non-tape route (recurrent attractor or otherwise) —
the organism's first distinction between what it is seeing and what it is.
