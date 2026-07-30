# F2: adaptability under regime cycling (preregistered 2026-07-30)

## Why this experiment exists

The project's goal is a dynamic organism as a vessel for intelligence; held-out
BPC on a stationary corpus measures compression, which is a stepping stone and
not the destination. This is the first fable experiment whose question is
*adaptability*: does the creature's persistent, structured state buy anything a
matched GRU's weights-plus-hidden-vector do not, when the world changes?

The prior art is a warning. `dmon/exp/nonstationary.py` (exp1, runs/exp1) ran
entropy-matched vocabulary-permutation switches every 4,000 chars and the GRU
won outright (2.01 vs 2.20 well-after-switch); all arms recovered within ~10
characters. Its own verdict: that design measures **in-context inference, not
continual adaptation**. F2 is designed so in-context inference cannot pass it:
what is scored is improvement *across regime revisits* (savings) and damage
*across regime absence* (interference) — properties of the learner's lifetime,
not of any single context window.

## Design

Stream: the training region is truncated to a multiple of 2×`block` and tiled
into alternating regimes — even blocks are Tiny Shakespeare (regime A), odd
blocks are the same text under a fixed vocabulary permutation (regime B;
entropy-, unigram- and structure-matched, so post-switch excess loss is pure
adaptation cost; permutation guard from exp1: fewer than vocab/8 fixed
points).

**Lane alignment (the exp1-shaped trap):** lanes are spaced `2·block` apart so
every lane is in the *same* regime at every moment. Staggered lanes would feed
the weights a stationary A+B mixture — no coherent regime switch would ever
reach the learner, and the experiment would silently measure nothing.

- `block` = 24,576 chars; batch = 4 lanes; chunk = 32; updates = 8,000
  → each lane walks ~10.4 blocks; ~5 visits per regime per lane
- **Constant LR after warmup** (lr = lr_min = 1e-3, warmup 200) for every arm.
  Stated openly: sol S12 showed cosine decay is a large *compression* win, but
  an annealed learner is structurally committed to a stationary world — decay
  trades away exactly the plasticity F2 measures. Both arms get the identical
  constant schedule, so the comparison stays fair; the tension between S12 and
  this choice is the plasticity–stability tradeoff made concrete.
- Geometry: F0 creature geometry (128 cells, h=128, K=12, ports 16/16/32,
  spt=4), GRU matched as in F0.
- Per-token raw records kept for every arm: (lane, position-in-block, regime,
  loss) — exp1 lost an arm to discarded raws; not repeated.

Arms (× seeds 7, 13, 21):

| Arm | Stream |
|-----|--------|
| `cycled_creature` | A/B alternating |
| `cycled_gru` | A/B alternating |
| `aonly_creature` | A only (control for interference) |
| `aonly_gru` | A only |

## Measures (second half of each run only, per exp1's discipline)

1. **Adaptation curve**: mean BPC vs chars-since-block-start, log-spaced bins
   (exp1's committed binning), per regime. Descriptive.
2. **Savings**: per regime visit v, cost(v) = mean excess BPC over the first
   2,048 chars of the block relative to that same block's final-quarter mean.
   The savings curve is cost(v) vs v. *Adaptive learner: cost falls with v.*
3. **Interference**: final-quarter BPC in A-blocks vs the A-only arm at
   matched cumulative A-chars (the cycled arm sees half as many A-chars per
   update; comparisons are indexed by A-char count, not update count).
4. Compression sanity: A-only arms' final BPC, so F2's constant-LR regime can
   be compared against F0's cosine numbers and the S12 cost of staying
   plastic is on the record.

## Pass / fail (creature vs GRU, 3 seeds)

- **Pass**: creature shows steeper savings (cost(v) declining faster) or less
  interference than the GRU, consistently across seeds, with effect size
  clearly above seed spread — while staying within 0.15 BPC of the GRU on
  compression in-regime (an organism that adapts by being uniformly bad has
  not adapted).
- **Fail**: creature and GRU indistinguishable on savings and interference
  (the substrate adds no adaptability at this scale — recorded, not spun), or
  creature wins adaptability only by violating the compression floor.
- Either way: if *neither* arm shows nonzero savings, the design still
  measures in-context inference and the block length must be revisited before
  any claim is made (the exp1 null repeated in new clothes).

## Launch

```bash
# on Aine, after F0 relaunch completes
bash fable/run_f2.sh
python3 -m fable.adapt_report --root fable/runs/f2 --out fable/runs/f2/REPORT.md
```
