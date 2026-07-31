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

---

# RESULTS (2026-07-30, 12 arms, 8000 updates, 3 seeds, zero skipped updates)

## Savings, with the ongoing-learning control

The A-only arms walk the same position blocks with no regime change, so their
slope is general improvement. Regime-attributable savings is the difference.

| Kind / stream | Visit 0 | Visit last | Slope/visit |
|---|---:|---:|---:|
| creature / A-only | +0.1314 | +0.0547 | −0.0767 |
| creature / cycled | **+1.1613** | +0.9374 | **−0.2239** |
| gru / A-only | +0.0807 | +0.0280 | −0.0526 |
| gru / cycled | **+0.3444** | +0.2148 | **−0.1296** |

**Regime-attributable savings**: creature **−0.1471/visit**, GRU
**−0.0770/visit**. Consistent across all three seeds (creature −0.239/−0.216/
−0.217; GRU −0.134/−0.130/−0.125), with between-seed spread ~0.02 against a
between-arm difference of ~0.09.

**The design worked**: both arms show large regime-attributable savings, so
this is measuring cross-visit retention rather than in-context inference. The
`dmon/exp` trap was avoided.

## Interference

| Kind | Cycled A steady | A-only steady | Gap |
|---|---:|---:|---:|
| creature | 2.4163 | 2.2172 | **+0.1991** |
| gru | 2.2209 | 2.0469 | **+0.1739** |

No creature advantage; the creature is slightly *worse*. (Both gaps bundle
interference with halved A-exposure, so only the difference is comparable.)

## Compression floor

A-only steady state: creature **2.2172**, GRU **2.0469** — creature
**+0.1702 BPC** behind, exceeding the preregistered 0.15 proviso.

## Amendment: zero-shot retention (the A→B→A test), added same day

Savings measures the *transient* — how fast a regime is re-acquired. It does
not measure what survived the absence. `fable/retention.py` measures that from
the same raw records: loss in the first 16 chars after a regime boundary, when
at most one optimizer step has landed inside the new block.

| | Retention BPC (cycled) | A-only | **Penalty** | Slope/visit |
|---|---:|---:|---:|---:|
| creature | 7.5884 | 2.1905 | **+5.3979** | −1.146 |
| gru | 3.7097 | 2.0596 | **+1.6502** | −0.416 |

**The creature forgets catastrophically — 3.3× the GRU's retention penalty.**
And 7.59 BPC is *above* the 6.02 chance level for this 65-symbol vocabulary:
at a regime boundary the creature is not uncertain, it is confidently
predicting under the wrong symbol map. The GRU degrades to 3.71, badly but
below chance — it hedges.

This **explains and invalidates** the savings result below. The creature's
"steeper savings slope" is a mechanical consequence of having vastly more to
recover: it recovers faster because it lost more. The favourable-looking
number was measuring the damage, not the repair.

## Verdict: **FAIL**, by the clause written to catch exactly this

The preregistration's fail condition included: *"or creature wins adaptability
only by violating the compression floor."* That is precisely what happened.

The creature's savings slope is genuinely ~2× the GRU's after the
ongoing-learning control — but it **starts 3.4× worse** (+1.161 vs +0.344 per
visit) and never catches up. In relative terms the ordering reverses: the GRU
cuts its adaptation cost by 37.6% across visits, the creature by 19.3%. A
steeper absolute decline from a much worse starting point is not evidence of
better adaptability; it is largely the extra headroom of being worse at the
task, which is the confound the compression proviso exists to detect. With
interference also going (slightly) the GRU's way, there is no axis on which
the creature is the more adaptable learner here.

**Recorded as a negative result, not spun.**

## The most important secondary finding

**The creature's F0 parity depended on the annealed schedule.**

- F0 (warmup + cosine to 3e-4): creature − GRU = **+0.009** (parity)
- F2 (constant 1e-3 after warmup): creature − GRU = **+0.170**

Same geometry, same corpus, same code. The GRU is far more robust to losing
the schedule than the creature is. This is uncomfortable for the project's
framing, because F2's constant LR was chosen on the argument that *an
annealed learner is structurally committed to a stationary world* — and a
creature meant to run continuously cannot rely on a decay schedule it will
outlive. So the S0 parity result holds only in the regime the organism is
least entitled to use.

Caveats on the comparison, stated rather than buried: F2 also used batch 4 vs
F0's batch 12, so schedule is not the only difference, and the two runs are
not a controlled A/B. Isolating it needs one arm — F0 geometry, F0 batch,
constant LR — which is cheap and should be run before this finding is leaned
on.

## What this does and does not license

- It does **not** falsify the substrate. It falsifies "this substrate, as
  currently configured, is a more adaptable learner than a matched GRU."
- It sharpens the diagnosis already coming from F0: the field is functional
  but **undifferentiated** (freeze ~0.5 BPC, shuffle ~0.05). A mean-field
  reservoir has no obvious mechanism for regime-specific specialization —
  which is exactly what savings would require. F2's negative is what one
  should expect from an undifferentiated field, and it raises the value of
  F5 (informed structure → differentiation) rather than lowering it.
- Re-run F2 against any configuration that moves the shuffle delta. That is
  the natural pairing: **differentiation is the hypothesis, adaptability is
  the payoff test.**

---

## Round 2 on the expression config (preregistered 2026-07-31, overnight)

The original verdict committed to this: "re-run F2 against any configuration
that moves the shuffle delta." F8 seed 7 moved it: shuffle-internal +0.182
(vs 0.036–0.066 incumbent) at unchanged BPC, with per-cell expression norms
spanning 0.0–5.7. Differentiation is the hypothesis, adaptability is the
payoff test — so: cycled + A-only expression-creature arms, seeds 7/13/21,
identical F2 protocol (constant 1e-3, B=4, block 24,576, 8k updates). GRU
arms are reused from round 1 (identical protocol). Same metrics including
zero-shot retention; same fail clauses. The question: does parametric cell
identity reduce the catastrophic-forgetting penalty (+5.40 vs GRU +1.65)
that a mean field could not protect against?

Caveat carried in: expression is unstable on seed 13 in the annealed regime
(F8 amendment 1 fix was insufficient there); F2's regime is constant 1e-3,
which every configuration has survived so far. Skips are reported per arm as
always.

---

# RESULTS, round 2 — expression config (2026-07-31 morning)

| Metric | Mean-field creature (r1) | Expression creature (r2) | GRU |
|---|---:|---:|---:|
| A-only steady BPC | 2.2172 | **2.1291** | 2.0469 |
| Regime-attributable savings /visit | −0.1471 | −0.1121 | −0.0770 |
| Zero-shot retention (cycled) | 7.5884 | 7.1814 | 3.7097 |
| **Retention penalty** | **+5.3979** | **+5.0523** | **+1.6502** |

Expression improved constant-LR compression by ~0.09 BPC and trimmed the
retention penalty by ~0.35 — but the creature still forgets catastrophically
(7.18 at a regime boundary remains *above* the 6.02 chance level; the
penalty is still 3.1× the GRU's). **Verdict: the original FAIL stands.**
Differentiation at the +0.08–0.18 shuffle level does not protect memory.
Either much stronger differentiation, explicit consolidation machinery, or
both. All r2 arms completed (constant 1e-3 remains the survivable regime).
