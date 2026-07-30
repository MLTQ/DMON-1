# F0: S0 close-out — concat readout at full budget (preregistered 2026-07-30)

## Question

Chase-1's concat-readout side arm beat its parameter-matched GRU (−0.028 BPC) at half
budget with the holdout curve still descending. Does that win survive full budget,
three seeds, stability machinery, and a schedule-matched baseline?

## Protocol

- Data: Tiny Shakespeare, contiguous 90/10 train/holdout (identical to grok runs)
- Geometry: 128 cells, h=128, K=12 dendrites, ports 16 in / 16 out / 32 mirror,
  steps-per-token 4, batch 12 lanes × chunk 32
- Readout: **concat** over output cells
- Credit machinery: none (deleted, not flagged off)
- Stability: RMSNorm on messages + pre-readout, grad clip 1.0, LR 3e-3 with 200-update
  linear warmup and cosine decay to 3e-4 at update 8000
- **The GRU gets the identical LR schedule** (sol never did this; fable always does)
- Updates: 8000; eval every 500 (holdout 2048 tokens, 256 warmup); log every 100 with
  state-health metrics (max |h|, mean message RMS, logit scale)
- Seeds: 7, 13, 21 — creature and matched GRU per seed
- Head-only bypass control: organism frozen at init, readout trained alone, seed 7,
  same schedule and budget
- Hardware: Aine — creature arms on 4090 (`cuda:0`), GRU + bypass on 2070S (`cuda:1`);
  mixed-device pairs disclosed in the report as always

## Pass / fail

- **Pass**: mean gap (creature − GRU, final held-out BPC, 3 seeds) ≤ 0.00; no NaNs;
  reset and shuffle deltas each > +1.0 on every seed; head-only control at least
  0.3 BPC worse than the full creature.
- **Soft pass**: mean gap ≤ 0.05 (S0's literal bar) with all guards above intact.
- **Fail**: gap > 0.05, or any NaN, or ablation deltas collapse, or the head-only
  control lands within 0.3 BPC of the creature (win belongs to the head, not the
  substrate).

## Interpretation commitments

- If pass: S0 is closed. F1 (growth) launches on this exact configuration.
- If the concat win vanishes at 8k while mean-readout numbers reproduce chase-1, the
  chase-1 concat result was an undertrained-GRU artifact (its GRU had only 4k
  updates); record that and fall back to readout search (attention readout, per-cell
  linear + learned mixing) before touching anything else.
- If NaNs persist despite normalization + decay: stability work is not done; do not
  buy it back by lowering LR alone (that trades the gap for stability and confounds
  the readout question).

## Launch

```bash
# on Aine, from repo root
bash fable/run_f0.sh
python3 -m fable.summarize --root fable/runs/f0 --out fable/runs/f0/LADDER.md
```

## Amendment 1 (2026-07-30): stability fix + relaunch

The first launch was killed at ~u6800 (seed-7 creature). The "RMSNorm on
messages" stabilizer was itself the destabilizer: raw message RMS is ~0.05, so
full normalization amplified messages — and their gradients — ~20× per
micro-step, compounding across the 128 micro-steps of a chunk's backward.
Gradient norms grew 5 → 2.3e5 → 5.0e9 over 200 updates (~u800–1000), finite
clipped updates degraded the model from 3.08 to 5.0 BPC, then gradients went
permanently non-finite and the skip guard froze the run into a zombie.

Controlled A/B on Aine (same seed/geometry): pre-fix code reproduces the
blowup in the same window on the other GPU; fixed code passes the death point
with zero skipped updates and h_max exactly 1.00.

Protocol changes for the relaunch, before any creature result existed:
1. `MessageClamp` (scale down to unit RMS only when above it; identity below)
   replaces RMSNorm — a guard that only ever damps.
2. Embeddings moved under weight decay (they are written raw into the mirror
   ring; unbounded embedding growth is unbounded state growth — grok/sol both
   decayed them; h_max had drifted to 1.35).
3. Fail-fast: 200 consecutive non-finite-gradient updates abort the run.

The already-completed schedule-matched GRU controls (2.051/…) and bypass are
unaffected and retained. Notable before any creature verdict: giving the GRU
the warmup+cosine schedule improved it ~0.2 BPC over chase-1's constant-LR
GRU (2.051 vs 2.258 at seed 7) — the honest bar is much higher than the one
chase-1's concat arm beat.

---

# RESULTS (2026-07-30, 8000 updates, 3 seeds, all arms 0 skipped updates)

| Seed | Creature | Matched GRU | Gap | ResetΔ | **ShufΔ** | MirrorΔ |
|-----:|---------:|------------:|----:|-------:|----------:|--------:|
| 7 | 2.0082 | 2.0294 | **−0.0211** | +6.752 | **+0.066** | +0.172 |
| 13 | 2.1154 | 2.0846 | +0.0308 | +3.913 | **+0.045** | +0.049 |
| 21 | 2.0850 | 2.0688 | +0.0162 | +4.598 | **+0.036** | +0.022 |

**Mean gap +0.0086 BPC.** Creature params 348,609 vs GRU 348,745 (0.04%).

Bypass control (seed 7, frozen substrate, readout trained alone): **3.6925**
— 1.68 BPC worse than the creature.

## Verdict: SOFT PASS on compression, **FAIL on the ablation guard**

Against the preregistered bars, item by item:

| Bar | Result |
|---|---|
| Hard pass: mean gap ≤ 0.00 | **not met** (+0.0086) |
| Soft pass: mean gap ≤ 0.05 | **met** |
| No NaNs | **met** — 0 skipped updates in all six arms |
| Reset Δ > +1.0 every seed | **met** (+3.9 to +6.8) |
| Shuffle Δ > +1.0 every seed | **FAILED** (+0.036 to +0.066) |
| Head-only ≥ 0.3 BPC worse | **met** by a wide margin (1.68) |

The preregistered fail clause "or ablation deltas collapse" is triggered.
Recording that plainly: **S0's literal bar is met and the milestone is not
cleanly passed.**

## What was learned

1. **The concat readout + stability + schedule package closed the gap.**
   grok's chase-1 mean-readout arms at the same budget: +0.129 mean gap.
   Fable: +0.009. Both sides also improved in absolute terms (creature
   2.39→2.01 at seed 7; GRU 2.26→2.03), so this is not an undertrained-GRU
   artifact — the alternative reading the preregistration required us to
   check. The creature is at parity with a parameter-matched, schedule-matched
   GRU on a continuous stream with online updates.
2. **The stability fix held at full budget**: zero non-finite gradients across
   six arms and 48,000 updates, h_max pinned at 1.00. Amendment 1's diagnosis
   was correct.
3. **The internal tissue is functional but undifferentiated** — see the
   correction below, which supersedes this section's first reading.
4. **The hand-set mirror ring is not earning its 32 cells** (≤0.17 BPC by
   both the zero-ablation and the freeze probe), which is weak evidence
   against hyperparameter timescales generally and for learned delays
   (F3/F5).

## Correction (same day): shuffle ≠ inert

The shuffle result was initially read as "internal tissue is near-inert" and
used to block F1. That inference was wrong, and the freeze probe
(`fable/probe.py`, run on these same checkpoints) shows why:

| Seed | Normal | Freeze internals | **Δ** | Freeze mirror | Δ | Shuffle internals | Δ |
|-----:|-------:|-----------------:|------:|--------------:|--:|------------------:|--:|
| 7 | 2.0082 | 2.2877 | **+0.280** | 2.1813 | +0.173 | — | +0.066 |
| 13 | 2.1154 | 2.9467 | **+0.831** | 2.1645 | +0.049 | — | +0.045 |
| 21 | 2.0850 | 2.4937 | **+0.409** | 2.1084 | +0.023 | — | +0.036 |

Internal tissue is worth **~0.5 BPC on average**. It does substantial work;
it is merely **permutation-invariant** — a mean-field population where *which*
cell holds which state carries almost nothing. Shuffle measures
differentiation, freeze measures work, and a single ablation cannot separate
them. Both now ship together in `fable/probe.py`, and the rule is recorded
there: "population P matters" cites freeze; "population P is
specialized" cites shuffle.

This does **not** rescue the preregistered guard — the shuffle bar was
+1.0 and the result is +0.05, so F0 still fails it. What changes is the
*interpretation*: the failure means undifferentiated tissue, not dead tissue,
and it is **not** a reproduction of `dmon/stream`'s inert-capacity failure
(which was specifically that *added* cells bought nothing — a claim about
scaling that F0 never tested).

## Consequences

- **F1 (growth) proceeds.** Its arms differ by 16 vs 64 internal cells, a 4×
  difference in a population worth ~0.5 BPC, so the effect is well above the
  ~0.107 BPC seed spread. F1 was briefly blocked on the wrong reading; the
  block is withdrawn. It also now carries the freeze probe, so "were the added
  cells recruited" is measured directly rather than inferred.
- **F1 gains a sharper hypothesis.** An undifferentiated population should
  scale **sublinearly** — adding more interchangeable cells averages rather
  than specializes. So F1's likely outcome is diminishing returns, and *that*
  is the argument for differentiation machinery (cell types, informed
  structure) rather than more tissue. F1 now tests the capacity axis on
  tissue known to be functional, which is exactly the test `dmon/stream`
  could not run.
- **F5's metric language is corrected**: its primary claim is about
  *differentiation* (shuffle), with freeze as the work check. See
  `f5-informed-growth.md`.
- **F2 is unaffected** and continues.

## Amendment 2 (2026-07-30): the transformer control, and a bug it exposed

`PROJECT.md`'s S0 pass condition names a parameter-matched GRU **and
transformer**; only the GRU had been run. Adding it exposed an error in the
first attempt worth recording, because it is the error this project has
logged three times.

The transformer was trained on 32-token chunks (matching the creature's
chunk) but evaluated with a growing context window up to `max_len=512`.
Position embeddings past index 32 therefore never received a gradient, and it
scored **4.7885** — near-useless. *The training distribution must contain what
is evaluated.* Fixed by making the transformer's chunk **be** its context:
trained and evaluated at `transformer_seq_len=128`, with batch rescaled
(12×32 → 3×128) so tokens per update stay equal to the creature's.

Corrected result, seed 7: **2.3206** (vs 4.7885 broken).

| Arm (seed 7) | Params | Held-out BPC |
|---|---:|---:|
| creature | 348,609 | **2.0082** |
| matched GRU | 348,745 | 2.0294 |
| matched transformer | 352,145 | 2.3206 |

The creature beats the transformer by **0.31 BPC** and matches the GRU. This
reproduces sol's S0 ordering (transformer 2.600 > GRU 2.255 at matched budget)
and confirms the GRU was the right primary baseline: it is the harder one, and
beating only the transformer would have been the weaker claim. Seeds 13 and 21
are running.

The architectural caveat travels with the number: the transformer has no
persistent state, so its context is its window, while the creature and GRU
carry state indefinitely. This is matched *parameters and stream*, not matched
memory — which is the comparison S0 asks for, but it is not a claim that the
creature would win at long-context modelling.

## Process note

The first ladder rendered by `summarize.py` put the **bypass** arm's ablation
deltas in the creature's row (bypass was iterated last and overwrote them),
which would have reported seed 7 as ShufΔ +0.001 / MirrorΔ +4.517. Caught and
fixed before any verdict was recorded; ablation columns are now namespaced per
arm and the bypass control gets its own table. Filed under the same discipline
that produced "the log must not be able to lie."
