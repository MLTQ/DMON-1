# F3: liquid cell rule head-to-head (preregistered 2026-07-30, finalized same night)

*Drafted after the LNN discussion; finalized and launched after F7/E1
promoted stability-by-construction from interesting to necessary: the GRU
rule died in five out of five attempts at sustained high LR (E1 3/3 at
constant 3e-3, F7 2/2 at 24k-horizon cosine), and GradGuard proved a
pathological parameter region cannot be crossed by refusing to step.*

## Question

Does replacing the shared GRUCell with a **liquid cell** — leaky integration
toward a bounded target with learned, bounded, input-dependent time
constants (`fable/liquid.py`) — buy the organism the stability the guard
stack cannot, without paying more than 0.05 BPC of capability?

The cell: `h' = (1−a)h + a·tanh(target)`, `a = 1/τ`, τ input-dependent and
bounded in [1, 100]. State is bounded by construction and the recurrent
backbone is contractive — verified in smoke: rule cells recover from |h|=5
abuse to <1.0 within 40 tokens. (Honesty: the W-paths through target and τ
can still grow, so gradient stability is *tested*, not proven — the arms
below include every regime that killed the incumbent.)

Parameter match: backbone width solved so the rule matches the GRU rule's
count — whole-organism 348,712 vs 348,609 (0.03%). Everything else
(graph, ports, mirror, readout, guard, schedule) identical.

## Arms (all liquid-creature; incumbents and baselines already exist at
identical protocol and are reused, not rerun)

| Arm | Regime | Seeds | Compares against |
|-----|--------|-------|------------------|
| L-A | F0 protocol (annealed 8k) | 7/13/21 | F0 creature 2.008/2.115/2.085; F0 GRU 2.029/2.085/2.069 |
| L-C | constant 3e-3, 8k | 7/13/21 | incumbent creature: died 3/3 (u6.6–7.9k); E1 GRU 2.25/2.41/2.43 |
| L-H | annealed 24k horizon | 7 | incumbent creature: died 2/2 (u8.8–9.6k); GRU 1.924; transformer 2.041 |

## Pass / fail (committed before launch)

- **Stability pass**: all three L-C arms and L-H run to completion with
  < 1% skipped updates. This is the claim the incumbent failed five times.
- **Capability pass**: L-A mean within +0.05 BPC of the incumbent creature
  (2.069 mean) — i.e. the stability is not bought with the task.
- **Full pass** = both. **Rejection** = L-A regression > 0.05 at matched
  budget (elegance does not vote), or L-C/L-H deaths at incumbent-like
  locations (no stability gain — the instability lives outside the cell
  rule, which would be its own important result).
- **Timescale read** (descriptive, reported either way): the trajectory of
  `alpha_mean` and the end-state spread of per-cell τ — did cells
  differentiate in time, where F0/F1 showed they never differentiate in
  space? A τ spread collapsing to one value means the mechanism was not
  used; a wide spread is the first differentiation of any kind observed in
  this substrate.
- Ablations (reset/shuffle/mirror) recorded as always.

## Launch

```bash
bash fable/run_f3.sh   # waits for E1b to drain the 4090, then 6 arms parallel + L-H
```

---

# RESULTS (2026-07-31, overnight): **REJECTED — on both preregistered axes**

| Arm | Regime | Outcome |
|---|---|---|
| la_s7/s13/s21 | annealed 8k | 2.1279 / 2.1879 / 2.2316 — all stable, 0 skips |
| lc_s7 | constant 3e-3 | **died u6058** |
| lc_s13 | constant 3e-3 | **died u5753** |
| lc_s21 | constant 3e-3 | "survived" with 2,245 skips (28%), bpc 3.51 — degenerate |
| lh_s7 | annealed 24k | **died u19040** (66% cumulative skips) |

- **Capability: FAIL.** L-A mean 2.1825 vs incumbent 2.0695 → **+0.113**,
  beyond the +0.05 rejection bar. Elegance does not vote.
- **Stability: FAIL.** The regimes that killed the GRU rule killed the
  liquid rule too — at essentially the same place under constant 3e-3
  (u5.8–6.1k vs u6.6–7.9k). L-H lasted 2× the incumbent's horizon before
  dying, but degenerately (LR floored, majority chunks skipped).
- **Timescale read: the mechanism went unused.** Mean α stayed ~0.07
  (τ ≈ 14) with almost no spread on every arm — cells did not differentiate
  temporally. Consistent with the F8 diagnosis: τ is computed by *shared*
  weights, so pre-expression cells had nothing to differentiate with.
  (L-A's shuffle deltas, 0.069–0.102, are mildly above the incumbent band —
  a whisper of what F8's expression made loud.)

## What the rejection bought

The **cliff is rule-independent.** Two different cell rules — one
contractive by construction — die in the same regimes at similar locations.
The instability's source is therefore in the *shared machinery* (attention
q/k/v growth, embedding, readout — the unbounded-scale paths into the
recurrence, which is also what every implementation-level blowup tonight
was). This exonerates the recurrence and gives the stability hunt a much
smaller haystack. Bounded state slows the pathology (L-H's 2× horizon) but
cannot stop what enters through the weights.

F3 is closed as a rejection with one carried question: whether liquid τ
*plus* per-cell expression (F8's mechanism, which is what τ lacked) behaves
differently — parked pending F8b, not assumed.
