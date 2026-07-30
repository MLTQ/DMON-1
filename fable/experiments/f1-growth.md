# F1: runtime growth on a substrate where capacity pays (preregistered 2026-07-30)

> **STATUS: COMPLETE (2026-07-30) — results at end of file.** Briefly blocked earlier the same day on
> a misreading of F0's shuffle result, then unblocked by measurement.
>
> The block claimed internal tissue was near-inert, citing shuffle-internal
> deltas of +0.036..+0.066. The freeze probe (`fable/probe.py`) on the same
> checkpoints then measured **+0.280 / +0.831 / +0.409** — internal tissue is
> worth ~0.5 BPC and is merely *permutation-invariant*. Shuffle measures
> differentiation, freeze measures work; the block conflated them. F1's arms
> differ by 16 vs 64 internal cells — a 4× difference in a population worth
> half a bit, well above F0's ~0.107 BPC seed spread — so the experiment is
> falsifiable and proceeds.
>
> **Added hypothesis (from the same finding).** An undifferentiated
> population should scale **sublinearly**: more interchangeable cells average
> rather than specialize. The preregistered prediction is therefore
> *diminishing returns* — `born` should beat `small` by clearly less than 4×
> the per-cell contribution implies. A strongly sublinear result is the
> quantitative argument for differentiation machinery (cell types, informed
> structure — F5) over simply adding tissue, and would be the most useful
> outcome this experiment can produce. `dmon/stream` could not run this test
> because its tissue did no work in the first place.
>
> **Added measurement.** Every arm is probed with `fable.probe` at the end:
> freeze-internal (did the tissue do work) and shuffle-internal (did it
> differentiate). For `grown`, the existing recruitment probe additionally
> freezes *only the added cells*, at the graft and at the end.

*Original preregistration follows, unchanged. Results at the end of the file.*

## Question

`dmon/stream` showed growth is mechanically cheap (transition cost ≈ 0) but its
lattice capacity axis was inert, making S2 unfalsifiable there. grok showed the
connectome's capacity axis is live (64→128 cells+width shrinks the GRU gap). Can a
creature **grow into** the large configuration at runtime — recruiting the new tissue
— rather than being born large?

Secondary hypothesis: chase-1's 192-cell arm diverged *from scratch*. If growth
reaches large-field performance stably where born-large diverges, progressive growth
is a stability mechanism, not just a capacity one.

## Protocol

Three arms, identical stream, identical total updates (8000), seeds 7/13/21. Ports
are fixed at 16 in / 16 out / 32 mirror for every arm, so the smallest field with
free tissue is 80 cells (16 internal):

| Arm | Geometry |
|-----|----------|
| `small` | 80 cells fixed (16 internal), h=128 |
| `grown` | 80 cells → 128 at update 2000 (+48 internal) |
| `born` | 128 cells from update 0 (this is F0's creature arm, copied in) |

Growth event (`fable/grow.py`): each new cell samples K dendrites from the whole
enlarged non-output pool; each old mutable cell donates its weakest dendrite slot
(never an output cell's forced sensory slot 0) to a new cell round-robin with the
edge logit reset to 0 — without donated slots, nothing would ever read new tissue
and it would be inert *by construction*. The graft is output-silent at t=0 because
new cells start at h=0, so their value-projected messages are zero until their
state develops. The readout, embedding, and shared rule are untouched; parameters
change by exactly `n_new × K` edge logits. Adam moments are preserved for every
surviving parameter and for the surviving slice of the resized logit tensor, so
the graft is not a disguised optimizer restart.

Measured:
- Held-out BPC every 500, all arms, full curves in the report
- Transition cost: BPC(eval straddling u2000) − BPC(eval before), `grown` vs `small`
- **Recruitment probe**: ablate (zero + freeze) the 48 added cells at u2500 and at
  u8000. Recruited means: ablation at u8000 hurts ≥ 0.05 BPC; at u2500 it hurts less
  than half that. If final-ablation cost ≈ 0, any `grown` win came from the donated
  slots' logit reset or elsewhere, not the tissue — check the operators, not the
  masks.
- State health (same monitor as F0) through the graft

## Pass / fail

- **Pass (S2 bar)**: `grown` final BPC beats `small` by ≥ 0.05 mean across seeds;
  transition cost ≤ +0.05 BPC; recruitment probe passes as defined above.
- **Strong pass**: additionally `grown` final within 0.03 of `born`.
- **Stability bonus** (reported either way): if a `born`-large arm diverges at any
  seed while the seed-matched `grown` arm does not, record progressive growth as a
  stabilizer.
- **Fail**: `grown` ≈ `small` (added tissue inert — the dmon/stream failure repeated
  on the substrate that was supposed to fix it, which would be a serious blow to the
  whole growth thesis), or transition cost > +0.2 (graft destabilizes), or
  recruitment probe fails (win is bookkeeping, not biology).

## Launch

```bash
# on Aine, from repo root, after F0
bash fable/run_f1.sh
python3 -m fable.summarize --root fable/runs/f1 --out fable/runs/f1/LADDER.md
```

---

# RESULTS (2026-07-30, 6 run arms + 3 copied born arms, zero skipped updates)

Final held-out BPC:

| Seed | small (80c) | grown (80→128) | born (128c) |
|-----:|------------:|---------------:|------------:|
| 7 | 2.0331 | 2.0007 | 2.0082 |
| 13 | 2.0854 | 2.0826 | 2.1154 |
| 21 | 2.1328 | 2.1240 | 2.0850 |
| **mean** | **2.0838** | **2.0691** | **2.0695** |

Graft mechanics (grown, u2000→u2500, small as shared-trend control):
transition excess −0.003 / +0.055 / +0.007 — mean **+0.020**, well under the
+0.05 bar. Recruitment probe (freeze the 48 added cells): final Δ +0.047 /
+0.040 / +0.017 (mean **+0.035**); just-after-graft Δ +0.039 / +0.035 /
−0.002.

Tissue probes, every arm: freeze-internal +0.14…+0.41 (the tissue works, at
every size); shuffle-internal +0.015…+0.030 (undifferentiated, at every
size).

## Verdict against the preregistered bars

| Bar | Result |
|---|---|
| Pass: grown beats small by ≥ 0.05 mean | **not met** (−0.015) |
| Strong pass: grown within 0.03 of born | met (−0.0004 — grown ≈ born) |
| Transition cost ≤ +0.05 | **met** (+0.020 mean) |
| Recruitment: final ≥ 0.05, early < half of final | **not met** (+0.035 final; early ≈ final) |
| Fail: graft destabilizes | did not occur |

**S2's capability claim: FAIL. S2's mechanism claim: PASS.** And the amended
prediction — sublinear scaling — confirmed about as decisively as it can be:

1. **Growth is mechanically safe and complete.** Transition cost ≈ 0; the
   grown organism is statistically indistinguishable from born-large by the
   end (mean gap −0.0004). Grafting mid-stream costs nothing and forfeits
   nothing. The machinery works.
2. **The capacity it adds is nearly worthless at this task and budget.** 4×
   the internal tissue buys 0.014 BPC however it is acquired (born−small =
   −0.014, grown−small = −0.015). The added cells' contribution is small and
   *flat from the moment of the graft* (+0.035 ≈ +0.024 early) — extra
   interchangeable mean-field capacity, absorbed instantly, never
   differentiated. Shuffle stays at ~0.02 in every arm at every size.
3. This is the connectome's version of `dmon/stream`'s S2 result, now
   properly attributable: there, growth was safe but the tissue did no work
   (freeze would have shown ~0); here the tissue works (freeze +0.14…+0.41)
   but does not *specialize*, so more of it averages instead of adding. The
   binding constraint is **differentiation, not capacity** — the F5 thesis,
   now with its quantitative motivation, and consistent with F0's flat
   utilization trajectory and Max's diagnosis that the task never forces
   information into the bulk (F6, running).

Caveat that travels with all of this: F7 has since shown every 8k-update
number is budget-bound (curves still falling at the horizon). If F7's
long-horizon run surfaces late differentiation, point 2's "never" softens to
"not within 8k updates."
