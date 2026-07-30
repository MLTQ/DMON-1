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
