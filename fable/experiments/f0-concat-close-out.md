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
