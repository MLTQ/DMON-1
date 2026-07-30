# Chase #1: scale + no_credit + long horizon (2026-07-30)

## Question

Does the GRU gap keep shrinking with scale and more tokens when reverse credit is off?

## Protocol

- Tiny Shakespeare 90/10 contiguous split
- **no_credit**: reward_gain=0, output_error_credit_gain=0, fast_plasticity_gain=0
- Updates: **8000** (primary); side ablations may use 4000
- Eval every 500, log every 100
- Seeds: **7, 13, 21**
- Matched GRU on identical stream budget

## Arms

| Arm | Geometry | Readout | Seeds | Updates |
|-----|----------|---------|-------|---------|
| `s128_nc_s{seed}` | 128 cells, h=128, K=12, ports 16/16/32, spt=4, B=12 | mean | 7,13,21 | 8000 |
| `s192_nc_s7` | 192 cells, h=192, K=12, ports 24/24/48, spt=4, B=8 | mean | 7 | 8000 |
| `s128_nc_concat_s7` | same as s128 | **concat** | 7 | 4000 (side) |

## Pass

- Mean gap (3 seeds) at s128 **≤ 0.08 BPC**, trending down late in training, **or**
- Clear continued improvement vs 2k-update scale gap (~0.12 mean)

## Fail

- Gap floors ≥ 0.10 at 8k with no late slope → capacity/schedule not enough; rethink messaging

## Launch

```bash
cd ~/Code/DMON-1 && bash grok/run_chase1.sh
python3 -m grok.summarize_ladder --root grok/runs/s0_chase1 --out grok/runs/s0_chase1/LADDER.md
```

## Results (2026-07-30, run on Aine; synced back 2026-07-30)

| Arm | Seed | Readout | Updates | Creature BPC | GRU BPC | Gap | ResetΔ | ShuffleΔ |
|-----|-----:|---------|--------:|-------------:|--------:|----:|-------:|--------:|
| s128_nc_s7 | 7 | mean | 8000 | 2.3869 | 2.2575 | +0.1293 | +2.537 | +1.661 |
| s128_nc_s21 | 21 | mean | 8000 | 2.4025 | 2.2719 | +0.1306 | +2.513 | +1.919 |
| s128_nc_s13 | 13 | mean | 8000 | **nan** (blowup ~u5200) | 2.4046 | — | — | — |
| s192_nc_s7 | 7 | mean | 8000 | **nan** (diverged from ~u1000) | 2.8944 | — | — | — |
| s128_nc_concat_s7 | 7 | **concat** | 4000 | **2.4804** | 2.5079 | **−0.0276** | +2.736 | +2.824 |

### Verdict

1. **Fail branch hit for the primary arms**: mean-readout gap floors at ~+0.13 at 8k
   with no late slope. Capacity + schedule alone do not close the gap. As
   preregistered: rethink messaging — and the messaging bottleneck turned out to be
   the *readout*.
2. **Concat readout flips the sign**: the side ablation beat its parameter-matched GRU
   (657,345 vs 657,985 params) at half the update budget, with its holdout curve still
   descending steeply at stop (2.80 → 2.50 → 2.48 over the last 1k updates). Mean
   pooling over 16 output cells was discarding most of the readable state.
3. **Stability is now the blocker**: seed 13 blew up late (NaN at ~u5200, clip=1.0,
   lr=3e-3 constant) and s192 diverged almost immediately (holdout pinned ~6.0 from
   u1000). Bigger fields need normalization and/or an LR schedule, not just clipping.
4. **Next**: concat readout at full budget, multi-seed, with stability work. This is
   the `fable/` spike's S0 charter.
