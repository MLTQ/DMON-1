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
