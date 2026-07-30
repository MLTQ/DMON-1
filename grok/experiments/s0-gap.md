# S0 gap ladder (2026-07-29)

## Question

Which fixed-topology lever closes the ~0.19 BPC gap vs a parameter-matched GRU?

## Protocol

- Data: Tiny Shakespeare, contiguous 90/10 train/holdout
- Updates: 2000 (primary); extend winners later
- Batch 16 × chunk 32 (scale uses batch 12); AdamW 3e-3; eval every 250
- Holdout: 2048 tokens, 256 warmup
- Seeds: 7 primary; 13 for baseline + scale
- Hardware: Aine — scale/deep on 4090 (`cuda:0`), baseline/no_credit on 2070S (`cuda:1`)

## Arms

| Arm | Change vs rebuilt default |
|-----|---------------------------|
| `baseline_s7/s13` | 64 cells, h=64, K=8, spt=3, credit+fast on |
| `no_credit_s7` | reward=0, reverse credit=0, fast=0 |
| `scale_s7/s13` | 128 cells, h=128, K=12, ports 16/16/32, spt=4 |
| `deep_s7` | 64 cells, h=64, K=12, spt=6 |

Each arm: creature + matched GRU.

## Launch

```bash
# on Aine
cd ~/Code/DMON-1 && bash grok/run_ladder.sh
# results
python3 -m grok.summarize_ladder
cat grok/runs/s0_gap/LADDER.md
```

## Pass / fail

See prior plan: gap ≤ 0.05 multi-seed, or clear lever that shrinks gap with real effect size.

## Results (2026-07-29, 2000 updates)

| Arm | Seed | Cells | H | Creature BPC | GRU BPC | Gap |
|-----|-----:|------:|--:|-------------:|--------:|----:|
| baseline | 7 | 64 | 64 | 2.631 | 2.436 | **+0.195** |
| baseline | 13 | 64 | 64 | 2.617 | 2.432 | **+0.185** |
| no_credit | 7 | 64 | 64 | 2.609 | 2.436 | **+0.173** |
| deep (spt=6,K=12) | 7 | 64 | 64 | 2.675 | 2.430 | **+0.245** |
| scale | 7 | 128 | 128 | 2.592 | 2.507 | **+0.085** |
| scale | 13 | 128 | 128 | 2.621 | 2.474 | **+0.148** |

Mean baseline gap ≈ **+0.190**. Mean scale gap ≈ **+0.116**.

### Verdict

1. **Scale is the only lever that clearly shrinks the GRU gap** (≈0.19 → ≈0.12 mean). Seed variance is real (0.085 vs 0.148).
2. **Credit off slightly helps** at 64×64 (+0.173 vs +0.195) — reverse credit/fast path is mild tax, not free win.
3. **Deeper recurrence hurts** (+0.245) — do not increase spt alone.
4. **S0 not passed** — still >0.05 BPC behind matched GRU.
5. **Next:** scale + no_credit (and longer updates / larger fields), not more depth or credit machinery.

Full table: `grok/runs/s0_gap/LADDER.md`.
