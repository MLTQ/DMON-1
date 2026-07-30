# F7: horizon calibration — train to plateau (preregistered 2026-07-30)

## Question

F0's loss curves are still falling hard at the 8,000-update horizon: mean
held-out Δ over the final 1,500 updates is creature **−0.108**, GRU −0.087,
transformer **−0.175** — with creature seed 7 dropping −0.259 in that window
*despite the LR having annealed to 3e-4*. The 8k budget was inherited from
chase-1, not chosen by convergence (Max's undertraining concern, confirmed by
measurement). Every F0/F1 conclusion is therefore a statement about
budget-bound models. Where do the curves — and the ordering — settle?

Secondary question the transformer's slope raises: its Δlate is the steepest
of the three, consistent with the standard scaling intuition that attention
wins with budget. Does the F0 ordering (creature ≈ GRU > transformer) hold,
tighten, or invert at 3× the horizon?

## Design

One seed (7 — calibration, not a multi-seed claim), 24,000 updates, cosine
horizon stretched to match (same peak 3e-3, same floor 3e-4, warmup 200).
Arms: creature, matched GRU, matched transformer — F0 geometry, F0 batch,
identical stream discipline. Evals every 1,000.

- **Read**: final BPCs, the update at which each curve's 1,500-update slope
  first flattens under 0.01, and the creature−GRU gap trajectory over
  training.
- The shuffle/mirror ablation trajectory is recorded as always — if
  differentiation appears late (shuffle delta rising after u8000), the flat
  trajectory seen in F0 was a budget artifact and the F6/F5 framing needs
  revisiting. F0's data says this is unlikely; recording it either way.
- Multi-seed confirmation only if the ordering *changes* — a calibration run
  that reproduces F0's ordering at plateau needs no replication to justify
  keeping 8k-with-disclosure or moving the default budget.

## Launch

```bash
bash fable/run_f7.sh   # waits for the 4090 to free (F1), then runs 3 arms
```
