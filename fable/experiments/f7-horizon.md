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

---

# RESULTS (2026-07-30, seed 7, 24,000 updates)

| Arm | 8k BPC (F0) | 24k BPC | Δ |
|---|---:|---:|---:|
| matched GRU | 2.0294 | **1.9241** | −0.105 |
| matched transformer | 2.3206 | **2.0411** | −0.280 |
| creature | 2.0082 | **did not finish** — twice | — |

Creature attempt 1 (pre-guard): gradient norms reached 1.5e19 *while finite*
at u8800 (lr ≈ 2.2e-3); clipped-garbage steps degraded the model from ~2.3
to chance before non-finite gradients triggered the abort at u9622.
Creature attempt 2 (GradGuard): fail-fast at u8942 after 200 consecutive
pathological chunks. The instructive detail: during a skip streak the
weights are frozen, and every fresh data chunk from those frozen weights
still produced >1e4 gradients — the *parameter region itself* was
pathological, and no-stepping cannot leave it. Homeostatic backoff only
matters if a step is ever taken.

## Verdict

1. **The budget was binding for everyone, and unevenly.** The GRU banked
   another 0.105; the transformer banked 0.280 and nearly caught the GRU's
   8k number — the attention-wins-with-budget crossover is approaching
   exactly as predicted.
2. **The creature cannot currently consume 3× budget at all.** Under this
   schedule shape it exits its stable envelope at u≈8.8–9.6k, robustly
   (two mechanisms, same location; chase-1 s13 died at u5.2k under constant
   3e-3 in grok's code). F0's 8k survival was annealing outrunning the
   cliff, not the absence of one.
3. Consequence for the scale question: the competitors convert budget into
   capability and the creature converts it into a crash. Until the substrate
   is stable at sustained LR, every scaling argument for the creature is
   moot. This promotes **F3 (liquid cell — contractive dynamics by
   construction)** from interesting to primary-path candidate, alongside a
   harness-level alternative: **rollback-on-streak** (periodic last-good
   snapshots, restore on pathological streaks — sol S5's reversible
   probation applied to optimization). GradGuard remains as the fail-fast
   backstop it proved to be.
4. Shuffle-delta trajectory up to the crash: still flat (no late
   differentiation before u8.8k). The F0-era conclusion stands as far as it
   could be tested.

E1 (running) doubles as the constant-3e-3 envelope test at 8k horizon:
grok's code died at u5.2k in that regime; if fable's creatures cross 8k,
the clamp+decay+guard stack widened the envelope even though it cannot
cross the ~9k cliff.
