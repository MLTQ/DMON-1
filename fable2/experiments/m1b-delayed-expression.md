# M1b: delayed expression — can pressure teach persistence to this anatomy?

Status: preregistered 2026-08-07, before any M1b optimizer update. The control
arm for the slow-tissue proposal: unchanged anatomy, persistence itself under
gradient pressure.

## Question

M1a measured the memory horizon: one FIFO span, no recurrent carry, decay
proportional to evicted demonstration tokens. M0's training never *asked* for
persistence — exposure and query were always adjacent, so no gradient ever
favored a non-tape route. If delays between exposure and query carry gradient,
can the existing recurrent anatomy learn to hold a mode past full FIFO
eviction?

## Prediction (preregistered)

**Fail, informatively.** The contraction findings (memory_lesion ≡ no_exposure
in four experiments; M1a's zero residual; alpha floor 0.02 ≈ 17-token time
constants; gradients through the contractive tissue vanish over exactly the
spans that matter) say the route does not exist to be found. This run exists
because an M1c slow-tissue success is unattributable without it — and because
the program has been surprised before. A pass here would falsify the
contraction reading and remove the need for new anatomy.

## Treatment (one change from M0)

Every training episode inserts N filler tokens (the fixed M1a English filler,
memory writes on, identical across arms) between exposure and query, N sampled
uniformly from {0, 64, 128, 192, 256, 320, 384} per update. Objective, arms,
organism, graft, optimizer, instruments: identical to M0. 300 updates, eval
every 50 (heavier eval), per-eval checkpoints.

## Evaluation

Each eval scores the arm battery at delay 0 (instrument continuity with M0 —
curves/trends read this) and the paired differential at delays {128, 256, 384}
(heldout, 32 direction-episodes each).

## Pass gates (heldout, read point, replication rule applies)

1. Differential at N=256 and N=384 (full eviction) positive in mean with
   strict-win majority (≥ 17/32) — any reliable post-eviction differential
   certifies a learned non-tape route.
2. Delay-0 differential must not fall below half its M0 value (+0.042) — the
   anatomy may not buy persistence by unlearning acquisition.
3. Standard battery at delay 0: floor/no-exposure/lesion margins positive;
   anti-sabotage clear.
4. Replication of any post-eviction differential on fresh demonstration
   samples.

## Stop rules

- A fail (predicted): stop at 300; no duration/scale moves; proceed to M1c
  (slow tissue, preregistered separately) with this run as the control arm.
- A pass: M1c is cancelled as unnecessary; proceed to delayed-retention
  characterization and mode switching (M2) on this anatomy.
- Stability incidents (the contraction is also the stability story; delayed
  BPTT is the longest credit path this substrate has carried): gradient
  rejection or non-finite telemetry ends the run loudly, recorded.

## Result

Pending.
