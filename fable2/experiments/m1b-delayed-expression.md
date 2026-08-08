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

## Amendment 1 (early stop; written at ~u48, before any evaluation existed)

A null does not need 300 updates; it needs a working treatment and a flat
signal across more than one round. At u100 (two eval rounds), stop and record
the null if BOTH hold:

- delay-0 differential ≥ half M0's value (+0.042) in both rounds — training
  demonstrably works; the absence of persistence is not the absence of
  learning;
- post-eviction differentials (N=256 and N=384) flat in both rounds: strict
  wins ≤ 17/32 and |mean| < 0.02.

If either fails (treatment broken, or any post-eviction signal), continue to
the original 300-update protocol. **Budget symmetry**: M1c inherits exactly the
budget M1b ends at, and the same early-stop rule — matched budgets are what
make the control comparison attributable.

## Amendment 2 (conditional redesign; written after u50, before u100)

u50 exposed a dilution confound: with 6/7 of episodes delayed and post-eviction
episodes unlearnable under the null, the acquisition signal is diluted ~7× and
the delay-0 differential is absent (+0.004/19 vs M0's +0.085/30 at u50) — the
run is failing its own treatment-validity gate, heading toward a null that says
nothing ("never acquired" masquerading as "did not persist").

Rule, recorded before u100: if at u100 the delay-0 differential remains below
+0.042, the run stops as **invalid-by-design (dilution), not null**, and M1b
relaunches redesigned — **warm-started from the M0 u300 checkpoint** (an
organism that already owns acquisition), same delay ladder, same objective,
fresh optimizer, 200 updates, eval every 50 with the same delayed battery and
validity gate (delay-0 differential must stay ≥ +0.042 — persistence may not be
bought by unlearning acquisition). The persistence question is unchanged; the
redesign only removes the dilution confound. M1c inherits the redesigned
protocol and budget.

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
