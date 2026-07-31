# E1: schedule isolation — does F0's parity survive constant LR? (preregistered 2026-07-30)

## Question

F2 found creature−GRU = +0.170 under constant LR where F0 found +0.009 under
warmup+cosine — but F2 also ran batch 4 vs F0's batch 12, so schedule and
token budget were confounded. This run isolates the schedule: F0 geometry,
F0 batch, F0 updates, **constant 3e-3 after warmup** (lr_min = lr), creature
and matched GRU, seeds 7/13/21.

A continually-running organism outlives any annealing schedule, so the
constant-LR gap is arguably the *more* honest S0 number. If the +0.17 gap
reproduces here, F0's parity is a property of the annealed regime, and the
qualification gets attached to the S0 verdict permanently.

Secondary purpose: first controlled test of `GradGuard` at sustained peak LR
— the exact regime that killed chase-1 seed 13 (u~5.2k, grok code) and F7's
unguarded creature (u~9.6k). The guard's skip counts and lr_scale trajectory
are part of the result. The GRU is expected never to trip it.

## Protocol

128 cells, h=128, K=12, ports 16/16/32, spt=4, B=12, T=32, 8,000 updates,
lr = lr_min = 3e-3, warmup 200, eval every 500. Runs on the 4090 *alongside*
the F7 guarded creature (both are launch-bound; wall-clock contention is
disclosed and harmless — same-device within every compared pair).

## Read

- **Gap reproduces (≥ ~0.10)**: F0's parity is annealing-dependent. S0
  verdict gets the permanent qualification; closing the constant-LR gap
  becomes a named problem (homeostatic optimization, F3 liquid dynamics, or
  schedule-free methods).
- **Gap collapses (≤ 0.05)**: F2's gap was mostly the token budget, not the
  schedule — the F2 write-up gets corrected, and the "creature leans on
  annealing" claim is withdrawn to exactly its F7-stability form.
- Any creature blowup the guard cannot hold to completion is itself a
  result: constant 3e-3 is outside the substrate's stable envelope even with
  homeostatic backoff.

---

# RESULTS (2026-07-30, overnight)

**All three creature arms died before completing 8,000 updates**: fail-fast
at u6564 (s7), u7428 (s13), u7923 (s21) — 200 consecutive pathological
chunks each, the same frozen-weights-still-produce-huge-gradients signature
as F7. Preregistered read #3 applies: **constant 3e-3 is outside the
substrate's stable envelope even with homeostatic backoff.** The envelope
did widen versus grok's stack (grok died at u5.2k in this regime; fable
reached 6.6–7.9k — the clamp/decay-structure/guard changes bought ~1.5–2.7k
updates), but the cliff is not crossable by guarding alone.

The GRUs completed untroubled (0 skips) but degraded: 2.2471 / 2.4053 /
2.4348 vs 2.03–2.08 annealed. So constant 3e-3 is a bad regime for
*everyone* — the GRU pays ~0.2–0.35 BPC, the creature pays its life.

E1's primary question (does F0's parity survive without annealing?) is
therefore **unanswerable at 3e-3** — there is no creature number to compare.
What was answered instead is stronger on the stability axis: the creature
does not merely *lean* on annealing, at peak LR it **requires annealing to
survive**.

## E1b addendum (preregistered before launch, same night)

The isolation question moves to the survivable LR: **constant 1e-3** — the
regime F2 used (where the creature trained fine at B=4) — now at F0's B=12,
8,000 updates, creature + matched GRU, seeds 7/13/21. Reads, committed in
advance:

- Gap ≈ +0.17 reproduces → F2's schedule-dependence finding stands,
  unconfounded; the S0 qualification becomes permanent.
- Gap ≤ 0.05 → F2's gap was mostly its B=4 token budget; the F2 write-up
  gets corrected and "parity depends on annealing" is withdrawn to its
  stability-only form (which E1 proper just proved on its own).
- In between → both effects real; report the split.
