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
