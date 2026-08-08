# M1c: slow tissue — the first new organ

Status: preregistered 2026-08-08, after M1b's confirmed null and before any
kernel change is exercised on GPU.

## Question

M1a measured zero recurrent carry; M1b showed 100 updates of direct pressure
cannot create any (the anatomy cannibalized acquisition instead — no route
exists to invest in). Does adding a small population of **slow cells** — the
first new anatomy since the Broca graft — give gradient descent a route, so
that an acquired mode survives full FIFO eviction?

## The anatomy

A sixth tissue type in the SOL2 kernel, placed after relay (appended last, so
every existing cell keeps its exact index — this is what makes checkpoint
grafting exact):

- `n_slow = 16` cells, shared hidden width, own `TissueRule` with
  **α ∈ [0.001, 0.05]** (time constants up to ~1000 microsteps vs the existing
  floor's ~50) and initial α = 0.02.
- Wired like internal tissue: slow cells sample dendrites from the full
  source pool (they can read memory and compute — the mode's route in);
  included in `internal_idx` (outputs may read them — the route out).
- The gating is the memory: α is input-computed as in every tissue, so the
  genome can learn spike-to-write / near-zero-to-hold. Slow cells double as
  constant-error carousels: at α → 0 their Jacobian approaches identity, fixing
  the vanishing-credit problem and the storage problem with one mechanism.
- `n_slow = 0` (the default) is contractually a bitwise no-op on the kernel —
  existing sol2 behavior, state-dict layout, and checkpoints are untouched.

## The graft

M1c warm-starts from the M0 u300 checkpoint into the augmented anatomy via an
explicit parameter graft (`fable2/slow.py`): every old tensor copies at its
original index with exact accounting (a copied-tensor census must match the
checkpoint census — no silent misses, no strict=False); slow rows are fresh;
one dormant dendrite slot per compute and output cell is rewired onto a slow
source at weak logit −3.0 (the growth machinery's own graft convention).
Deactivating exactly those slots must reproduce the ungrafted computation
bitwise (contract), and the run's validity gate doubles as the behavioral
check that the graft preserved acquisition.

## Protocol

Identical to M1b2, by design: same corpus, delay ladder {0..384}, objective,
eval battery with delayed differentials at {128, 256, 384}, validity gate
(delay-0 differential ≥ +0.042), 200 updates, eval every 50, per-eval
checkpoints, early-stop symmetry. The control arm is M1b2-at-u100 (resumable
to match budget if needed). One addition: a **slow-lesion arm** in every
delayed evaluation — the normal arm with slow tissue frozen at zero throughout
filler and query.

## Pass gates (heldout, read point, replication rule applies)

1. Post-eviction differential (N=256 and N=384) positive in mean with
   strict-win majority (≥ 17/32).
2. **Slow-lesion attribution**: the slow-lesion arm at N=256/384 must collapse
   the differential (lesioned ≤ half of intact) — the persistence must live in
   the new tissue, not in an incidental side effect of the graft.
3. Validity: delay-0 differential ≥ +0.042 at every eval — persistence may not
   be bought with acquisition.
4. Replication of any post-eviction differential on fresh demonstration
   samples.

## Stop rules

- Early stop at u100 if both rounds show post-eviction flat (≤ 17/32,
  |mean| < 0.02) with validity intact — the anatomy didn't take; the next move
  is diagnosis (is anything routed through slow cells at all? are α values
  pinned high?), not budget.
- Stability: the slow cells are the first weakly-contractive elements in a
  substrate whose stability story is contraction. Gradient rejection,
  non-finite telemetry, or `hidden_absmax` growth without bound ends the run
  loudly; the α ceiling (0.05) and the unchanged bounded operators are the
  mitigations.
- A pass promotes M2 (mode switching and reacquisition on the slow-tissue
  anatomy) and licenses the H-organ build in parallel.

## Result — first attempt (stopped by rule at u100) and diagnosis

The first run (`runs/fable2-m1c/`) stopped at u100: post-eviction flat with
slow-lesion a no-op everywhere, and the validity gate broken (+0.011 → +0.006
vs the +0.042 floor). Per the stop rules, diagnosis before budget
(`m1c-diagnosis.json`):

- **Write path alive**: mode-conditional content lands in slow cells at
  exposure end (delta 0.019 ≈ their full magnitude).
- **Gate never gated**: α frozen at its 0.0200 initialization (variance 1e-5).
  At fixed α=0.02 the exposure content decays to ~1.7e-7 across 256 filler
  tokens — 17-token tissue in practice; the slow-state growth through filler
  (ratio 2.36) is filler dilution overwriting the mode.
- **Credit starved twice**: slow-rule gradient 0.0012 vs organ 23 (the −3.0
  graft logit ≈ 2% read share throttles credit) and nothing survives to query
  time to credit anyway. Grafted edge logits moved 0.007 in 100 updates.

The gate could never learn to hold because the evidence that holding pays was
erased before every lesson.

## Amendment 1 — M1c-b (initialization, not architecture; pre-execution)

Two constants change, everything else identical: `slow_initial_alpha`
0.02 → **0.002** (hold-by-default: ~21% span-wide retention at init, exposure
still writes to ~half saturation over 672 microsteps; the gate learns to
spike from a holding baseline instead of learning to hold from an erasing
one) and graft edge logit −3.0 → **−1.0** (read share ~15–30%, credit
unthrottled). The validity gate polices the stronger splice's behavioral
cost; the stability clause covers the weaker contraction. Run:
`runs/fable2-m1cb/`.

## Result (M1c-b) — stopped by rule at u100

The initialization fixes changed the story but not the ending
(`runs/fable2-m1cb/`): the **first nonzero lesion attribution in the
program** appeared — at d128 the differential halves-to-thirds under
slow-lesion in both rounds (u50: +0.0085 → +0.0040; u100: +0.0042 →
+0.0013) — but the validity gate broke for the third consecutive delayed
run (+0.007 → +0.003 vs the +0.042 floor) and every differential
compressed with it.

Three replications now share one signature: **the mixed-delay objective
itself is destructive.** Five of seven delay draws are episodes the
organism cannot yet satisfy; their hinge gradient has no productive
direction and trades against the delay-0 policy — acquisition erodes faster
than any new route can capitalize on it. The organ showed its first
flicker; the curriculum poisoned the patient.

## Amendment 2 — M1c-c (anchored interleave; pre-execution)

One change: every update with a delayed episode also computes the full
paired triple at delay 0 (same item, mode, and demonstration sample) and
**sums both losses**, so each optimizer step must respect acquisition while
learning persistence — delay pressure can add routes but can no longer pay
for them with the delay-0 policy. Ladder, anatomy (n_slow 16, α₀ 0.002,
graft −1.0), budget, gates, and instruments unchanged. Per-update cost
roughly doubles on delayed draws; run: `runs/fable2-m1cc/`.

## Result (M1c-c)

Pending.
