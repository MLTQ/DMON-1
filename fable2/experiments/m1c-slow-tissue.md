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

## Result

Pending.
