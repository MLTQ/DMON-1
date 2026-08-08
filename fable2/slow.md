# slow.py

## Purpose

M1c's graft machinery: carry an M0-anatomy checkpoint into a slow-tissue
organism exactly, and couple the new tissue in via the growth machinery's own
weak-slot convention. Preregistration: `experiments/m1c-slow-tissue.md`; the
kernel side (tissue type, α band, layout) lives in sol2 (`config.py`,
`tissues.py`, `model.py`) behind `n_slow`, default-inert.

## Components

- `graft_from_checkpoint` — every checkpoint tensor is copied `identical`
  (same shape) or `prefix` (old rows into the leading rows of a row-extended
  tensor; exact because slow appends after relay so old indices never move).
  The census must account for every tensor — no strict=False, no silent
  misses; anything else raises.
- `rewire_dormant_slots_to_slow` — one dormant dendrite slot per compute and
  output cell pointed at a slow source (round-robin) at raw logit −3.0;
  re-asserts the output-sink and read-boundary invariants afterward.
- `deactivate_grafted_slots` — contract helper: slots-off graft must reproduce
  the donor computation (tested to 1e-5; coupling tested by exact inequality).

## Decisions

- **Graft, not growth**: M1c warm-starts with a fresh optimizer, so no moment
  migration is needed — a fresh-build-plus-census copy is simpler and exact,
  and relay growth is explicitly refused while slow tissue exists
  (`growth._refuse_slow_tissue`).
- The slow-lesion arm (`run_mode_arm(..., freeze_slow=True)`) freezes slow
  tissue from the *filler onward* — exposure runs with slow alive so the mode
  may write into it; what survives eviction without slow cells is not theirs.
