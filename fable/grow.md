# grow.py

## Purpose

Runtime growth — the operation that justifies choosing a cellular substrate at
all (PROJECT.md S2) — plus the F1 arm runner and recruitment probe.

## Components

- `grow_field(model, opt, n_new)` — in-place field extension: appended
  `sources`/`logit` rows, donated slots, optimizer moment surgery, state
  migrator. Returns `(new_ids, migrate, grafts)`.
- `recruitment_probe(checkpoint, added, device)` — holdout BPC with added
  cells zeroed+frozen vs untouched.
- CLI: `python -m fable.grow --arm {small,grown,born} ...` (F1;
  small=80 fixed, grown=80→128 at `--grow-at`, born=128).

## Decisions

- **Donated slots are mandatory, not optional.** Old cells' dendrites are
  fixed at wiring time; without each old mutable cell donating its weakest
  slot to a new cell, nothing would ever read new tissue and F1 would be
  unfalsifiable by construction (the dmon/stream lattice failure, in
  connectome costume).
- **Output-silent graft by state, not by readout surgery**: new cells start at
  h=0 so their value-projected messages are zero; readout/embedding/rule are
  untouched. Parameter delta is exactly `n_new × K`.
- **Adam moments survive** for all shared parameters (object identity) and the
  old slice of the resized logit tensor — otherwise "growth helps" could be a
  disguised optimizer restart (check the operators, not the masks).
- Growth fires at `grow_at + 1`, after the `grow_at` eval, so the transition
  cost has a clean before-point.
- The post-graft probe checkpoint (`post_graft.pt`) is saved at
  `probe_at + 1`; both probes run post-hoc from checkpoints so the training
  run itself is untouched by probe machinery.

## Contracts

- Enforced by smoke_test.py: exact param delta, wiring preserved outside
  donated slots, every graft reads a new cell, reachability survives, state
  migrates with old values bit-identical, moments preserved (old slice
  nonzero, new slice zero).
