# fable/ — synthesis spike (S0 close-out + first real growth test)

A third experimental line for the character organ, built on what `sol/` and `grok/`
*measured*, not on what they hoped. Written 2026-07-30, immediately after syncing the
chase-1 results from Aine.

## The evidence this spike synthesizes

Every claim below has a run behind it. Sources: `grok/runs/s0_gap/LADDER.md`,
`grok/runs/s0_chase1/LADDER.md`, `grok/findings.md`, `sol/experiments/`,
`dmon/stream` commit history.

1. **The substrate learns and its state is causal.** Reset-each-token and cell-shuffle
   ablations cost +2.5 and +1.7 BPC everywhere they were run. This part is settled.
2. **Credit machinery is a tax at S0.** `no_credit` beat baseline in the gap ladder
   (+0.173 vs +0.195). Eligibility, reward, fast efficacy, reverse credit: all
   default-off in sol's own findings, confirmed by grok. Fable deletes them entirely
   rather than flagging them off — they live in `sol/` if they are ever needed.
3. **Metabolism is neutral-to-harmful at S0** (sol S9/S10, grok findings #4). Deferred
   to S1, per the architecture's own ordering.
4. **Scale is the lever that works — on the connectome.** Gap 0.19 → 0.12 mean going
   64→128 cells. (On the 2D lattice, capacity was inert — `dmon/stream` S2 showed
   born-large ≈ grown ≈ small, an 11×11 core doing all the work. The sparse directed
   connectome is the substrate where added cells pay. This is why fable is a
   connectome, not a grid.)
5. **The readout was a bottleneck hiding under the gap.** Chase-1's side arm — concat
   readout instead of mean-pooling over output cells — beat its parameter-matched GRU
   (**−0.028 BPC**, 657,345 vs 657,985 params) at *half* the update budget, holdout
   still descending at stop. The primary mean-readout arms floored at +0.13 (the
   preregistered fail branch). Mean pooling was throwing away most of the readable
   state.
6. **Stability is the actual blocker now.** At lr 3e-3 constant, clip 1.0: seed 13
   NaN'd at ~5.2k updates; 192 cells diverged from ~1k updates. Nothing in the stack
   normalizes the recurrent messages, and the LR never decays — despite sol S12
   finding that late cosine decay was **the largest capability gain in its entire
   series** (−0.114 best / −0.196 final BPC, post-best regression 0.082 → 0.000).
7. **The schedule must go to both arms.** sol never re-ran its GRU control after S0,
   so its decay gains were never mirrored into the baseline and its "gap" numbers
   are indefensible from S12 onward. Fable gives creature and GRU the identical
   warmup+cosine schedule, always.
8. **Scale means parameters-through-the-connectome, not tissue count.** sol S14: 14×
   cells at fixed rule width bought 0.010 BPC. grok's scale arm (cells *and* hidden
   *and* dendrites, 104k → 528k params) shrank the gap. F1's growth therefore grows
   the field in a configuration where added cells carry real capacity.

## What fable is

The smallest organism that keeps every mechanism that earned status and none that
didn't, plus the two things the evidence now demands: a **concat readout** and
**stability machinery** (message normalization, LR warmup+cosine, state guards that
log rather than lie).

Kept from grok/sol: sparse directed dendrites with local attention, one shared
GRUCell rule with per-cell persistent state, stream-written mirror ring
(write-only, stop-grad — the BYOL collapse guard), multi-lane chunk BPTT on a
continuous stream, contiguous 90/10 split, held-out BPC with reset/shuffle
ablations, parameter-matched GRU control on the identical stream.

Deleted: eligibility traces, delayed reward, fast efficacy, reverse credit
transport, routing games, metabolism, structural rewiring. (All still exist in
`sol/`; fable is not where they get relitigated.)

New: RMSNorm on dendrite messages and readout, warmup+cosine LR **applied
identically to creature and GRU**, a state-health monitor (max |h|, message norms,
logit scale — logged next to BPC so a blowup is visible *before* it is a NaN), and
**runtime growth** as a first-class operation (F1 below).

## Experiments

### F0 — S0 close-out (the concat bet, at full budget)

Chase-1's concat arm won at 4k updates as a side ablation. F0 is that result made
load-bearing: 128 cells, h=128, concat readout, no credit, 8000 updates, seeds
7/13/21, matched GRU per seed, on Aine.

- **Pass**: mean gap ≤ 0.00 BPC (creature beats or matches GRU) across 3 seeds, no
  NaNs, reset/shuffle deltas stay strongly positive. This clears S0's original bar
  ("competitive with parameter-matched GRU") with room.
- **Soft pass**: mean gap ≤ 0.05 — S0's literal bar.
- **Fail**: gap > 0.05, or any NaN, or ablation deltas collapse (which would mean the
  concat head is bypassing the substrate — the readout doing the work instead of the
  organism. Guard: `probe_bypass` ablation, below).

**The bypass guard.** A concat readout is a bigger head, and a bigger head could in
principle learn the task from the input port's immediate echo alone. F0 therefore
keeps the reset ablation *and* adds a head-only control: freeze the organism at init,
train only the readout head, same budget. If head-only lands anywhere near the full
creature, the win is the head's, not the substrate's. (This is the "beat the best
baseline, not a dead one" discipline — the denominator here is what a readout alone
can do.)

### F1 — growth on the substrate where capacity pays (S2, first honest attempt)

`dmon/stream` proved the *mechanism* of growth is easy (transition cost ≈ 0) but its
lattice had an inert capacity axis, so the test was vacuous — "S2 unfalsifiable on
this task rather than failed." grok proved the connectome's capacity axis is live
(64→128 shrinks the gap). Neither line ever ran growth where capacity mattered.
That conjunction is exactly one experiment:

Three arms, same stream, same total updates, seed-matched:
- `small`: 64 cells fixed
- `grown`: 64 cells, grown to 128 at update 2000 (new cells wired by the same
  dendrite-sampling rule, new rows zero-init into the readout so the graft is
  silent at t=0; optimizer state rebuilt, param count changes only by the readout
  rows — the shared rule is *why* growth doesn't touch rule parameters)
- `born`: 128 cells from update 0

- **Pass** (S2's bar): `grown` final BPC beats `small` decisively and approaches
  `born`; transition cost ≈ 0; ablating the added cells at the end hurts (they were
  recruited), ablating them just after the graft doesn't (the improvement is learned,
  not free).
- **Fail**: added cells stay inert (grown ≈ small), or the graft destabilizes what
  worked (transition cost large), or growth "helps" but ablating new cells at the end
  is free (the win came from optimizer reset or readout size, not recruitment — check
  the operators, not the masks).

F1 also carries a stability hypothesis worth one arm: chase-1's s192 diverged *from
scratch*. If `grown` reaches large-field performance stably where `born`-large
diverges, progressive growth is not just a capacity mechanism — it is the way large
fields get trained at all. That would make S2 load-bearing for scale, not a parlor
trick.

### F2 — adaptability under regime cycling

Compression is the stepping stone, not the destination: the organism is meant
to be a *vessel for intelligence*, which means adaptability has to be measured
directly, not inferred from BPC. F2 cycles the stream between Tiny Shakespeare
and an entropy-matched vocabulary permutation of it, with lanes aligned so the
switch is coherent, and scores **savings** (does re-adapting to a returning
regime get cheaper with each visit?) and **interference** (does time in B
damage steady-state A?) — lifetime properties that in-context inference cannot
fake. That trap is documented: `dmon/exp` ran the block-switch version and the
GRU won it by re-inferring the symbol map in ~10 characters. Full design and
pass/fail: `experiments/f2-adaptability.md`.

### Ordering

F0 first — it is the S0 gate and every F1 number is denominated in the F0
configuration. F1 launches only after F0's verdict is recorded. F2 is
orthogonal (different schedule regime, own controls) and runs on the second
GPU alongside F0's creature queue.

## Run

```bash
# Contracts + learning smoke (CPU, ~2 min)
.venv/bin/python -m fable.smoke_test

# Local short train (any arm: creature | gru | bypass)
.venv/bin/python -m fable.train --model creature --updates 500 --device cpu \
  --out-dir fable/runs/dev

# F0 on Aine (both GPUs; writes fable/runs/f0/LADDER.md)
bash fable/run_f0.sh

# F1 growth (after F0 verdict)
bash fable/run_f1.sh
```

## Layout

| File | Role |
|------|------|
| `README.md` | This charter |
| `config.py` | One dataclass, no flag graveyard |
| `graph.py` | Directed dendrites + attention aggregation |
| `cell.py` | Shared GRUCell rule |
| `model.py` | Organism: ports, mirror ring, tick, concat readout |
| `stream.py` | Multi-lane continuous corpus (90/10 contiguous) |
| `train.py` | One loop for every arm: chunk BPTT + warmup/cosine + health monitor + CLI |
| `evaluate.py` | Held-out BPC; reset / shuffle-internal / mirror-zero ablations |
| `baselines.py` | Parameter-matched GRU (closed-form sizing, seeded init) |
| `grow.py` | Runtime growth + recruitment probe (F1) + CLI |
| `adapt.py` | Regime-cycling stream + savings/interference analysis (F2) + CLI |
| `summarize.py` | LADDER.md writer |
| `smoke_test.py` | Contract gate before any GPU time |
| `run_f0.sh` / `run_f1.sh` / `run_f2.sh` | Aine launchers |
| `experiments/` | Preregistrations + results |

## Relation to the other lines

- `sol/` — the full scientific stack; where credit/metabolism/structural machinery
  lives and where anything fable proves gets promoted for rigor.
- `grok/` — the previous lean line; fable supersedes it as the fast-iteration line
  (grok's benchmark code produced chase-1 and remains the reference for its runs).
- `dmon/stream` — falsified lattice line; its growth harness design (small/grown/born,
  transition cost) is reused by F1 on a substrate where it can actually bind.
- `dmon/` (rest) — bookmarked display-organ work, untouched, returns at S4.
