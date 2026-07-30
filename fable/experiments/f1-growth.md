# F1: runtime growth on a substrate where capacity pays (preregistered 2026-07-30)

*Launches only after the F0 verdict is recorded, on the F0-winning configuration.*

## Question

`dmon/stream` showed growth is mechanically cheap (transition cost ≈ 0) but its
lattice capacity axis was inert, making S2 unfalsifiable there. grok showed the
connectome's capacity axis is live (64→128 cells+width shrinks the GRU gap). Can a
creature **grow into** the large configuration at runtime — recruiting the new tissue
— rather than being born large?

Secondary hypothesis: chase-1's 192-cell arm diverged *from scratch*. If growth
reaches large-field performance stably where born-large diverges, progressive growth
is a stability mechanism, not just a capacity one.

## Protocol

Three arms, identical stream, identical total updates (8000), seeds 7/13/21. Ports
are fixed at 16 in / 16 out / 32 mirror for every arm, so the smallest field with
free tissue is 80 cells (16 internal):

| Arm | Geometry |
|-----|----------|
| `small` | 80 cells fixed (16 internal), h=128 |
| `grown` | 80 cells → 128 at update 2000 (+48 internal) |
| `born` | 128 cells from update 0 (this is F0's creature arm, copied in) |

Growth event (`fable/grow.py`): each new cell samples K dendrites from the whole
enlarged non-output pool; each old mutable cell donates its weakest dendrite slot
(never an output cell's forced sensory slot 0) to a new cell round-robin with the
edge logit reset to 0 — without donated slots, nothing would ever read new tissue
and it would be inert *by construction*. The graft is output-silent at t=0 because
new cells start at h=0, so their value-projected messages are zero until their
state develops. The readout, embedding, and shared rule are untouched; parameters
change by exactly `n_new × K` edge logits. Adam moments are preserved for every
surviving parameter and for the surviving slice of the resized logit tensor, so
the graft is not a disguised optimizer restart.

Measured:
- Held-out BPC every 500, all arms, full curves in the report
- Transition cost: BPC(eval straddling u2000) − BPC(eval before), `grown` vs `small`
- **Recruitment probe**: ablate (zero + freeze) the 48 added cells at u2500 and at
  u8000. Recruited means: ablation at u8000 hurts ≥ 0.05 BPC; at u2500 it hurts less
  than half that. If final-ablation cost ≈ 0, any `grown` win came from the donated
  slots' logit reset or elsewhere, not the tissue — check the operators, not the
  masks.
- State health (same monitor as F0) through the graft

## Pass / fail

- **Pass (S2 bar)**: `grown` final BPC beats `small` by ≥ 0.05 mean across seeds;
  transition cost ≤ +0.05 BPC; recruitment probe passes as defined above.
- **Strong pass**: additionally `grown` final within 0.03 of `born`.
- **Stability bonus** (reported either way): if a `born`-large arm diverges at any
  seed while the seed-matched `grown` arm does not, record progressive growth as a
  stabilizer.
- **Fail**: `grown` ≈ `small` (added tissue inert — the dmon/stream failure repeated
  on the substrate that was supposed to fix it, which would be a serious blow to the
  whole growth thesis), or transition cost > +0.2 (graft destabilizes), or
  recruitment probe fails (win is bookkeeping, not biology).

## Launch

```bash
# on Aine, from repo root, after F0
bash fable/run_f1.sh
python3 -m fable.summarize --root fable/runs/f1 --out fable/runs/f1/LADDER.md
```
