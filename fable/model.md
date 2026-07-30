# model.py

## Purpose

The organism: index-addressed cell blocks, mirror ring, tick loop, concat
readout, chunked forward for BPTT.

## Components

- `OrganismState` — `h [B,N,H]` + `mirror_cursor`; `detach()` at chunk edges.
- `StepHealth` — max |h|, message RMS, logit scale; collected on the last step
  of a chunk and logged next to BPC (a blowup must be visible while finite).
- `Fable` — `step` (one token: mirror write, spt micro-steps, readout),
  `forward_chunk` (mean CE over the chunk), `initial_state`.

## Decisions

- **Blocks are index buffers, not layout assumptions.** Growth appends internal
  cells past the output block, so "output cells are the last O indices" is
  false after the first graft. All addressing goes through
  `input_idx/mirror_idx/internal_idx/output_idx/mutable_idx`.
- **Concat readout only.** Chase-1's decisive result: mean pooling over the
  output bank was discarding most readable state (−0.028 gap for concat vs
  +0.13 floor for mean). Fable does not carry the losing modes.
- **Mirror ring**: FIFO of raw detached token embeddings, write-only from the
  stream (BYOL-collapse guard per ARCHITECTURE.md); excluded from rule compute
  entirely; `mirror_zero` ablation in evaluate.py prices its contribution
  separately from distributed state (grok debt #23).
- **Sensory frontend is `emb * gain_i + bias_i` per input cell** (2·I·H
  params). grok's `input_proj` was a rank-≤H linear-on-linear frontend eating
  31–60% of the budget and distorting the matched comparison (debt #21).
- **RMSNorm on messages** every micro-step: fixes per-step signal scale, which
  is the substrate-side half of the chase-1 NaN fix (the optimizer-side half is
  in train.py).
- Sensory drive enters at micro-step 0 only (grok convention, kept).
- Out-of-place `index_copy` everywhere state is written — in-place writes on
  graph tensors are an autograd hazard.

## Contracts

- `n_ports ≤ n_cells` asserted at build; `vocab_size > 0` required.
- `step` never mutates mirror cells through the rule; only the stream writes
  them, detached.
- `frozen_idx` (eval-only) zeroes those cells after every micro-step — used by
  F1's recruitment probe.
