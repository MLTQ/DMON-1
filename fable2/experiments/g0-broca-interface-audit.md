# G0: Broca-graft interface audit

Status: preregistered 2026-08-06, before any GPU execution of the fable2 line.

## Question

Is the multi-depth injection interface mechanically sound on frozen Qwen3.5-4B —
exactly silent at birth, causally reachable at every preregistered depth, and
recruitable through the paired objective — before any training is attempted?

## Lineage

This is the "interface sensitivity/gradient audit" required by DMON-1-gws.22, in
the position the C1 line's backbone audit (l0c1q) occupied. It exists because the
program has paid for skipping cheap checks before expensive runs (HANDOFF.md: a
40-minute sweep in a regime where nothing could survive).

## Setup

- Backbone: frozen `Qwen/Qwen3.5-4B`, BF16, local files only, physical RTX 4090
  UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- Corpus: `sol2/experiments/l0c1-wiki-memory-corpus.json`, development split only.
- Config: fable2 defaults — depth fractions (0.25, 0.5, 0.75, 1.0), 4 control
  tokens, rank 8, control gain 1.0, seed 7.
- Command: `bash fable2/launch_g0.sh <result-root>`.

## Gates (all must hold; any failure stops the line before G1)

1. **Exact silence**: max |injected − bare| logit delta is exactly 0.0 on the live
   gradient path.
2. **Depth reachability**: bounded random controls at each depth alone produce
   nonzero fp32 label-logit deltas. The max/min sensitivity ratio is recorded; a
   ratio above 100 flags a dead or dominant site and requires a depth-selection
   amendment before G1, not a silent proceed.
3. **Recruitment**: at zero-init, every depth head receives finite nonzero
   gradient through the paired loss; after exactly one optimizer step, trunk,
   sensor, organism tissues, and connectome all receive finite nonzero gradient.
4. **Frozen backbone**: zero gradient tensors on Qwen parameters in both stages.
5. **Head fidelity**: `native_logit_error` is recorded; a value above BF16 noise
   (> 0.1) blocks G1 until explained.

Peak VRAM is recorded. If the audit itself cannot fit alongside the frozen model
on the 4090, that is a G0 failure, not a license to use the 2070S.

## Result

Pending.
