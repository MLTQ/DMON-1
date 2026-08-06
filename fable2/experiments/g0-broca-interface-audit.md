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

Complete 2026-08-06 on the physical RTX 4090. **All gates pass**; G1 is licensed.
Artifact: `runs/fable2-g0/g0-audit.json` on Aine, SHA-256
`2bd6e254781de1eef960ce665f4fd5b2b5b3ec56d678c3b556b3cdfa7dfbfb29`.

- Qwen3.5-4B exposes **32** decoder layers (l0c1q's audit table said 24 hybrid
  blocks for the 2B/4B; the fraction design absorbed the discrepancy — resolved
  depths 7/15/23/31). `native_logit_error` was exactly 0.0.
- Zero-init no-op: max |injected − bare| logit delta exactly 0.0 on the live
  gradient path.
- Per-depth fp32 label sensitivity to bounded random controls: 7.38 (d7), 2.46
  (d15), 0.98 (d23), 0.93 (d31); max/min ratio 7.97 — nondegenerate, no
  amendment required. Shallower sites are amplified by more downstream compute,
  as expected.
- Recruitment: all four depth heads live at exact zero-init (norms 0.003–0.025);
  after one optimizer step trunk, sensor, bases, organism tissues, graph, and
  expression all live. Zero backbone gradient tensors in both stages.
- Peak allocation/reservation 15.25/16.13 GiB.
