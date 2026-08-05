# `wiki_output_credit.py`

## Purpose

Defines the training-only fixed target code used by L0-C1h to deliver dense,
passage-conditioned credit directly to dedicated output tissue without adding an
inference path or trainable auxiliary parameters.

## Components

### `fixed_output_codebook`

- Builds four deterministic orthonormal Walsh-style label codes.
- Applies a seed-controlled dimension permutation and sign mask without consuming the
  global random stream.

### `output_tissue_credit_loss`

- Mean-pools final output-cell state for each matched passage branch.
- Scores normalized state against the fixed codebook and applies four-way target loss.
- Reports auxiliary accuracy and paired output-state RMS separation.

## Contracts

- The hidden width is divisible by four.
- The codebook has no parameters and is regenerated from serialized configuration.
- Both paired branches share the same question and carry incompatible target labels.
- Development, held-out evaluation, and inference never consult auxiliary scores.
- Gradient credit enters only through output tissue and the organism's existing graph.
