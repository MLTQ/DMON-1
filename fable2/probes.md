# probes.py

## Purpose

G2a: no-training relevance-gradient probes on a trained Broca checkpoint —
grade exposures by how much passage structure they retain (prose / shuffled /
unigram soup / other passages / none) and measure what actually survives the
FIFO→injection transport. Preregistration: `experiments/g2a-transport-probes.md`.

## Components

- `CONDITIONS` — seven fixed arms; `_exposure_ids` materializes each with a
  per-question sha-seeded generator (deterministic across runs and hosts).
- `run_probes` — rebuilds the system from the checkpoint's own config, loads the
  organism state dict, verifies depth agreement, runs every condition from a
  fresh episode state in eval mode, and writes per-question and mean losses plus
  the named gradient differences.

## Decisions

- **Shuffle and soup are length-matched to the own passage** so every organism
  arm processes identical token counts — the conditions differ only in
  structure, never in exposure duration.
- **`pool_random` samples the corpus unigram pool**, not the tokenizer range —
  uniform vocab draws would mostly produce tokens the backbone never maps to
  wiki-like states, confounding "no structure" with "out of distribution".
- Off-domain partners come from meta_train (never heldout), keeping the probe
  read-only with respect to the heldout split's pairing structure.

## Contracts

- Checkpoint schema/depth mismatches raise; conditions are contract-tested for
  determinism, length-matching, and full coverage in `test_fable2.py`.
