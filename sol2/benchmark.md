# `benchmark.py`

## Purpose

Materializes and optionally executes the preregistered S0-R stability × identity
factorial.

## Components

- `factorial` — four creature treatments plus parameter-matched GRU and transformer
  controls for each identity-sized parameter budget.
- CLI — writes an explicit manifest by default; `--execute` runs the listed arms and
  resumes any arm whose `latest.pt` checkpoint already exists.

## Contracts

- Seeds, treatments, output paths, and complete configurations are recorded before a
  GPU result exists.
- Bounded/unbounded creature pairs have identical parameter counts.
- Each private-identity parameter budget gets freshly matched controls.
- Manifest generation does not allocate a GPU or begin training.
- A stopped sequential launch continues from complete process checkpoints rather than
  silently restarting finished compute.
