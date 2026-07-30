# config.py

## Purpose

Single source of configuration for the fable organism, its stream, its optimizer,
and its evaluation cadence. Deliberately *not* a mechanism switchboard.

## Components

- `FableConfig` — one frozen-in-spirit dataclass. `scaled()` returns a modified
  copy (used by F1's small/grown/born arms); `to_json`/`from_json` ride in every
  checkpoint and run directory.

## Decisions

- **No mechanism flags.** sol/grok accumulated dozens of `--no-X` toggles because
  mechanisms were on trial. The trials concluded (see `fable/README.md` §evidence);
  what remains is geometry and optimization. If a new mechanism goes on trial it
  gets a branch or a new field with a preregistration, not a default-off flag.
- **One LR schedule for every model.** `lr`, `lr_min`, `warmup_updates`, `updates`
  define warmup+cosine used identically by creature and GRU. sol S12 found decay
  was its largest gain and never gave it to the baseline; that asymmetry is
  structurally impossible here (both arms read the same fields).
- **`updates` doubles as the cosine horizon** so a run's schedule is fully
  determined by its config — no hidden scheduler state to mismatch on resume.

## Contracts

- `vocab_size` is 0 until the corpus fills it; consumers must not build embeddings
  before then.
- `n_input + n_output + n_mirror <= n_cells` (checked in `model.py`).
- Growth (F1) changes `n_cells` only; rule parameters are shared and untouched by
  construction.
