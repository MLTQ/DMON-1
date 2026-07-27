# `promote.py`

## Purpose

Selects the strongest completed SOL run by held-out BPC and atomically installs its
checkpoint as the local console's live organism.

## Components

### Candidate validation
- **Does**: Requires a SOL `summary.json`, loadable `best.pt`, finite BPC, and exact
  agreement between summary, checkpoint metadata, and completed JSONL history.
- **Does**: Rejects promotion eligibility when final or worst post-best BPC regresses
  beyond the configured threshold.
- **Rationale**: A stale or mislabeled tensor must never become the visible organism.
- **Rationale**: A transiently good checkpoint from an organism that later collapses is
  experimental evidence, not a viable live creature.

### `promote_best_checkpoint`
- **Does**: Ranks valid candidates, atomically copies the winner to `sol/runs/live.pt`,
  records unstable candidates separately, and writes `live.json` provenance.
- **Interacts with**: `load_organism` in `checkpoint.py` and `serve.py`.

### CLI
- **Does**: Accepts repeated `--run` directories and an optional destination.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `serve.py` | Default promoted checkpoint is `sol/runs/live.pt` | Default destination |
| Local operator | `live.json` names the source run, BPC, checkpoint/run updates, stability, eligible ranking, and rejected candidates | Manifest fields |
