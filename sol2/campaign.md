# `campaign.py`

## Purpose

Provides immutable manifests and atomic dependency-aware execution for multi-seed,
multi-size SOL2 campaigns. It lets independent workers share a campaign directory
without duplicating acquisitions or treating partial output as completion.

## Components

### `write_manifest` / `load_manifest`

- Validate job identifiers, commands, relative artifacts, and dependencies.
- Hash canonical protocol and job content and refuse to overwrite a different plan.

### `claim_ready_job` / `finish_job`

- Serialize state changes with an OS file lock and atomically persist state JSON.
- Claim only jobs whose dependencies completed successfully.
- Require both a zero exit code and the declared artifact before recording completion.
- Optionally require an exact nested JSON predicate, such as `mastered == true`, before
  releasing dependent jobs.
- Permit explicit retry of failed work and time-based recovery of abandoned claims.
- Enforce a persistent maximum-attempt count so deterministic failures cannot loop and
  consume an unattended compute allocation indefinitely.

### `run_worker`

- Replaces portable `{python}` and `{root}` command placeholders at execution time.
- Optionally pins one worker to a physical CUDA identifier through its environment.
- Captures one append-only log per job and waits for other workers when downstream work
  is blocked by a currently running dependency.
- Persists elapsed wall time for every completed or failed attempt beside its worker and
  exit status.

## Contracts

- A manifest is immutable once written; a different hash requires a new directory.
- Workers cannot double-claim a job while sharing a filesystem with working `flock`.
- Output-file existence alone never promotes an unclaimed or failed job.
- A syntactically complete but scientifically ineligible artifact remains a visible
  failed job when its frozen completion predicate does not pass.
- Failed dependencies leave descendants pending and visible rather than running them.
- Stale-worker reclamation is opt-in because wall-time guesses can otherwise duplicate
  valid long training jobs.
- Commands are argument vectors, never shell strings.
- Campaign state is operational metadata, not scientific result data.
