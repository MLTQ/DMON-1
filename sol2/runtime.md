# `runtime.py`

## Purpose

Makes the architecture's concurrency claim executable: a foreground organism consumes
tokens continuously while a separate model copy computes bounded truncated updates.

## Components

### `LearningWindow`

An immutable token/target window with the exact persistent state and weight version
that preceded its first token.

### `RuntimeStats`

Counts ticks, queued/learned/rejected/stale windows, and active parameter version.

### `AsyncOrganismRuntime`

- Advances one active state under a short lock.
- Queues completed causal windows without pausing for backward.
- Trains a persistent shadow copy on a background thread.
- Atomically publishes accepted parameters while retaining foreground state.
- Drops work that is too stale or exceeds queue capacity.

## Decisions

- The active state is not reset or rolled back when weights change. Small versioned
  updates are the approximation being tested.
- The shadow optimizer persists across windows; it is not recreated per job.
- GPU publication synchronizes before the short atomic copy. This does not claim a
  zero-pause swap, only that stream ingestion and backward are separate activities.
- S0-R uses the synchronous scientific harness first. This runtime has a CPU contract
  gate and graduates to GPU only after the bounded kernel is validated.

## Contracts

- Every learning target is the token following its recorded input.
- A learning window starts from state captured before its first input.
- Rejected updates never reach the active model.
- Publishing weights never resets foreground hidden state or memory position.
- `close()` drains the worker safely and is idempotent.
