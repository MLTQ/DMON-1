# `launch_s1p2_organs.sh`

## Purpose

Provides the immutable one-branch launch command for the S1-P2 organ-attachment GPU
experiment. Service orchestration can schedule branches in waves without duplicating or
silently changing scientific arguments.

## Components

### Branch launcher

- Accepts one canonical branch and a shared result root.
- Selects only physical RTX 4090 UUID
  `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- Loads the promoted larger/deeper acquisition checkpoint.
- Uses the preregistered batch-24 checkpoint configuration, 2,000 adaptation updates,
  200-update evaluations with 16 batches, 64-batch terminal estimates, and the frozen
  removal/recovery budgets.
- Always requests resume; an absent branch checkpoint starts normally.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU services | Exactly two positional arguments | CLI order or branch vocabulary |
| `jewels` workload | No SOL2 visibility of the 2070S | Changing or removing the UUID mask |
| S1-P2 analysis | Output directory equals the canonical branch name | Custom branch subdirectories |

## Notes

The launcher starts one process. The operator schedules at most two simultaneous
branches on the 4090 and never spills a branch to another device.
