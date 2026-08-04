# `consolidated_attachment.py`

## Purpose

Runs one resumable S1-P3/S1-P3b branch from a mastered larger-organism checkpoint. It
can activate directional reserve dendrites, attaches an independent B organ, trains
without A rehearsal, and measures whether utility-gated plasticity permits simultaneous
A/B capability.

## Components

### Branch construction

Every branch independently reloads the same acquisition checkpoint, recalibrates A
utility, attaches the same seeded B module, and preserves mature AdamW moments. Branches
are ordinary plasticity, mean-matched uniform protection, measured consolidation, and
within-tissue shuffled utility. Directional calibration avoids protecting every outgoing
edge merely because its source cell is useful.

### Reserve growth

When enabled, each branch activates two dormant inputs on the lowest-utility half of
internal targets: one from mature high-utility tissue and one from reserve tissue. No
edge is deleted. A fixed-example before/after score records graft perturbation and an
optional maximum-drop guard can invalidate an overly disruptive graft before B trains.
The exact graft ledger is checkpointed and checked during resume.
Terminal results augment each ledger row with raw-logit drift and its final bounded
attention bias, so recruitment is visible even when an acute lesion is small.

### Protected training

The usual procedural episode and gradient guard run unchanged. For protected branches,
`ConsolidationPolicy` scales only the realized substrate update; B remains freely
plastic. Checkpoints preserve model, optimizer, living state, counters, histories,
utility profile, and RNG state.

### Attribution

Every interval scores A and B from separate cloned states and reports identity, adapter,
edge, and genome drift by equal-count measured-utility quartile. Terminal probes compare
equal-size high- and low-utility internal-cell lesions, zero private adapters in reserve
or all internal cells, and disable the newly grafted edges alongside existing lesions.

## Contracts

- All branches consume the same B generator seeds and evaluation examples.
- A's physical organ remains byte-identical.
- Calibration must reproduce exactly before a checkpoint may resume.
- Reserve grafts must reproduce exactly before a checkpoint may resume.
- A and B evaluations never prime one another's living state.
- The runner makes no success claim from a tiny smoke run; result-bearing budgets live
  in the frozen S1-P3 experiment document and 4090-only launcher.
- CLI completion prints compact behavioral and drift summaries; full neuron telemetry
  remains in `metrics.json` rather than flooding the service journal.
