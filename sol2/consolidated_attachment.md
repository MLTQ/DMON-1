# `consolidated_attachment.py`

## Purpose

Runs one resumable S1-P3 branch from a mastered larger-organism checkpoint. It attaches
an independent B organ, trains without A rehearsal, and measures whether utility-gated
plasticity permits simultaneous A/B capability.

## Components

### Branch construction

Every branch independently reloads the same acquisition checkpoint, recalibrates A
utility, attaches the same seeded B module, and preserves mature AdamW moments. Branches
are ordinary plasticity, measured consolidation, and within-tissue shuffled utility.

### Protected training

The usual procedural episode and gradient guard run unchanged. For protected branches,
`ConsolidationPolicy` scales only the realized substrate update; B remains freely
plastic. Checkpoints preserve model, optimizer, living state, counters, histories,
utility profile, and RNG state.

### Attribution

Every interval scores A and B from separate cloned states and reports parameter drift
by measured-utility quartile. Terminal probes compare equal-size high- and low-utility
internal-cell lesions, alongside the existing tissue and topology lesions.

## Contracts

- All branches consume the same B generator seeds and evaluation examples.
- A's physical organ remains byte-identical.
- Calibration must reproduce exactly before a checkpoint may resume.
- A and B evaluations never prime one another's living state.
- The runner makes no success claim from a tiny smoke run; result-bearing budgets live
  in the frozen S1-P3 experiment document and 4090-only launcher.
