# `organ_attachment_analysis.py`

## Purpose

Combines the four independently executed S1-P2 branch artifacts and applies the
decision rules frozen in `experiments/s1p2-organ-attachment.md`.

## Components

### `analyze_organ_attachment`

- Requires control, full, organ-only, and scratch `metrics.json` files beneath one
  result root.
- Rejects protocol mismatches or any branch with rejected optimizer updates.
- Computes final B capability, early acquired-minus-scratch transfer, A retention,
  removal/reattachment recovery, modular organ-only reuse, and causal lesion penalties.
- Reports each gate independently and a conjunctive strong-success result.

### CLI

- Prints the aggregated JSON and optionally writes it through `--out`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| S1-P2 result review | Branch directories use canonical branch names | Renaming branch output directories |
| Frozen decision | Thresholds match the preregistration | Post-result threshold changes |
| Attribution | Full, organ-only, and scratch protocols are identical | Aggregating mismatched budgets or seeds |

## Notes

The organ-only 22.5% diagnostic is reported numerically but intentionally excluded from
the conjunctive biological success gate.
