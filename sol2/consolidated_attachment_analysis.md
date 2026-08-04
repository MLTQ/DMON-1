# `consolidated_attachment_analysis.py`

## Purpose

Aggregates the three matched S1-P3 branch artifacts and applies the frozen simultaneous
retention and attribution gates without manual arithmetic.

## Outputs

- Terminal A, B, and worst-organ accuracy for each branch.
- Measured and shuffled gains over the equally large plastic control.
- Low/high measured-utility drift ratios for cell expression and edges.
- High- versus low-utility B lesion penalties.
- Stability, A-organ integrity, and the preregistered primary-success verdict.

## Contracts

- Reads only canonical `plastic`, `consolidated`, and `shuffled` directories.
- Uses terminal 64-batch estimates, never interval telemetry, for the gates.
- Reports recruitment and reuse diagnostics separately from the primary success gate.
- A one-seed success remains a mechanism result and does not bypass replication.
