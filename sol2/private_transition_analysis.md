# `private_transition_analysis.py`

## Purpose

Aggregates the four matched S1-P3b arms and evaluates the frozen capability,
allocation, reserve-recruitment, stability, and integrity criteria without interpreting
a single failed gate as a verdict on the broader architecture.

## Contracts

- Reads `plastic`, `uniform`, `consolidated`, and `shuffled` metrics from one root.
- Capability and measured-allocation success remain separate conclusions.
- Allocation compares measured protection with the mean-matched uniform arm; shuffling
  must remove at least half of a positive measured advantage.
- Adapter and graft causal probes are reported independently from behavioral success.
- Missing private-adapter lesions produce explicit null measurements rather than an
  invented zero effect.
