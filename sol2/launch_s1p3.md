# `launch_s1p3.sh`

## Purpose

Provides the immutable one-branch launch command for the S1-P3 consolidation
experiment.

## Contracts

- Accepts exactly one canonical branch and one result root.
- Selects only RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- Never exposes the RTX 2070S reserved for `jewels`.
- Loads only the 400-cell S1-P3 acquisition checkpoint.
- Uses the frozen 2,000-update, 64-calibration-batch protocol and always requests
  exact resume.
