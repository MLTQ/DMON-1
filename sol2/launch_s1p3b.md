# `launch_s1p3b.sh`

## Purpose

Launches one resumable S1-P3b uniform-screen or result branch from the mastered private
transition acquisition checkpoint.

## Contracts

- Accepts branch, genome rate, result root, and optional frozen-budget overrides.
- Selects only RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39` and never
  exposes the RTX 2070S reserved for `jewels`.
- Always enables directional calibration and append-only reserve growth.
- Enforces the five-point fixed-example A perturbation bound before B training.
- Defaults to the 3,000-update result budget; the 1,000-update screen passes explicit
  update/evaluation/final-evaluation budgets.
