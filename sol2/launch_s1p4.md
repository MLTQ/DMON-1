# `launch_s1p4.sh`

## Purpose

Launches one frozen S1-P4 anchor/development branch from the mastered S1-P3b
acquisition checkpoint.

## Contracts

- Accepts one canonical branch and one result root, with optional smoke-budget
  overrides.
- Selects only RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- Never exposes the RTX 2070S reserved for other work.
- Uses the frozen proximal rate, pressure thresholds, growth size, event cap, and
  five-point structural perturbation guard.
- Defaults to the 3,000-update, 16-batch interval, and 64-batch terminal protocol and
  always requests exact resume.
