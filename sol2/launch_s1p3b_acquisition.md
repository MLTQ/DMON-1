# `launch_s1p3b_acquisition.sh`

## Purpose

Launches the frozen S1-P3b seed-7 mastery-gated acquisition run with a rank-4 private
transition adapter on the 400-cell organism.

## Contracts

- Accepts one result directory and always requests exact resume.
- Selects only RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- Never exposes the RTX 2070S reserved for `jewels`.
- Uses the frozen geometry, curriculum, 64-batch mastery check, and 10,000-update cap.
