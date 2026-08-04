# `genome_rate_screen.py`

## Purpose

Selects the tissue-genome plasticity rate from the three S1-P3b uniform pilot branches
without allowing the result arms to influence that choice.

## Contracts

- Accepts explicit `RATE=METRICS_PATH` candidates and refuses duplicate rates.
- Maximizes terminal `min(A,B)`.
- Among candidates within two percentage points of the best worst-organ score, prefers
  higher B accuracy; an exact remaining tie prefers the lower rate.
- Writes an optional atomic JSON selection artifact.
