# `routing_randomization.py`

## Purpose

Builds checkpoint-reproducible crossover assignments and exact randomization evidence
for live reverse-credit routing trials. It removes fixed corpus-window phase without
creating another organism or consuming the training RNG.

## Components

### `crossover_schedule`
- **Does**: Expands one assignment code into four-window `ABBA` or `BAAB` blocks.
- **Rationale**: Every block is arm-balanced and has zero covariance with a linear time
  trend.

### `deterministic_assignment_code`
- **Does**: Mixes trial/proposal identity, then keys the block-orientation bits by the
  topology seed.
- **Rationale**: Schedule assignment must differ across organisms while exact resume and
  global RNG state remain unchanged.

### `schedule_advantage`
- **Does**: Computes candidate-minus-incumbent mean reward for a balanced assignment.

### `RandomizationResult`
- **Does**: Retains observed advantage, extreme null assignments, total assignments,
  and the derived exact p-value.

### `exact_one_sided_randomization_test`
- **Does**: Enumerates every valid block assignment and returns its complete exact rank.
- **Rationale**: A commit must beat the corpus-phase null actually available to that
  trial, not merely have a positive noisy mean.

### `exact_one_sided_randomization_p_value`
- **Does**: Preserves a compact p-value-only analysis interface.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `routing.py` | Boolean schedule length equals trial length and null rank is exact | Block encoding or tail convention |
| Checkpoint resume | Assignment code recreates the stored schedule without RNG | Integer mixing or schedule layout |
| Experiment analysis | Each four-window block is balanced and trend-neutral | `ABBA` / `BAAB` patterns |

## Notes

- Exact enumeration is capped at sixteen blocks (`64` updates, `65,536` assignments).
- A one-sided p-value tests whether candidate routing is unusually beneficial; harmful
  or merely typical assignments reject.
