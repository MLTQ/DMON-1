# `development.py`

## Purpose

Turns persistent plasticity pressure into bounded, auditable developmental decisions
without assigning cells to an organ or allowing unconstrained expansion.

## Components

### `DevelopmentController`

- Tracks consecutive high-pressure and pressure-plus-plateau checks.
- Enforces minimum age, B mastery stop, refractory interval, and maximum event count.
- Serializes only decision state; model growth remains in `growth.py`.
- Exposes immutable threshold configuration separately so resume can reject a changed
  developmental protocol before restoring streak state.

### `DevelopmentDecision` / `decision_payload`

Records every threshold input and trigger reason in JSON-safe form.

## Contracts

- The controller observes only B accuracy, gradient pressure, and update number.
- A retention evaluation never enters the growth trigger.
- A single noisy check cannot trigger growth under the frozen two-check patience rule.
- Growth stops after B reaches 80%, after two events, or during the refractory period.
- Exact resume restores streaks, prior B accuracy, event count, and refractory state.
