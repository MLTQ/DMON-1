# `config.py`

## Purpose

Defines the complete geometry, stability treatment, stream, optimizer, and evaluation
contract for a SOL2 run. The serialized configuration travels with every checkpoint.

## Components

### `Sol2Config`

- Separates input, memory, compute, relay, and output tissues.
- Exposes the two S0-R factors: bounded graph operators and private cell identity.
- Defaults to a constant learning rate because indefinite plasticity must not depend on
  reaching the end of a one-shot cosine schedule.
- Computes total and mutable cell counts from tissue counts.

## Contracts

- Every tissue is non-empty.
- Memory cells are excluded from learned rule updates.
- `initial_active_dendrites <= n_dendrites`; unused slots are dormant growth capacity.
- `scaled()` validates the resulting configuration.
- Training and all controls consume the same optimizer schedule fields.
