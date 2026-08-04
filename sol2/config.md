# `config.py`

## Purpose

Defines the complete geometry, stability treatment, stream, optimizer, and evaluation
contract for a SOL2 run. The serialized configuration travels with every checkpoint.

## Components

### `Sol2Config`

- Separates input, memory, compute, relay, and output tissues.
- Configures a fixed organ-query bank that can read variable-sized output tissue.
- Exposes the two S0-R factors: bounded graph operators and private cell identity.
- Optionally gives mutable cells zero-output private low-rank transition residuals;
  rank zero preserves legacy checkpoint structure exactly.
- Defaults to a constant learning rate because indefinite plasticity must not depend on
  reaching the end of a one-shot cosine schedule.
- Computes total and mutable cell counts from tissue counts.

## Contracts

- Every tissue is non-empty.
- Every output organ has at least one learned query.
- Memory cells are excluded from learned rule updates.
- `n_mutable` is also the exact number of private-expression rows in identity arms.
- Positive adapter rank creates exactly one down/up parameter pair per mutable cell;
  adapter gain is nonnegative and the up projection starts at zero.
- `initial_active_dendrites <= n_dendrites`; unused slots are dormant growth capacity.
- `scaled()` validates the resulting configuration.
- Training and all controls consume the same optimizer schedule fields.
