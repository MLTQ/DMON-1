# `interventions.py`

## Purpose

Provides reversible causal interventions that distinguish private differentiation and
installed topology from generic recurrent activity.

## Components

### `shuffled_private_expression`

Permutes gain/bias rows separately within compute and relay tissue. Tissue roles and
parameter distributions remain fixed; only the assignment of private identities to
cells changes.

### `degree_preserving_rewire`

Permutes installed source assignments while preserving every target's active slot
count and the global source-degree multiset. Output-target sources are shuffled only
among themselves, preserving the no-sensory-bypass contract.

### `zero_private_adapters`

Zeros all or selected private adapter up projections, removing cell-local transitions
without changing shared tissue rules, private expression, or anatomy.

### `disable_graft_edges`

Makes only a recorded additive graft dormant, then restores the exact active mask.

## Contracts

- Both interventions restore parameters/buffers even if evaluation raises.
- Adapter and graft interventions also restore exact tensors on exceptional exit.
- Neither intervention changes trainable values, optimizer state, or target degree.
- A penalty is evidence of causal dependence, not automatically evidence of useful
  capability; it is interpreted alongside normal held-out performance.
