# `tissues.py`

## Purpose

Implements typed neural tissue. SOL2 shares a rule within a tissue, not across every
cell in the organism, and allows private cell expression on top in `model.py`.

## Components

### `TissueRule`

- Concatenates current state, incoming message, and external drive.
- Integrates toward a `tanh`-bounded target.
- Learns a channel-wise update rate bounded to `[0.02, 0.80]`.
- Uses the same bounded/unbounded operator treatment as the connectome.

### `TissueBank`

Owns separate input, compute, relay, and output rules. Initial time constants reflect
their roles but remain learnable.

## Contracts

- If mutable state starts in `[-1, 1]`, every tissue update remains in `[-1, 1]`.
- Tissue rules have separate parameters and may evolve independently.
- Memory tissue is not represented here because only the external stream writes it.
- Operator norms are exposed for the health log.
