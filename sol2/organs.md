# `organs.py`

## Purpose

Defines external effectors as attentive interfaces over dedicated output tissue. An
organ can change its tissue size without exposing or resizing the internal substrate.
Sensor/effector bundles can also be attached without replacing an existing organ.

## Components

### `OrganHealth`

Reports attention entropy, maximum allocation, and the entropy-equivalent number of
participating output cells.

### `AttentiveOutputOrgan`

- Owns a small bank of learned organ queries.
- Cross-attends only to the output-cell tensor supplied by `Sol2`.
- Uses damp-only bounded query/key/value handling in the bounded treatment.
- Decodes fixed query slots rather than fixed cell positions.
- Uses finite-gain normalization (`eps=1e-2`) so a silent/new organ is not amplified
  hundreds of times before it has learned an internal path.

### `SensorEffectorOrgan`

- Owns an independent token embedding, input-cell sensory affine parameters, attentive
  output reader, and decoder.
- Produces the bounded memory write and sensory drive expected by `Sol2.step`.
- Can be removed and reattached as one module without exposing the shared substrate.

## Decisions

- With a fixed small query bank, ordinary cross-attention is already linear in output
  tissue size and is simpler to audit than kernelized linear attention.
- The organ cannot address the whole organism. Sparse anatomy must first deliver
  information into its dedicated output tissue.

## Contracts

- The input contains output cells only; bypass access is not part of the API.
- Attention statistics reveal collapse onto one cell or permanently uniform reading.
- Operator norms participate in the same health telemetry as graph and tissue rules.
- Each attached sensor/effector bundle owns disjoint parameters; organ selection is an
  explicit `Sol2.step` argument rather than an inferred token convention.
