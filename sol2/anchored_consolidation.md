# `anchored_consolidation.py`

## Purpose

Constrains cumulative displacement of mature A substrate with a checkpoint-centered
quadratic potential and measures whether B learning demand falls on anchored versus
accessible plastic structure.

## Components

### `AnchorProfile` / `make_anchor_profile`

- Builds plastic, uniform-anchor, measured-anchor, and developmental treatments.
- Uses causal cell and directional-edge utility from `consolidation.py`.
- Dormant anatomy and structure born after the A checkpoint have zero protection.

### `ProximalAnchorPolicy`

- Captures immutable mature-substrate anchors before B attachment.
- Applies `anchor + (updated-anchor)/(1 + rate*protection)` after accepted AdamW steps.
- Supports row-resized parameters after relay growth by anchoring only the original
  prefix; appended rows remain fully plastic.
- Reports protection-weighted gradient pressure and mean squared anchor displacement.

### `anchor_profile_summary`

Produces JSON-safe per-tissue protection and installed-edge summaries.

## Contracts

- A and B organ parameters, sensory parameters, and embeddings are never anchored.
- Plastic treatment is update-identical to ordinary accepted AdamW steps.
- Uniform and measured treatments use identical anchor rate and shared-genome strength.
- Adam normalization and weight decay cannot bypass the post-step proximal operator.
- Growth never changes an existing anchor or assigns protection to a new row.
- Pressure excludes organ gradients and includes new-row gradients in its accessible
  denominator.
