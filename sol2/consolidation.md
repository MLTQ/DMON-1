# `consolidation.py`

## Purpose

Measures which mature SOL2 cells and connections carry procedure A, then converts that
measurement into a graded energy cost on later parameter updates. Useful structure is
protected without requiring every cell to be occupied; low-utility cells remain a
plastic reserve for a new organ.

## Components

### `calibrate_causal_utility`

- Accumulates held-out A gradients on private identity, private low-rank adapters, and
  edge-attention logits without updating weights or the living state.
- Combines private-expression demand, incoming edge demand, and downstream readers of a
  cell, then ranks utility within tissue so one tissue cannot monopolize protection by
  scale alone.
- Ranks identity, adapter-down, and adapter-up demand separately before averaging the
  private factor, preventing matrix-size or parameterization scale from silently
  suppressing one form of cell-local computation.
- In the S1-P3-compatible mode, protects an edge when its own gradient, source, or
  target is useful. The directional mode protects it only for its own gradient or its
  target's utility, allowing a useful source to feed new reserve computation without
  freezing every outgoing connection.

### `make_utility_profile`

Builds four treatments: fully plastic, uniform protection, measured consolidation, and
a within-tissue shuffled-utility control. Uniform protection gives every cell and every
installed edge the respective mean measured plasticity, isolating placement from the
global amount of update suppression. The shuffled arm preserves the full distribution
of protection while breaking its relationship to learned function.

### `ConsolidationPolicy`

Scales the realized AdamW delta after the optimizer step. This is intentionally not a
gradient multiplier: Adam's normalization can cancel a persistent scalar on gradients,
and decoupled weight decay can alter a parameter even when its gradient is protected.
The post-step policy makes both sources of modification pay the same energy cost while
leaving optimizer moments intact.

## Contracts

- Calibration never changes model parameters, optimizer state, or the supplied living
  state.
- Dormant dendrites remain fully plastic and receive no fabricated utility.
- The plastic branch is update-identical to ordinary AdamW.
- Consolidated and shuffled arms have the same utility and plasticity distributions
  within each tissue; uniform matches their global mean cell and installed-edge
  plasticity.
- Private identity and private-adapter rows pay the same cell-specific update cost.
- The tissue genome (typed rules and shared graph operators) changes slowly but is not
  permanently frozen; attached-organ parameters remain freely plastic.
