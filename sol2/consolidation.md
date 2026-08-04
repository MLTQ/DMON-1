# `consolidation.py`

## Purpose

Measures which mature SOL2 cells and connections carry procedure A, then converts that
measurement into a graded energy cost on later parameter updates. Useful structure is
protected without requiring every cell to be occupied; low-utility cells remain a
plastic reserve for a new organ.

## Components

### `calibrate_causal_utility`

- Accumulates absolute held-out A gradients on private cell expression and edge-attention
  logits without updating weights or the living state.
- Combines private-expression demand, incoming edge demand, and downstream readers of a
  cell, then ranks utility within tissue so one tissue cannot monopolize protection by
  scale alone.
- Protects an edge when its own gradient, its source, or its target is useful.

### `make_utility_profile`

Builds three matched treatments: fully plastic, measured consolidation, and a
within-tissue shuffled-utility control. The shuffled arm preserves the distribution of
protection while breaking its relationship to learned function.

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
  within each tissue.
- The tissue genome (typed rules and shared graph operators) changes slowly but is not
  permanently frozen; attached-organ parameters remain freely plastic.
