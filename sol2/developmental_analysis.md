# `developmental_analysis.py`

## Purpose

Aggregates the four matched S1-P4 branches while keeping anchor attribution,
developmental benefit, causal recruitment, and simultaneous capability as separate
claims.

## Components

### `analyze`

- Reads plastic, uniform-anchor, measured-anchor, and developmental terminal artifacts.
- Computes pressure and growth summaries plus frozen behavioral contrasts.
- Measures causal grown-cell and grown-adapter penalties on B and checks whether removing
  growth merely rescues A from harmful interference.

## Contracts

- Capability requires one anchored arm at A/B >=80%.
- Measured placement and developmental growth each require a 10-point worst-organ gain
  over their direct control.
- Causal growth requires at least a 10-point B penalty when born cells are frozen.
- Decimal threshold comparisons include a machine-scale epsilon so an encoded
  difference of exactly 10 points is not rejected by binary floating-point rounding.
- A positive result also requires zero rejected updates and byte-identical A organs.
- Failure of one gate remains localized; the analyzer emits no universal architecture
  conclusion.
