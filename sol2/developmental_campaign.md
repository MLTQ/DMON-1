# `developmental_campaign.py`

## Purpose

Generates a capacity-gated candidate campaign spanning multiple seeds, organism sizes,
and S1-P4 treatments. The file prepares reproducible work; it does not freeze a new
scientific claim before the seed-7 S1-P4 result is complete.

## Components

### `geometry_for_cells`

- Keeps eight input cells, 64 stream-memory cells, and eight output cells fixed.
- Allocates remaining cells to compute and relay tissue at a fixed 4:1 ratio.
- Scales each developmental event to four percent of starting cell count.

### `build_jobs`

- Adds one production-path capacity preflight per requested geometry.
- Makes every seed acquisition depend on its geometry preflight and require actual
  length-four mastery rather than mere checkpoint existence.
- Makes plastic, uniform-anchor, measured-anchor, and developmental arms depend on the
  exact mastered seed/geometry checkpoint.
- Records all budgets and treatment values explicitly in commands and protocol JSON.

## Contracts

- Default starting sizes are exactly 400, 800, 1,600, and 3,200 cells.
- Interface and stream-memory capacity remain fixed so the primary scaling axis is the
  recurrent compute/relay substrate.
- Installed dendrite degree and hidden width remain fixed as cell count grows.
- A failed/OOM capacity probe blocks that geometry before acquisition consumes compute.
- An acquisition that reaches its cap without mastery blocks all attachment arms.
- Manifests use portable `{python}`/`{root}` placeholders and contain no host GPU UUID.
- The generated protocol remains marked candidate until its claims and stopping rules
  are frozen after S1-P4 interpretation.
