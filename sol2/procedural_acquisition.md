# `procedural_acquisition.py`

## Purpose

Runs a resumable, mastery-gated acquisition phase before any organ attachment fork.
It compares organism geometries without allowing a partially learned length-four
procedure to masquerade as a transfer failure.

## Components

### `run_acquisition_calibration`

- **Does**: Admits composition lengths one through four in fixed 1,000-update stages,
  evaluates held-out length four at fixed intervals, and stops after repeated mastery
  or a hard update cap.
- **Interacts with**: `train_phase` and `evaluate_regime` in
  `procedural_benchmark.py`.
- **Rationale**: Curriculum depth is independent of total training duration, leaving
  sustained full-depth practice when the run is extended.

### `neuron_telemetry`

- **Does**: Records per-cell activation magnitude/variation, private-identity magnitude,
  private-adapter up-projection magnitude, tissue membership, installed in-degree, and
  active edge biases at every mastery check.
- **Rationale**: The visualizer should show measured differentiation and activity from
  real checkpoints rather than infer a decorative anatomy.

### Checkpoint helpers

- **Does**: Atomically preserve model, optimizer, living recurrent state, counters,
  records, evaluations, and RNG state after every interval.
- **Rationale**: A long calibration may resume without becoming a new organism.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU service manifests | `acquisition.pt` and `metrics.json` update every interval | Output names or non-atomic writes |
| Organ-attachment experiment | Checkpoint contains mastered model, optimizer, and living state | Removing process state |
| Neuron visualizer | Each evaluation contains `telemetry.cells` and `telemetry.edges` | Telemetry field names |
| Scaling comparison | Arms see the same staged task samples for a shared seed | Generator seed schedule |

## Decisions

- Acquisition exposes `--cell-adapter-rank` and `--cell-adapter-gain`; both are stored
  in the checkpoint configuration so attachment reconstructs identical geometry.

- Mastery defaults to at least 80% held-out length-four accuracy twice consecutively.
- The current and larger geometries receive identical examples and evaluation budgets.
- Larger capacity is promoted only if it is used and improves mastery efficiency; size
  is not itself a success metric.
- CPU and CUDA resume share device-independent RNG restoration from `checkpoint.py`.
