# `hardware_preflight.py`

## Purpose

Measures the exact eager SOL2 procedural-training path on an unfamiliar CPU or CUDA
host before result-bearing jobs begin. It reports empirical latency, throughput, peak
PyTorch memory, and a software/hardware fingerprint rather than relying on advertised
low-precision inference TOPS.

## Components

### `run_hardware_preflight`

- Builds the requested typed organism and performs unreported warmup updates.
- Measures accepted updates and examples per second on the base topology.
- Optionally appends relay cells through the production growth operation, warms the new
  topology, and measures it separately.
- Records peak allocated/reserved CUDA memory and a conservative concurrency estimate.

### Hardware helpers

- `_device_fingerprint` records architecture, memory, compute capability, BF16 support,
  PyTorch/CUDA/Python versions, host architecture, and git revision.
- `_measure_phase` synchronizes CUDA around wall-clock measurement and resets peak
  allocator telemetry per phase.

## Contracts

- Uses float32 and the same `guarded_step`/procedural episode path as current S1 work.
- A preflight result is capacity evidence only; it is never a capability result.
- Rejected updates are visible and invalidate a host/configuration combination.
- The concurrency field is explicitly an allocator-based estimate. Separate processes
  must validate it before co-scheduling result arms.
- JSON output is atomic and contains the complete requested pre-growth configuration;
  append-only growth cannot rewrite the recorded baseline geometry.
- CPU execution remains supported for deterministic contract tests.
