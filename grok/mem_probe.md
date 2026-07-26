# mem_probe.py

## Purpose
One-shot CUDA memory probe for choosing batch/truncate settings on Aine before long runs.

## Notes
- BPTT memory scales roughly with `batch × cells × hidden × dendrites × steps_per_token × truncate_every`.
- On the 2070S (8GB): B=16, T=64, spt=4 OOM; B=16, T=32, spt=3 ~fits.
