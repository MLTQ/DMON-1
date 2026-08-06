# launch_g0.sh / launch_g1.sh

## Purpose

Aine launchers. Both pin the RTX 4090 by UUID
(`GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`) — never by CUDA ordinal, which orders
fastest-first and differs from `nvidia-smi` — and run frozen `Qwen/Qwen3.5-4B` in
BF16 with `--local-files-only` against the C1 wiki corpus.

## Order

`launch_g0.sh <result-root>` writes `g0-audit.json`; G1 may launch only after its
`gates_all_pass` is recorded true in the preregistration. `launch_g1.sh
<result-root> [updates] [eval-every]` resumes from `broca.pt` if present.

## Decision

The 2070S belongs to `jewels` and is never visible to these scripts.
