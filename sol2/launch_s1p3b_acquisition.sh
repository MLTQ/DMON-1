#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <result-root>" >&2
  exit 2
fi

sol2_result_root="$1"
sol2_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
sol2_gpu_uuid="GPU-21d45575-7ece-a97c-35a0-294f7bce9c39"

cd "$sol2_repo_root"
exec env CUDA_VISIBLE_DEVICES="$sol2_gpu_uuid" /usr/bin/python3 \
  -m sol2.procedural_acquisition \
  --device cuda:0 \
  --out-dir "$sol2_result_root" \
  --seed 7 \
  --batch-size 24 \
  --max-updates 10000 \
  --evaluation-interval 1000 \
  --stage-updates 1000 \
  --eval-batches 64 \
  --mastery-accuracy 0.80 \
  --mastery-checks 2 \
  --operator-bound 4.0 \
  --n-memory 64 \
  --n-compute 256 \
  --n-relay 64 \
  --hidden 192 \
  --n-dendrites 16 \
  --initial-active-dendrites 12 \
  --steps-per-token 5 \
  --organ-queries 8 \
  --cell-adapter-rank 4 \
  --cell-adapter-gain 0.5 \
  --resume
