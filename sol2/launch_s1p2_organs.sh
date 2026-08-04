#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <control|full|organ_only|scratch> <result-root>" >&2
  exit 2
fi

sol2_branch="$1"
sol2_result_root="$2"
case "$sol2_branch" in
  control|full|organ_only|scratch) ;;
  *) echo "unknown branch: $sol2_branch" >&2; exit 2 ;;
esac

sol2_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
sol2_checkpoint="$sol2_repo_root/runs/s1p2-acquisition/large-s7/acquisition.pt"
sol2_gpu_uuid="GPU-21d45575-7ece-a97c-35a0-294f7bce9c39"

cd "$sol2_repo_root"
exec env CUDA_VISIBLE_DEVICES="$sol2_gpu_uuid" /usr/bin/python3 \
  -m sol2.organ_attachment \
  --device cuda:0 \
  --acquisition-checkpoint "$sol2_checkpoint" \
  --out-dir "$sol2_result_root/$sol2_branch" \
  --branch "$sol2_branch" \
  --adaptation-updates 2000 \
  --eval-every 200 \
  --eval-batches 16 \
  --final-eval-batches 64 \
  --max-steps 4 \
  --a-detached-updates 500 \
  --recovery-updates 250 \
  --resume
