#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 7 ]]; then
  echo "usage: $0 <plastic|uniform|consolidated|shuffled> <genome-rate> <result-root> [updates] [eval-every] [eval-batches] [final-eval-batches]" >&2
  exit 2
fi

sol2_branch="$1"
sol2_genome_rate="$2"
sol2_result_root="$3"
sol2_updates="${4:-3000}"
sol2_eval_every="${5:-300}"
sol2_eval_batches="${6:-16}"
sol2_final_eval_batches="${7:-64}"
case "$sol2_branch" in
  plastic|uniform|consolidated|shuffled) ;;
  *) echo "unknown branch: $sol2_branch" >&2; exit 2 ;;
esac

sol2_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
sol2_checkpoint="$sol2_repo_root/runs/s1p3b-acquisition/private-r4-s7/acquisition.pt"
sol2_gpu_uuid="GPU-21d45575-7ece-a97c-35a0-294f7bce9c39"

cd "$sol2_repo_root"
exec env CUDA_VISIBLE_DEVICES="$sol2_gpu_uuid" /usr/bin/python3 \
  -m sol2.consolidated_attachment \
  --device cuda:0 \
  --acquisition-checkpoint "$sol2_checkpoint" \
  --out-dir "$sol2_result_root/$sol2_branch" \
  --branch "$sol2_branch" \
  --adaptation-updates "$sol2_updates" \
  --eval-every "$sol2_eval_every" \
  --eval-batches "$sol2_eval_batches" \
  --final-eval-batches "$sol2_final_eval_batches" \
  --utility-batches 64 \
  --max-steps 4 \
  --threshold 0.65 \
  --temperature 0.10 \
  --minimum-plasticity 0.02 \
  --genome-plasticity "$sol2_genome_rate" \
  --directional-edges \
  --reserve-growth \
  --reserve-target-fraction 0.5 \
  --reserve-slots-per-target 2 \
  --reserve-edge-raw -1.5 \
  --max-graft-a-drop 0.05 \
  --resume
