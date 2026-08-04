#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "usage: $0 <plastic|uniform_anchor|measured_anchor|developmental> <result-root> [updates] [eval-every] [eval-batches] [final-eval-batches]" >&2
  exit 2
fi

sol2_branch="$1"
sol2_result_root="$2"
sol2_updates="${3:-3000}"
sol2_eval_every="${4:-300}"
sol2_eval_batches="${5:-16}"
sol2_final_eval_batches="${6:-64}"
case "$sol2_branch" in
  plastic|uniform_anchor|measured_anchor|developmental) ;;
  *) echo "unknown branch: $sol2_branch" >&2; exit 2 ;;
esac

sol2_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
sol2_checkpoint="$sol2_repo_root/runs/s1p3b-acquisition/private-r4-s7/acquisition.pt"
sol2_gpu_uuid="GPU-21d45575-7ece-a97c-35a0-294f7bce9c39"

cd "$sol2_repo_root"
exec env CUDA_VISIBLE_DEVICES="$sol2_gpu_uuid" /usr/bin/python3 \
  -m sol2.developmental_attachment \
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
  --anchor-rate 0.01 \
  --growth-cells 16 \
  --high-pressure 0.75 \
  --plateau-pressure 0.60 \
  --plateau-gain 0.03 \
  --patience-checks 2 \
  --growth-min-update 600 \
  --growth-refractory 600 \
  --max-growth-events 2 \
  --max-growth-a-drop 0.05 \
  --resume
