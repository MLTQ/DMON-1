#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 corpus|train <result-root> [updates] [eval-every]" >&2
  exit 2
fi

m0_stage="$1"
m0_result_root="$2"
m0_updates="${3:-300}"
m0_eval_every="${4:-25}"
m0_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
m0_gpu_uuid="GPU-21d45575-7ece-a97c-35a0-294f7bce9c39"
m0_corpus="$m0_repo_root/runs/fable2-m0/corpus.json"

cd "$m0_repo_root"
mkdir -p "$m0_result_root"
case "$m0_stage" in
  corpus)
    exec env CUDA_VISIBLE_DEVICES="$m0_gpu_uuid" /usr/bin/python3 \
      -m fable2.modes --local-files-only --out "$m0_corpus"
    ;;
  train)
    if [[ ! -f "$m0_corpus" ]]; then
      echo "M0 corpus not found at $m0_corpus — run '$0 corpus' first" >&2
      exit 3
    fi
    exec env CUDA_VISIBLE_DEVICES="$m0_gpu_uuid" /usr/bin/python3 \
      -m fable2.train_modes \
      --model Qwen/Qwen3.5-4B --device cuda:0 --dtype bfloat16 --local-files-only \
      --corpus "$m0_corpus" --out-dir "$m0_result_root" \
      --updates "$m0_updates" --eval-every "$m0_eval_every" \
      --checkpoint-every "$m0_eval_every" --resume
    ;;
  *)
    echo "unknown stage: $m0_stage" >&2
    exit 2
    ;;
esac
