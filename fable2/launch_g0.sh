#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <result-root>" >&2
  exit 2
fi

fable2_result_root="$1"
fable2_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
fable2_gpu_uuid="GPU-21d45575-7ece-a97c-35a0-294f7bce9c39"

cd "$fable2_repo_root"
mkdir -p "$fable2_result_root"
exec env CUDA_VISIBLE_DEVICES="$fable2_gpu_uuid" /usr/bin/python3 \
  -m fable2.audit \
  --model Qwen/Qwen3.5-4B \
  --device cuda:0 \
  --dtype bfloat16 \
  --local-files-only \
  --corpus sol2/experiments/l0c1-wiki-memory-corpus.json \
  --out "$fable2_result_root/g0-audit.json"
