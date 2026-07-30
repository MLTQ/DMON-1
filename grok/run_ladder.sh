#!/usr/bin/env bash
# Launch S0 gap ladder on Aine. Usage from repo root on the training host.
set -euo pipefail
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT"
mkdir -p grok/runs/s0_gap logs

# torch: cuda:0 = 4090, cuda:1 = 2070S
UPDATES="${UPDATES:-2000}"

run_one() {
  local name="$1"
  local device="$2"
  shift 2
  local out="grok/runs/s0_gap/${name}"
  local log="logs/s0_gap_${name}.log"
  echo "[$(date -Is)] START ${name} on ${device} -> ${out}"
  python3 -u -m grok.benchmark \
    --updates "$UPDATES" \
    --device "$device" \
    --out-dir "$out" \
    "$@" \
    >"$log" 2>&1
  echo "[$(date -Is)] DONE  ${name}"
}

# Wave 1: parallel on both GPUs
run_one baseline_s7 cuda:1 \
  --n-cells 64 --hidden 64 --n-dendrites 8 --steps-per-token 3 --seed 7 &
PID_BASE=$!

run_one scale_s7 cuda:0 \
  --n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 --n-mirror 32 \
  --steps-per-token 4 --seed 7 --batch-size 12 --chunk-length 32 &
PID_SCALE=$!

wait $PID_BASE $PID_SCALE

# Wave 2: credit ablation on 2070S, deep recurrence on 4090
run_one no_credit_s7 cuda:1 \
  --n-cells 64 --hidden 64 --n-dendrites 8 --steps-per-token 3 --seed 7 --no-credit &
PID_NC=$!

run_one deep_s7 cuda:0 \
  --n-cells 64 --hidden 64 --n-dendrites 12 --steps-per-token 6 --seed 7 &
PID_DEEP=$!

wait $PID_NC $PID_DEEP

# Wave 3: multi-seed for baseline + best-looking scale arm (always run scale seed13)
run_one baseline_s13 cuda:1 \
  --n-cells 64 --hidden 64 --n-dendrites 8 --steps-per-token 3 --seed 13 &
PID_B13=$!

run_one scale_s13 cuda:0 \
  --n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 --n-mirror 32 \
  --steps-per-token 4 --seed 13 --batch-size 12 &
PID_S13=$!

wait $PID_B13 $PID_S13

python3 -m grok.summarize_ladder --root grok/runs/s0_gap --out grok/runs/s0_gap/LADDER.md
echo "[$(date -Is)] LADDER COMPLETE"
cat grok/runs/s0_gap/LADDER.md
