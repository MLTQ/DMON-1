#!/usr/bin/env bash
# Chase #1: scale + no_credit + long horizon. Run on Aine from repo root.
set -euo pipefail
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT"
mkdir -p grok/runs/s0_chase1 logs

UPDATES="${UPDATES:-8000}"
# torch: cuda:0 = 4090, cuda:1 = 2070S

run_one() {
  local name="$1"
  local device="$2"
  shift 2
  local out="grok/runs/s0_chase1/${name}"
  local log="logs/chase1_${name}.log"
  echo "[$(date -Is)] START ${name} on ${device}"
  python3 -u -m grok.benchmark \
    --updates "$UPDATES" \
    --device "$device" \
    --out-dir "$out" \
    --no-credit \
    --log-every 100 \
    --eval-every 500 \
    "$@" \
    >"$log" 2>&1
  echo "[$(date -Is)] DONE  ${name} exit=$?"
}

# --- Wave 1: primary 128×128 no_credit seeds 7 + 13 ---
run_one s128_nc_s7 cuda:0 \
  --n-cells 128 --hidden 128 --n-dendrites 12 \
  --n-input 16 --n-output 16 --n-mirror 32 \
  --steps-per-token 4 --batch-size 12 --chunk-length 32 \
  --seed 7 --readout-mode mean &
PID7=$!

run_one s128_nc_s13 cuda:1 \
  --n-cells 128 --hidden 128 --n-dendrites 12 \
  --n-input 16 --n-output 16 --n-mirror 32 \
  --steps-per-token 4 --batch-size 8 --chunk-length 32 \
  --seed 13 --readout-mode mean &
PID13=$!

wait $PID7 $PID13

# --- Wave 2: seed 21 on 4090 + concat readout side ablation on 2070S (shorter) ---
run_one s128_nc_s21 cuda:0 \
  --n-cells 128 --hidden 128 --n-dendrites 12 \
  --n-input 16 --n-output 16 --n-mirror 32 \
  --steps-per-token 4 --batch-size 12 --chunk-length 32 \
  --seed 21 --readout-mode mean &
PID21=$!

# side ablation: 4k updates, same geometry, concat readout
(
  name=s128_nc_concat_s7
  out="grok/runs/s0_chase1/${name}"
  log="logs/chase1_${name}.log"
  echo "[$(date -Is)] START ${name} on cuda:1"
  python3 -u -m grok.benchmark \
    --updates 4000 \
    --device cuda:1 \
    --out-dir "$out" \
    --no-credit \
    --log-every 100 \
    --eval-every 500 \
    --n-cells 128 --hidden 128 --n-dendrites 12 \
    --n-input 16 --n-output 16 --n-mirror 32 \
    --steps-per-token 4 --batch-size 8 --chunk-length 32 \
    --seed 7 --readout-mode concat \
    >"$log" 2>&1
  echo "[$(date -Is)] DONE  ${name}"
) &
PIDC=$!

wait $PID21 $PIDC

# --- Wave 3: larger field seed 7 on 4090 ---
run_one s192_nc_s7 cuda:0 \
  --n-cells 192 --hidden 192 --n-dendrites 12 \
  --n-input 24 --n-output 24 --n-mirror 48 \
  --steps-per-token 4 --batch-size 6 --chunk-length 32 \
  --seed 7 --readout-mode mean

python3 -m grok.summarize_ladder --root grok/runs/s0_chase1 --out grok/runs/s0_chase1/LADDER.md
echo "[$(date -Is)] CHASE1 COMPLETE"
cat grok/runs/s0_chase1/LADDER.md
