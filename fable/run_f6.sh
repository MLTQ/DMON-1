#!/usr/bin/env bash
# F6: state pressure. Runs on the 2070S after the transformer controls drain;
# the 4090 keeps F1. Preregistration: fable/experiments/f6-state-pressure.md
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/f6 logs

DEVICE="${DEVICE:-cuda:1}"
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 4 --chunk-length 32 \
     --updates 8000 --log-every 200"

while pgrep -f "fable.train.*cuda:1" > /dev/null; do
  echo "[$(date -Is)] waiting for cuda:1"; sleep 120
done

run_arm() { # kind seed
  local kind="$1" seed="$2"
  local out="fable/runs/f6/${kind}_s${seed}"
  local log="logs/f6_${kind}_s${seed}.log"
  echo "[$(date -Is)] START f6/${kind}_s${seed}"
  python3 -u -m fable.recall --kind "$kind" --device "$DEVICE" \
    --out-dir "$out" --seed "$seed" $GEO >"$log" 2>&1
  echo "[$(date -Is)] DONE  f6/${kind}_s${seed} exit=$?"
}

for seed in 7 13 21; do
  run_arm creature "$seed"
  run_arm gru "$seed"
  run_arm transformer "$seed"
done

python3 -m fable.recall_report --root fable/runs/f6 --out fable/runs/f6/REPORT.md
echo "[$(date -Is)] F6 COMPLETE"
