#!/usr/bin/env bash
# F6 round 2: B=12 (F0's token rate), inject 0.06, 12k updates.
# Preregistered re-dial in fable/experiments/f6-state-pressure.md
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/f6_1 logs
DEVICE="${DEVICE:-cuda:1}"
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 12 --chunk-length 32 \
     --updates 12000 --log-every 400 --inject-prob 0.06"
for seed in 7 13 21; do
  for kind in creature gru transformer; do
    out="fable/runs/f6_1/${kind}_s${seed}"
    log="logs/f61_${kind}_s${seed}.log"
    echo "[$(date -Is)] START f6_1/${kind}_s${seed}"
    python3 -u -m fable.recall --kind "$kind" --device "$DEVICE" \
      --out-dir "$out" --seed "$seed" $GEO >"$log" 2>&1
    echo "[$(date -Is)] DONE  f6_1/${kind}_s${seed} exit=$?"
  done
done
python3 -m fable.recall_report --root fable/runs/f6_1 --out fable/runs/f6_1/REPORT.md
echo "[$(date -Is)] F6.1 COMPLETE"
