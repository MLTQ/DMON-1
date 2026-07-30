#!/usr/bin/env bash
# F7: horizon calibration — 24k updates, seed 7, three arms on the 4090
# after F1 drains. Preregistration: fable/experiments/f7-horizon.md
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/f7 logs

DEVICE="${DEVICE:-cuda:0}"
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 12 --chunk-length 32"

while pgrep -f "fable.grow" > /dev/null || pgrep -f "fable.train.*cuda:0" > /dev/null; do
  echo "[$(date -Is)] waiting for cuda:0"; sleep 180
done

for model in creature gru transformer; do
  out="fable/runs/f7/s7"
  log="logs/f7_s7_${model}.log"
  echo "[$(date -Is)] START f7/s7 ${model}"
  python3 -u -m fable.train --model "$model" --device "$DEVICE" \
    --out-dir "$out" --seed 7 $GEO --updates 24000 \
    --eval-every 1000 --log-every 200 >"$log" 2>&1
  echo "[$(date -Is)] DONE  f7/s7 ${model} exit=$?"
done

python3 -m fable.summarize --root fable/runs/f7 --out fable/runs/f7/LADDER.md
echo "[$(date -Is)] F7 COMPLETE"
