#!/usr/bin/env bash
# E1: schedule isolation — constant LR at F0 geometry/batch.
# Preregistration: fable/experiments/e1-constant-lr.md
# Runs on the 4090 alongside F7 (both launch-bound; contention disclosed).
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/e1_constlr logs
DEVICE="${DEVICE:-cuda:0}"
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 12 --chunk-length 32 \
     --updates 8000 --lr 3e-3 --lr-min 3e-3 --warmup-updates 200 \
     --eval-every 500 --log-every 100"
for seed in 7 13 21; do
  for model in creature gru; do
    out="fable/runs/e1_constlr/s${seed}"
    log="logs/e1_s${seed}_${model}.log"
    echo "[$(date -Is)] START e1/s${seed} ${model}"
    python3 -u -m fable.train --model "$model" --device "$DEVICE" \
      --out-dir "$out" --seed "$seed" $GEO >"$log" 2>&1
    echo "[$(date -Is)] DONE  e1/s${seed} ${model} exit=$?"
  done
done
python3 -m fable.summarize --root fable/runs/e1_constlr --out fable/runs/e1_constlr/LADDER.md
echo "[$(date -Is)] E1 COMPLETE"
cat fable/runs/e1_constlr/LADDER.md
