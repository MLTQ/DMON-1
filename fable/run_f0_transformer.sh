#!/usr/bin/env bash
# F0 amendment 2: the transformer control PROJECT.md's S0 condition requires.
# Runs on the 2070S (F2 is finished; F1 owns the 4090).
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p logs
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 12 --chunk-length 32"
for seed in 7 13 21; do
  echo "[$(date -Is)] START f0/s${seed} transformer"
  python3 -u -m fable.train --model transformer --device "${DEVICE:-cuda:1}" \
    --out-dir "fable/runs/f0/s${seed}" --seed "$seed" \
    $GEO --updates 8000 --eval-every 500 --log-every 100 \
    > "logs/f0_s${seed}_transformer.log" 2>&1
  echo "[$(date -Is)] DONE  f0/s${seed} transformer exit=$?"
done
python3 -m fable.summarize --root fable/runs/f0 --out fable/runs/f0/LADDER.md
