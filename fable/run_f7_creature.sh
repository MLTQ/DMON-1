#!/usr/bin/env bash
# F7 creature rerun with GradGuard homeostat (first attempt died at u9622:
# finite 1.5e19 gradients destroyed the model before inf appeared).
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p logs
echo "[$(date -Is)] START f7/s7 creature (guarded rerun)"
python3 -u -m fable.train --model creature --device cuda:0 \
  --out-dir fable/runs/f7/s7 --seed 7 \
  --n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
  --n-mirror 32 --steps-per-token 4 --batch-size 12 --chunk-length 32 \
  --updates 24000 --eval-every 1000 --log-every 200 \
  > logs/f7_s7_creature_guarded.log 2>&1
echo "[$(date -Is)] DONE  f7/s7 creature exit=$?"
python3 -m fable.summarize --root fable/runs/f7 --out fable/runs/f7/LADDER.md
echo "[$(date -Is)] F7 CREATURE RERUN COMPLETE"
