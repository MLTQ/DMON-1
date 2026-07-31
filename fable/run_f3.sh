#!/usr/bin/env bash
# F3: liquid cell head-to-head. Preregistration: fable/experiments/f3-liquid-cell.md
# Waits for E1b to drain the 4090, then six 8k arms in parallel, then L-H (24k).
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/f3 logs
DEVICE="${DEVICE:-cuda:0}"
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 12 --chunk-length 32 \
     --cell-rule liquid --eval-every 500 --log-every 100"

while pgrep -f "e1b_const1e3" > /dev/null; do
  echo "[$(date -Is)] waiting for E1b on cuda:0"; sleep 120
done

launch() { # name extra-args...
  local name="$1"; shift
  python3 -u -m fable.train --model creature --device "$DEVICE" \
    --out-dir "fable/runs/f3/${name}" $GEO "$@" \
    > "logs/f3_${name}.log" 2>&1 &
  echo "[$(date -Is)] START f3/${name}"
}

# Wave 1: annealed (L-A) + constant 3e-3 (L-C), all seeds, parallel
launch la_s7  --seed 7  --updates 8000
launch la_s13 --seed 13 --updates 8000
launch la_s21 --seed 21 --updates 8000
launch lc_s7  --seed 7  --updates 8000 --lr 3e-3 --lr-min 3e-3
launch lc_s13 --seed 13 --updates 8000 --lr 3e-3 --lr-min 3e-3
launch lc_s21 --seed 21 --updates 8000 --lr 3e-3 --lr-min 3e-3
wait
echo "[$(date -Is)] wave 1 done"

# Wave 2: the F7 cliff (24k horizon), solo
launch lh_s7 --seed 7 --updates 24000 --eval-every 1000 --log-every 200
wait

python3 -m fable.summarize --root fable/runs/f3 --out fable/runs/f3/LADDER.md
echo "[$(date -Is)] F3 COMPLETE"
cat fable/runs/f3/LADDER.md
