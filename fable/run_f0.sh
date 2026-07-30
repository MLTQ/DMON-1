#!/usr/bin/env bash
# F0: S0 close-out — concat readout at full budget. Run on Aine from repo root.
# Preregistration: fable/experiments/f0-concat-close-out.md
#
# Robustness: no `set -e` — one failed arm must not kill the ladder (grok's
# chase-1 script would have aborted mid-wave on any failure and always
# reported exit=0; both fixed here).
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/f0 logs

UPDATES="${UPDATES:-8000}"
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 12 --chunk-length 32"
COMMON="$GEO --updates $UPDATES --eval-every 500 --log-every 100"

run_one() { # dir model device seed
  local dir="$1" model="$2" device="$3" seed="$4"
  local out="fable/runs/f0/${dir}"
  local log="logs/f0_${dir}_${model}.log"
  echo "[$(date -Is)] START f0/${dir} ${model} on ${device}"
  python3 -u -m fable.train --model "$model" --device "$device" \
    --out-dir "$out" --seed "$seed" $COMMON >"$log" 2>&1
  local code=$?
  echo "[$(date -Is)] DONE  f0/${dir} ${model} exit=${code}"
}

# 4090: the three creature arms, sequential, all at B=12 (chase-1 ran seed 13
# at B=8 on the small card — a 33% token-budget confound; not repeated here).
(
  run_one s7  creature cuda:0 7
  run_one s13 creature cuda:0 13
  run_one s21 creature cuda:0 21
) &
CREATURES=$!

# 2070S: matched GRUs (cheap) + the head-only bypass control.
(
  run_one s7  gru cuda:1 7
  run_one s13 gru cuda:1 13
  run_one s21 gru cuda:1 21
  run_one s7  bypass cuda:1 7
) &
CONTROLS=$!

wait "$CREATURES"
wait "$CONTROLS"

python3 -m fable.summarize --root fable/runs/f0 --out fable/runs/f0/LADDER.md
echo "[$(date -Is)] F0 COMPLETE"
cat fable/runs/f0/LADDER.md
