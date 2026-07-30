#!/usr/bin/env bash
# F1: runtime growth. Run on Aine from repo root, AFTER the F0 verdict.
# Preregistration: fable/experiments/f1-growth.md
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/f1 logs

UPDATES="${UPDATES:-8000}"
GEO="--hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 --n-mirror 32 \
     --steps-per-token 4 --batch-size 12 --chunk-length 32"
COMMON="$GEO --updates $UPDATES --eval-every 500 --log-every 100"

run_arm() { # arm device seed
  local arm="$1" device="$2" seed="$3"
  local out="fable/runs/f1/${arm}_s${seed}"
  local log="logs/f1_${arm}_s${seed}.log"
  echo "[$(date -Is)] START f1/${arm}_s${seed} on ${device}"
  python3 -u -m fable.grow --arm "$arm" --device "$device" \
    --out-dir "$out" --seed "$seed" $COMMON >"$log" 2>&1
  echo "[$(date -Is)] DONE  f1/${arm}_s${seed} exit=$?"
}

# born arms are F0's creature runs (identical config, seeds, and stream);
# copy their results in rather than burning GPU hours on byte-identical runs.
for seed in 7 13 21; do
  if [ -f "fable/runs/f0/s${seed}/creature.json" ]; then
    mkdir -p "fable/runs/f1/born_s${seed}"
    cp "fable/runs/f0/s${seed}/creature.json" "fable/runs/f1/born_s${seed}/creature.json"
    echo "born_s${seed}: copied from f0 (disclosed in LADDER by identical numbers)"
  fi
done

# All arms sequential on the 4090: the 2070S is occupied by F2's queue, and
# mixed-device arms would confound the comparison anyway (sol S13 lost its
# causal attribution exactly that way).
DEV="${DEV:-cuda:0}"
for seed in 7 13 21; do
  run_arm grown "$DEV" "$seed"
  run_arm small "$DEV" "$seed"
done

python3 -m fable.summarize --root fable/runs/f1 --out fable/runs/f1/LADDER.md
echo "[$(date -Is)] F1 COMPLETE"
cat fable/runs/f1/LADDER.md
