#!/usr/bin/env bash
# F2: adaptability under regime cycling. Run on Aine from repo root.
# Preregistration: fable/experiments/f2-adaptability.md
# Designed to share the box with F0: everything here runs on the 2070S
# (cuda:1) so the 4090 keeps the F0 creature queue.
set -u
ROOT="${ROOT:-$HOME/Code/DMON-1}"
cd "$ROOT" || exit 1
mkdir -p fable/runs/f2 logs

DEVICE="${DEVICE:-cuda:1}"
UPDATES="${UPDATES:-8000}"
BLOCK="${BLOCK:-24576}"
# Constant LR after warmup for EVERY arm (lr == lr-min): an annealed learner
# cannot stay plastic — see the preregistration for why this deviates from
# F0's cosine.
GEO="--n-cells 128 --hidden 128 --n-dendrites 12 --n-input 16 --n-output 16 \
     --n-mirror 32 --steps-per-token 4 --batch-size 4 --chunk-length 32 \
     --lr 1e-3 --lr-min 1e-3 --warmup-updates 200"
COMMON="$GEO --updates $UPDATES --block $BLOCK --log-every 200 --device $DEVICE"

# Wait for the F0 control queue to release the 2070S.
while pgrep -f "fable.train.*cuda:1" > /dev/null; do
  echo "[$(date -Is)] waiting for cuda:1 to free up"; sleep 120
done

run_arm() { # kind stream seed
  local kind="$1" stream="$2" seed="$3"
  local out="fable/runs/f2/${stream}_${kind}_s${seed}"
  local log="logs/f2_${stream}_${kind}_s${seed}.log"
  echo "[$(date -Is)] START f2/${stream}_${kind}_s${seed}"
  python3 -u -m fable.adapt --kind "$kind" --stream "$stream" \
    --out-dir "$out" --seed "$seed" $COMMON >"$log" 2>&1
  echo "[$(date -Is)] DONE  f2/${stream}_${kind}_s${seed} exit=$?"
}

for seed in 7 13 21; do
  run_arm gru cycled "$seed"
  run_arm gru aonly "$seed"
  run_arm creature cycled "$seed"
  run_arm creature aonly "$seed"
done

echo "[$(date -Is)] F2 COMPLETE — analyses in fable/runs/f2/*/analysis.json"
