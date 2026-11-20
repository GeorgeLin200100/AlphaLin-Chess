#!/bin/bash
# 三进程自博弈：1 个 learner + 2 个 actor，各占一个核。
# 用法: bash scripts/run3.sh [小时数] [每步模拟次数]
set -u
HOURS=${1:-8}
SIMS=${2:-160}
cd "$(dirname "$0")/.." || exit 1
RUN=runs/main
QUEUE=$RUN/queue
MODEL=$RUN/model.npz
mkdir -p "$QUEUE" "$RUN/games"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python src/learner.py --run "$RUN" --queue "$QUEUE" --model "$MODEL" \
    --hours "$HOURS" --use-q --buffer 30000 --train-every 4 \
    --eval-every 5 --eval-games 20 --gate-every 20 --gate-games 20 &
LPID=$!
sleep 4

.venv/bin/python src/actor.py --id 1 --queue "$QUEUE" --run "$RUN" --model "$MODEL" \
    --hours "$HOURS" --sims "$SIMS" --seed 1 &
P1=$!
.venv/bin/python src/actor.py --id 2 --queue "$QUEUE" --run "$RUN" --model "$MODEL" \
    --hours "$HOURS" --sims "$SIMS" --seed 2 &
P2=$!

trap 'kill -TERM $LPID $P1 $P2 2>/dev/null' TERM INT
wait
