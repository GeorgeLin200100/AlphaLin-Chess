#!/bin/bash
# 三小时测量轮：续训（learner --resume）+ 2 个 actor + 固定开局评测。
# 用法: bash scripts/run_measure.sh [小时数] [每步模拟次数] [评测间隔分钟]
set -u
HOURS=${1:-3}
SIMS=${2:-160}
EVERY=${3:-30}
cd "$(dirname "$0")/.." || exit 1
RUN=runs/main
QUEUE=$RUN/queue
MODEL=$RUN/model.npz
mkdir -p "$QUEUE" "$RUN/games"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python src/learner.py --run "$RUN" --queue "$QUEUE" --model "$MODEL" \
    --hours "$HOURS" --resume --buffer 30000 --train-every 4 --lr 5e-4 --steps 20 \
    --eval-every 10 --eval-games 40 --gate-every 40 --gate-games 30 &
LPID=$!
sleep 6

.venv/bin/python src/actor.py --id 1 --queue "$QUEUE" --run "$RUN" --model "$MODEL" \
    --hours "$HOURS" --sims "$SIMS" --seed 1 --no-q --drop-adjudicated &
P1=$!
.venv/bin/python src/actor.py --id 2 --queue "$QUEUE" --run "$RUN" --model "$MODEL" \
    --hours "$HOURS" --sims "$SIMS" --seed 2 --no-q --drop-adjudicated &
P2=$!

nice -n 19 .venv/bin/python src/gauntlet.py --run "$RUN" --every-min "$EVERY" \
    --hours "$HOURS" --sims 120 --lines 20 --start-delay 60 &
WPID=$!

trap 'kill -TERM $LPID $P1 $P2 $WPID 2>/dev/null' TERM INT
wait
