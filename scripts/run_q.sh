#!/bin/bash
# 实验：叶子吃子修正（quiescence，去 --no-q）+ 自博弈 sims 240，2 小时。
# 评测也开 --use-q，测的是实际出厂的搜索配置；开局集沿用 runs/main/openings.json。
# 用法: bash scripts/run_q.sh [小时数] [自博弈 sims] [评测间隔分钟]
set -u
HOURS=${1:-2}
SIMS=${2:-240}
EVERY=${3:-30}
cd "$(dirname "$0")/.." || exit 1
RUN=runs/q1
QUEUE=$RUN/queue
MODEL=$RUN/model.npz
mkdir -p "$QUEUE" "$RUN/games"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python src/learner.py --run "$RUN" --queue "$QUEUE" --model "$MODEL" \
    --hours "$HOURS" --resume --use-q --buffer 30000 --train-every 4 --lr 5e-4 --steps 20 \
    --eval-every 10 --eval-games 40 --gate-every 40 --gate-games 30 &
LPID=$!
sleep 6

.venv/bin/python src/actor.py --id 1 --queue "$QUEUE" --run "$RUN" --model "$MODEL" \
    --hours "$HOURS" --sims "$SIMS" --seed 1 --drop-adjudicated &
P1=$!
.venv/bin/python src/actor.py --id 2 --queue "$QUEUE" --run "$RUN" --model "$MODEL" \
    --hours "$HOURS" --sims "$SIMS" --seed 2 --drop-adjudicated &
P2=$!

GAI_HOURS=$(awk "BEGIN{print $HOURS+0.4}")
nice -n 19 .venv/bin/python src/gauntlet.py --run "$RUN" --every-min "$EVERY" \
    --hours "$GAI_HOURS" --sims "$SIMS" --use-q --lines 20 --start-delay 120 &
WPID=$!

trap 'kill -TERM $LPID $P1 $P2 $WPID 2>/dev/null' TERM INT
wait
