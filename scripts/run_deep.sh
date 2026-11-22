#!/bin/bash
# 实验：自博弈搜索深度 240 -> 480（其余不动，512 主干 + quiescence + lr 5e-4）。
# 评测仍固定 240 sims（和前几轮同一把尺子），每 40 分钟一点。
# 用法: bash scripts/run_deep.sh [小时数] [自博弈 sims] [评测间隔分钟]
set -u
HOURS=${1:-3}
SIMS=${2:-480}
EVERY=${3:-40}
cd "$(dirname "$0")/.." || exit 1
export POL_H1=512 POL_H2=512
RUN=runs/deep
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
    --hours "$GAI_HOURS" --sims 240 --use-q --lines 20 --start-delay 120 &

trap 'kill -TERM $LPID $P1 $P2 2>/dev/null' TERM INT
wait
