#!/bin/bash
# 八小时轮：自博弈 480 次模拟、3 个 actor、手数上限对齐评测的 160 手。
# 评测固定 240 sims / 20 个固定开局 / 40 局，每 60 分钟一点（评测本身要约 20 分钟）。
# 用法: bash scripts/run_deep8.sh [小时数] [sims] [评测间隔分钟] [手数上限] [actor 数]
set -u
HOURS=${1:-8}
SIMS=${2:-480}
EVERY=${3:-60}
MAXPLY=${4:-160}
NACT=${5:-3}
cd "$(dirname "$0")/.." || exit 1
export POL_H1=512 POL_H2=512
RUN=${RUN:-runs/deep8}
LR=${LR:-5e-4}
QUEUE=$RUN/queue
MODEL=$RUN/model.npz
mkdir -p "$QUEUE" "$RUN/games"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python src/learner.py --run "$RUN" --queue "$QUEUE" --model "$MODEL" \
    --hours "$HOURS" --resume --use-q --buffer 30000 --train-every 4 --lr "$LR" --steps 20 \
    --eval-every 10 --eval-games 20 --gate-every 40 --gate-games 30 &
LPID=$!
sleep 6

PIDS=""
for i in $(seq 1 "$NACT"); do
    .venv/bin/python src/actor.py --id "$i" --queue "$QUEUE" --run "$RUN" --model "$MODEL" \
        --hours "$HOURS" --sims "$SIMS" --max-ply "$MAXPLY" --seed "$i" --drop-adjudicated &
    PIDS="$PIDS $!"
done

GAI_HOURS=$(awk "BEGIN{print $HOURS+0.4}")
nice -n 19 .venv/bin/python src/gauntlet.py --run "$RUN" --every-min "$EVERY" \
    --hours "$GAI_HOURS" --sims 240 --use-q --lines 20 --start-delay 180 &

trap 'kill -TERM $LPID $PIDS 2>/dev/null' TERM INT
wait
