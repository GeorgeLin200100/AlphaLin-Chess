#!/bin/bash
# 实验：策略主干 256x2 -> 512x2（widen.py 加宽，其余全不动）。
# 起点权重来自上一轮 166 轮的网络，价值主干逐位相同，初始策略策略分布偏移很小。
# 用法: bash scripts/run_wide.sh [小时数] [自博弈 sims] [评测间隔分钟]
set -u
HOURS=${1:-2.5}
SIMS=${2:-240}
EVERY=${3:-30}
cd "$(dirname "$0")/.." || exit 1
export POL_H1=512 POL_H2=512
RUN=runs/wide
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

GAI_HOURS=$(awk "BEGIN{print $HOURS+0.35}")
nice -n 19 .venv/bin/python src/gauntlet.py --run "$RUN" --every-min "$EVERY" \
    --hours "$GAI_HOURS" --sims "$SIMS" --use-q --lines 20 --start-delay 120 &

trap 'kill -TERM $LPID $P1 $P2 2>/dev/null' TERM INT
wait
