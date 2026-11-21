"""自博弈 worker：只负责下棋并把样本写进队列目录，不做训练。

多个 worker 共用 learner 发布的权重（runs/<id>/model.npz），按 mtime 热加载。
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nn import Net
from selfplay import play_self_game, save_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-q", action="store_true")
    ap.add_argument("--temp-plies", type=int, default=60)
    ap.add_argument("--drop-adjudicated", action="store_true", default=False)
    args = ap.parse_args()

    os.makedirs(args.queue, exist_ok=True)
    os.makedirs(os.path.join(args.run, "games"), exist_ok=True)
    rng = random.Random(args.seed * 1000 + args.id)
    nprng = np.random.default_rng(args.seed * 1000 + args.id)

    # 等 learner 发布第一版权重
    deadline = time.time() + args.hours * 3600
    while not os.path.exists(args.model):
        if time.time() > deadline:
            return
        time.sleep(1)

    net, mtime = None, None
    games, skipped, t_start = 0, 0, time.time()
    last_games_save = 0
    while time.time() < deadline:
        mt = os.path.getmtime(args.model)
        if mt != mtime:
            try:
                net = Net.load(args.model)
                mtime = mt
            except Exception:
                time.sleep(0.5)
                continue
        payload, meta = play_self_game(net, args.sims, rng, nprng,
                                       use_q=not args.no_q,
                                       temp_plies=args.temp_plies)
        if payload is None:
            continue
        games += 1
        if meta.get("adjudicated"):
            skipped += 1          # 只统计，不再丢弃
        gid = f"a{args.id}_{int(time.time()*1000)}_{games}"
        save_game(args.queue, gid, payload, meta)

        # 每 40 局存一份可回放的对局
        if games - last_games_save >= 40:
            last_games_save = games
            try:
                path = os.path.join(args.run, "games", f"{gid}.json")
                with open(path, "w") as f:
                    json.dump({"moves": meta["moves"], "result": meta["result"],
                               "tag": "selfplay", "iter": 0, "games": games}, f,
                              ensure_ascii=False)
            except Exception:
                pass

        elapsed = max(time.time() - t_start, 1)
        with open(os.path.join(args.queue, f"status_a{args.id}.json"), "w") as f:
            json.dump({"id": args.id, "games": games, "skipped": skipped,
                       "games_per_hour": round(games / elapsed * 3600, 1),
                       "elapsed_min": round(elapsed / 60, 1),
                       "pid": os.getpid()}, f)

    with open(os.path.join(args.queue, f"status_a{args.id}.json"), "w") as f:
        json.dump({"id": args.id, "games": games, "done": True, "pid": os.getpid()}, f)


if __name__ == "__main__":
    main()
