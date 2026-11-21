"""旁路评测：训练不停，定期拿最新权重去跟"贪心"和"Stockfish Skill 0"下一批。

和 learner 分开跑（nice 19，让位给训练），结果追加到 runs/<run>/gauntlet.jsonl。
用法: nice -n 19 .venv/bin/python src/watch_eval.py --run runs/main --every-min 20
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nn import Net
from evaluate import play_game
from xiangqi import RED, BLACK


def batch(net, opponent, games, sims, seed):
    rng = random.Random(seed)
    w = d = l = 0
    for i in range(games):
        side = RED if i % 2 == 0 else BLACK
        res = play_game(net, opponent, sims, side, rng)
        if res == 0:
            d += 1
        elif res == side:
            w += 1
        else:
            l += 1
    return {"win": w, "draw": d, "loss": l, "games": games,
            "score": round((w + 0.5 * d) / games, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--every-min", type=float, default=20)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--greedy-games", type=int, default=8)
    ap.add_argument("--sf-games", type=int, default=2)
    ap.add_argument("--hours", type=float, default=12)
    args = ap.parse_args()

    out_path = os.path.join(args.run, "gauntlet.jsonl")
    model = os.path.join(args.run, "model.npz")
    deadline = time.time() + args.hours * 3600
    seen_mtime = None
    while time.time() < deadline:
        if not os.path.exists(model):
            time.sleep(20)
            continue
        mt = os.path.getmtime(model)
        if mt == seen_mtime:
            time.sleep(30)
            continue
        seen_mtime = mt
        try:
            net = Net.load(model)
            g = batch(net, "greedy", args.greedy_games, args.sims, seed=11)
            s = batch(net, "sf", args.sf_games, args.sims, seed=12)
            row = {"t": time.strftime("%H:%M:%S"), "greedy": g, "sf": s}
            with open(out_path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{row['t']}] 贪心 {g['win']}/{g['draw']}/{g['loss']} 得分 {g['score']:.2f}"
                  f" | SF {s['win']}/{s['draw']}/{s['loss']}", flush=True)
        except Exception as e:
            print("评测失败:", e, flush=True)
        time.sleep(args.every_min * 60)


if __name__ == "__main__":
    main()
