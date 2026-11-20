"""三个模型横向对比：同一批对手、同样的局数、交替先后手。

用法: python src/compare.py --sims 120 --random-games 30 --greedy-games 10 --sf-games 4
输出: runs/compare.json + 控制台表格
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nn import Net
from evaluate import play_game


def gauntlet(path, sims, n_random, n_greedy, n_sf):
    net = Net.load(path)
    out = {}
    for name, opponent, games in (("random", "random", n_random),
                                  ("greedy", "greedy", n_greedy),
                                  ("sf", "sf", n_sf)):
        rng = random.Random(4242)
        w = d = l = 0
        for i in range(games):
            side = 1 if i % 2 == 0 else -1
            res = play_game(net, opponent, sims, side, rng)
            if res == 0:
                d += 1
            elif res == side:
                w += 1
            else:
                l += 1
        out[name] = {"win": w, "draw": d, "loss": l, "games": games,
                     "score": round((w + 0.5 * d) / games, 3)}
        print(f"  {os.path.basename(os.path.dirname(path)):8s} vs {name:7s}: "
              f"{w} 胜 {d} 和 {l} 负  得分 {out[name]['score']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--random-games", type=int, default=30)
    ap.add_argument("--greedy-games", type=int, default=10)
    ap.add_argument("--sf-games", type=int, default=4)
    args = ap.parse_args()

    models = [p for p in ("runs/night1/best.npz", "runs/night2/best.npz",
                          "runs/main/best.npz") if os.path.exists(p)]
    result = {}
    for path in models:
        print(f"[{path}]", flush=True)
        result[path] = gauntlet(path, args.sims, args.random_games,
                                args.greedy_games, args.sf_games)
    with open("runs/compare.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\n结果已写入 runs/compare.json")
