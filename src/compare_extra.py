"""补全对比数据：night3 全量 + 前两版的 SF 局 + 版本间对局。"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nn import Net
from evaluate import play_game
from xiangqi import Board, START_FEN, adjudicate, RED, BLACK
from mcts import search


def gauntlet(path, sims, n_random, n_greedy, n_sf):
    net = Net.load(path)
    out = {}
    for name, opponent, games in (("random", "random", n_random),
                                  ("greedy", "greedy", n_greedy),
                                  ("sf", "sf", n_sf)):
        if games <= 0:
            continue
        rng = random.Random(4242)
        w = d = l = 0
        t0 = time.time()
        for i in range(games):
            side = RED if i % 2 == 0 else BLACK
            res = play_game(net, opponent, sims, side, rng)
            if res == 0:
                d += 1
            elif res == side:
                w += 1
            else:
                l += 1
        out[name] = {"win": w, "draw": d, "loss": l, "games": games,
                     "score": round((w + 0.5 * d) / games, 3)}
        print(f"  {path:24s} vs {name:6s}: {w} 胜 {d} 和 {l} 负  得分 {out[name]['score']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


def head_to_head(a_path, b_path, sims, games):
    a = Net.load(a_path)
    b = Net.load(b_path)
    rng = random.Random(777)
    score = 0.0
    w = d = l = 0
    for i in range(games):
        a_side = RED if i % 2 == 0 else BLACK
        board = Board(START_FEN)
        while True:
            res = board.result()
            if res is not None:
                break
            if board.move_count >= 160:
                res = adjudicate(board)
                break
            net = a if board.side == a_side else b
            visits, _ = search(board, net, sims=sims)
            if not visits:
                res = -board.side
                break
            mv = max(visits.items(), key=lambda kv: kv[1])[0]
            board.make(*mv)
        if res == 0:
            score += 0.5
            d += 1
        elif res == a_side:
            score += 1
            w += 1
        else:
            l += 1
    print(f"  {a_path} 对 {b_path}: {w} 胜 {d} 和 {l} 负  得分 {score/games:.2f}", flush=True)
    return {"win": w, "draw": d, "loss": l, "games": games, "score": round(score / games, 3)}


if __name__ == "__main__":
    res = {}
    print("[night3 全量]", flush=True)
    res["night3"] = gauntlet("runs/main/best.npz", 120, 30, 10, 4)

    print("[补 SF 局]", flush=True)
    res["night1_sf"] = gauntlet("runs/night1/best.npz", 120, 0, 0, 4)
    res["night2_sf"] = gauntlet("runs/night2/best.npz", 120, 0, 0, 4)

    print("[版本间对局]", flush=True)
    res["h2h_3v2"] = head_to_head("runs/main/best.npz", "runs/night2/best.npz", 120, 20)
    res["h2h_3v1"] = head_to_head("runs/main/best.npz", "runs/night1/best.npz", 120, 20)

    with open("runs/compare_extra.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("done")
