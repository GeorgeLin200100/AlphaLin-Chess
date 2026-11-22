"""搜索到底加不加值：同一份权重，三种选点方式各下 40 局（同期同开局、红黑各一遍）。

  search240  现在的正式配置：240 次模拟 + 安全过滤 + 叶子吃子修正
  policy     完全不搜，直接取网络策略概率最高的着法
  policy+sf  不搜，但先用安全过滤把"走完会被白吃"的着法剔掉

如果 search240 和 policy 得分差不多，说明搜索没产生比网络更好的目标，
自博弈就是在自我模仿，天花板锁在网络本身的估值上。

用法: .venv/bin/python scripts/arm_compare.py --ckpt runs/wide/model.npz --games 40
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/wide/model.npz")
    ap.add_argument("--openings", default="runs/main/openings.json")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=240)
    ap.add_argument("--max-ply", type=int, default=160)
    ap.add_argument("--sweep", default="",
                    help="只跑搜索臂，按这批 sims 逐个测，例如 60,120,240,480")
    args = ap.parse_args()

    with np.load(args.ckpt) as d:
        w = d["W1p"].shape[1]
    os.environ["POL_H1"] = os.environ["POL_H2"] = str(w)
    from nn import Net  # noqa: E402
    from mcts import search  # noqa: E402
    from evaluate import greedy_move  # noqa: E402
    from xiangqi import (Board, START_FEN, adjudicate, is_repetition,  # noqa: E402
                         move_to_index, feature_indices, RED, BLACK)
    from mcts import NO_CAPTURE_DRAW  # noqa: E402
    import quiescence  # noqa: E402

    net = Net.load(args.ckpt)
    lines = json.load(open(args.openings))["lines"]
    n_lines = min(len(lines), max(1, args.games // 2))

    def policy_probs(b):
        legal = b.legal_moves()
        idx = [move_to_index(f, t) for f, t in legal]
        probs, _ = net.evaluate_sparse(feature_indices(b), idx)
        return legal, probs

    fallback = {"n": 0}

    def make_pick_search(sims):
        def pick(b):
            visits, _ = search(b, net, sims=sims, safety=True, use_q=True)
            if not visits:            # 低 sims 时安全过滤可能把根节点候选清空
                fallback["n"] += 1
                return pick_policy_sf(b)
            return max(visits.items(), key=lambda kv: kv[1])[0]
        return pick

    pick_search = make_pick_search(args.sims)

    def pick_policy(b):
        legal, probs = policy_probs(b)
        return legal[int(probs.argmax())]

    def pick_policy_sf(b):
        legal, probs = policy_probs(b)
        keep = [(legal[i], probs[i]) for i in range(len(legal))
                if quiescence.forced_loss(b, legal[i]) < 1.0]
        if len(keep) < 3:
            keep = list(zip(legal, probs))
        return max(keep, key=lambda kv: kv[1])[0]

    def play(select, line, side, rng):
        b = Board(START_FEN)
        for mv in line:
            b.make(*mv)
        while True:
            res = b.result()
            if res is not None:
                return res
            if b.move_count >= args.max_ply:
                return adjudicate(b)
            if is_repetition(b):
                return 0
            if b.move_count - b.last_capture >= NO_CAPTURE_DRAW:
                return adjudicate(b)
            mv = select(b) if b.side == side else greedy_move(b, rng)
            b.make(*mv)

    print(f"权重 {args.ckpt}（宽 {w}），{n_lines} 个开局、每点 {2 * n_lines} 局，"
          f"对手一步贪心", flush=True)
    results = {}
    if args.sweep:
        arms = [(f"search{s}", make_pick_search(int(s))) for s in args.sweep.split(",")]
    else:
        arms = [("search240", pick_search), ("policy", pick_policy),
                ("policy+sf", pick_policy_sf)]
    for name, select in arms:
        w_ = d_ = l_ = 0
        t0 = time.time()
        for oi, line in enumerate(lines[:n_lines]):
            moves = [tuple(m) for m in line]
            for side in (RED, BLACK):
                rng = random.Random(1013 * oi + (0 if side == RED else 1) + 7)
                res = play(select, moves, side, rng)
                if res == 0:
                    d_ += 1
                elif res == side:
                    w_ += 1
                else:
                    l_ += 1
        n = w_ + d_ + l_
        score = (w_ + 0.5 * d_) / n
        results[name] = dict(win=w_, draw=d_, loss=l_, games=n, score=round(score, 3),
                             search_empty_fallbacks=fallback["n"])
        print(f"{name:10s} 胜/和/负 {w_}/{d_}/{l_}  得分 {score:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = os.path.join(HERE, "..", "reports", "arm_compare.json")
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "sims": args.sims, "results": results},
                  f, ensure_ascii=False, indent=1)
    print("写盘:", out)


if __name__ == "__main__":
    main()
