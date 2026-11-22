"""目标自洽性体检：同一个局面搜两次，目标本身差多少。

为什么要做：如果两次独立搜索给出的推进分布差别很大，那么"网络拟合目标"这件事
就有一个不可消除的地板。loss 卡住之前，先量这个地板有多高。

做法：从最近一局自博弈里均匀取 N 个局面，每个局面用两个不同的 Dirichlet 噪声种子
各搜 240 次，比较两次的 top1 / top5 / KL（在两次支撑集的并集上归一后算）。
再顺手把"网络对这份目标的 KL"放在同一张表里对比。

用法: .venv/bin/python scripts/diag_target.py --run runs/wide --n 40 --sims 240
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from xiangqi import Board, START_FEN, move_to_index  # noqa: E402


def positions_from_game(path, n):
    d = json.load(open(path))
    moves = [(m["frm"], m["to"]) for m in d["moves"]]
    boards = []
    b = Board(START_FEN)
    for mv in moves:
        if b.result() is not None:
            break
        boards.append(b.copy())
        b.make(*mv)
    if len(boards) <= n:
        return boards
    idx = np.linspace(0, len(boards) - 1, n).astype(int)
    return [boards[i] for i in idx]


def target_from_visits(visits, top_k=16):
    total = sum(visits.values())
    top = sorted(visits.items(), key=lambda kv: -kv[1])[:top_k]
    return {move_to_index(f, t): n / total for (f, t), n in top}


def kl_union(a, b):
    keys = set(a) | set(b)
    pa = np.array([a.get(k, 0.0) for k in keys], dtype=np.float64)
    pb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float64)
    pa = pa / max(pa.sum(), 1e-12)
    pb = pb / max(pb.sum(), 1e-12)
    return float((pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/wide")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sims", type=int, default=240)
    ap.add_argument("--dirichlet", type=float, default=0.3,
                    help="根部探索噪声强度，0 = 关掉")
    args = ap.parse_args()
    model = args.model or os.path.join(args.run, "model.npz")

    with np.load(model) as d:
        w = d["W1p"].shape[1]
    os.environ["POL_H1"] = os.environ["POL_H2"] = str(w)
    from nn import Net  # noqa: E402
    from mcts import search  # noqa: E402

    net = Net.load(model)
    games = sorted(glob.glob(os.path.join(args.run, "games", "*.json")))
    if not games:
        raise SystemExit("找不到对局文件")
    boards = positions_from_game(games[-1], args.n)
    print(f"模型 {model}（宽 {w}），对局 {os.path.basename(games[-1])}，取 {len(boards)} 个局面，"
          f"每次 {args.sims} sims")

    top1 = []
    top5 = []
    kls = []
    for i, b in enumerate(boards):
        r1 = np.random.default_rng(1000 + i)
        r2 = np.random.default_rng(9000 + i)
        v1, _ = search(b, net, sims=args.sims, dirichlet=args.dirichlet, rng=r1, use_q=True, safety=True)
        v2, _ = search(b, net, sims=args.sims, dirichlet=args.dirichlet, rng=r2, use_q=True, safety=True)
        if not v1 or not v2:
            continue
        t1, t2 = target_from_visits(v1), target_from_visits(v2)
        k1 = max(t1.items(), key=lambda kv: kv[1])[0]
        k2 = max(t2.items(), key=lambda kv: kv[1])[0]
        top1.append(1.0 if k1 == k2 else 0.0)
        s5_1 = set(k for k, _ in sorted(t1.items(), key=lambda kv: -kv[1])[:5])
        s5_2 = set(k for k, _ in sorted(t2.items(), key=lambda kv: -kv[1])[:5])
        top5.append(len(s5_1 & s5_2) / 5.0)
        kls.append(kl_union(t1, t2))

    print(f"同一局面搜两次（只换噪声种子，{len(top1)} 个局面）：")
    print(f"  top1 一致率      {np.mean(top1):.1%}")
    print(f"  top5 重合率      {np.mean(top5):.1%}")
    print(f"  两次目标间 KL    {np.mean(kls):.3f} nats（单向）")


if __name__ == "__main__":
    main()
