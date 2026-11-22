"""固定开局棋力评测：每个评测点都下同一批开局、红黑各一遍，逐局记录。

目的：把评测噪声压下来，让"曲线有没有上升"变成可判断的结论。
- 开局集固定（runs/<run>/openings.json，默认 20 条 4 步安静开局）
- 每条开局红黑各下一局（默认 40 局/点，分辨率 0.025）
- 逐局结果写盘，后续可用同开局配对检验（src/trend.py）
- 对手固定为一步贪心（确定性，除少量 tie-break 噪声，且按开局索引定种子）

用法:
  nice -n 19 .venv/bin/python src/gauntlet.py --run runs/main --every-min 30 --hours 3
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
from xiangqi import Board, START_FEN, RED, BLACK


def make_openings(n_lines=20, plies=4, seed=2024):
    """随机生成一批'安静'开局（只走不吃子的着法），固定种子、去重。"""
    rng = random.Random(seed)
    lines, seen = [], set()
    guard = 0
    while len(lines) < n_lines and guard < 20000:
        guard += 1
        b = Board(START_FEN)
        mv_list = []
        ok = True
        for _ in range(plies):
            cand = [(f, t) for f, t in b.legal_moves() if b.sq[t] is None]
            if not cand:
                ok = False
                break
            mv = cand[rng.randrange(len(cand))]
            b.make(*mv)
            mv_list.append([mv[0], mv[1]])
        if not ok or b.result() is not None:
            continue
        key = tuple(tuple(m) for m in mv_list)
        if key in seen:
            continue
        seen.add(key)
        lines.append(mv_list)
    return lines


def load_or_make_openings(run_dir, n_lines, plies):
    path = os.path.join(run_dir, "openings.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["lines"], path
    lines = make_openings(n_lines=n_lines, plies=plies)
    with open(path, "w") as f:
        json.dump({"plies": plies, "lines": lines}, f, ensure_ascii=False, indent=1)
    return lines, path


def eval_point(net, lines, sims, max_ply=160, use_q=False):
    rows = []
    w = d = l = 0
    for oi, line in enumerate(lines):
        moves = [tuple(m) for m in line]
        for side in (RED, BLACK):
            rng = random.Random(1013 * oi + (0 if side == RED else 1) + 7)
            res = play_game(net, "greedy", sims, side, rng,
                            max_ply=max_ply, start_moves=moves, use_q=use_q)
            if res == 0:
                d += 1
            elif res == side:
                w += 1
            else:
                l += 1
            rows.append({"o": oi, "side": side, "res": res})
    n = len(rows)
    return {"win": w, "draw": d, "loss": l, "games": n,
            "score": round((w + 0.5 * d) / n, 4), "per_game": rows}


def last_iter(run_dir):
    p = os.path.join(run_dir, "stats.jsonl")
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4000))
            tail = f.read().decode("utf-8", "replace").strip().splitlines()
        row = json.loads(tail[-1])
        return row.get("iter"), row.get("games")
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--every-min", type=float, default=30)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--lines", type=int, default=20,
                    help="开局条数，每条约红黑各一局")
    ap.add_argument("--plies", type=int, default=4)
    ap.add_argument("--hours", type=float, default=3)
    ap.add_argument("--use-q", action="store_true", default=False,
                    help="评测时也开启叶子吃子修正（测的是实际出厂的搜索配置）")
    ap.add_argument("--start-delay", type=float, default=20,
                    help="等 learner 发布权重后再评（秒）")
    args = ap.parse_args()

    model = os.path.join(args.run, "model.npz")
    out_path = os.path.join(args.run, "gauntlet2.jsonl")
    lines, op_path = load_or_make_openings(args.run, args.lines, args.plies)
    print(f"开局集 {len(lines)} 条 <- {op_path}；每点 {2 * len(lines)} 局，"
          f"{args.sims} sims", flush=True)

    deadline = time.time() + args.hours * 3600
    first = True
    while time.time() < deadline:
        if not os.path.exists(model):
            time.sleep(10)
            continue
        time.sleep(args.start_delay if first else 0)
        first = False
        try:
            net = Net.load(model)
            t0 = time.time()
            g = eval_point(net, lines, args.sims, use_q=args.use_q)
            it, sp_games = last_iter(args.run)
            row = {"t": time.strftime("%H:%M:%S"), "ts": round(time.time(), 1),
                   "iter": it, "selfplay_games": sp_games,
                   "minutes": round((time.time() - t0) / 60, 2),
                   "opponent": "greedy", "sims": args.sims,
                   "use_q": bool(args.use_q), **g}
            with open(out_path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{row['t']}] iter {it} 自博弈 {sp_games} 局 -> "
                  f"胜/和/负 {g['win']}/{g['draw']}/{g['loss']} 得分 {g['score']:.3f} "
                  f"(评测 {row['minutes']:.1f} min)", flush=True)
        except Exception as e:
            print("评测失败:", repr(e), flush=True)
        remain = deadline - time.time()
        if remain <= 0:
            break
        time.sleep(min(args.every_min * 60, remain))


if __name__ == "__main__":
    main()
