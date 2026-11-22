"""对训练产物做一次可靠的棋力评估（多局、固定对手、交替先后手）。

用法: python src/evaluate.py [--sims 120] [--games 12] [--model runs/main/best.npz]
对手:
  random     纯随机合法着法
  greedy     一步贪心：优先吃价值最高的子
  sf          Fairy-Stockfish（Skill Level 0，固定 20ms/步），UCI 走子
"""
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiangqi import Board, START_FEN, PIECE_VALUE, adjudicate, RED, BLACK, TYPE_CHARS
from nn import Net
from mcts import search

SF = "/usr/games/fairy-stockfish"


def move_to_iccs(frm, to):
    """内部坐标(0..89) -> Fairy-Stockfish 坐标(a1..i10)。rank 1 = 红方底线。"""
    fx, fy = frm % 9, frm // 9
    tx, ty = to % 9, to // 9
    return f"{chr(97+fx)}{fy+1}{chr(97+tx)}{ty+1}"


class UciEngine:
    def __init__(self, path=SF, skill=0, movetime=20, variant="xiangqi"):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self._cmd("uci")
        self._cmd(f"setoption name UCI_Variant value {variant}")
        self._cmd(f"setoption name Skill Level value {skill}")
        self._cmd("setoption name Threads value 1")
        self._cmd("setoption name Hash value 16")
        self._cmd("isready")
        self.movetime = movetime

    def _cmd(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()
        if s == "isready":
            while True:
                line = self.p.stdout.readline().strip()
                if line == "readyok":
                    return

    def bestmove(self, moves_iccs):
        pos = "position startpos" + ((" moves " + " ".join(moves_iccs)) if moves_iccs else "")
        self._cmd(pos)
        self.p.stdin.write(f"go movetime {self.movetime}\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline().strip()
            if line.startswith("bestmove"):
                return line.split()[1]

    def close(self):
        try:
            self._cmd("quit")
        except Exception:
            pass
        self.p.kill()


def iccs_to_move(s):
    """解析 a1..i10 形式的着法（rank 可能是两位数）。非法/空着返回 None。"""
    if not s or not s[0].isalpha() or (len(s) > 1 and not s[1].isdigit()):
        return None
    fx = ord(s[0]) - 97
    i = 1
    while s[i].isdigit():
        i += 1
    fy = int(s[1:i]) - 1
    tx = ord(s[i]) - 97
    ty = int(s[i+1:]) - 1
    return fy * 9 + fx, ty * 9 + tx


def greedy_move(board, rng):
    best, best_score = None, -1e9
    me = board.side
    for f, t in board.legal_moves():
        q = board.sq[t]
        gain = PIECE_VALUE[q[0]] if q else 0.0
        # 避免送子：走完后看自己这个子会不会被白吃
        board.make(f, t)
        risk = 0.0
        if board.attacked(t, -me):
            risk = PIECE_VALUE[board.sq[t][0]] * 0.6
        board.unmake()
        score = gain - risk + rng.random() * 0.05
        if score > best_score:
            best, best_score = (f, t), score
    return best


def play_game(net, opponent, sims, net_side, rng, max_ply=160, start_moves=None, use_q=False):
    board = Board(START_FEN)
    iccs_hist = []
    eng = None
    if start_moves:
        for mv in start_moves:
            iccs_hist.append(move_to_iccs(*mv))
            board.make(*mv)
    if opponent == "sf":
        eng = UciEngine()
    while True:
        res = board.result()
        if res is not None:
            break
        if board.move_count >= max_ply:
            res = adjudicate(board)
            break
        if board.side == net_side:
            visits, _ = search(board, net, sims=sims, safety=True, use_q=use_q)
            if not visits:
                res = -board.side
                break
            mv = max(visits.items(), key=lambda kv: kv[1])[0]
        elif opponent == "random":
            legal = board.legal_moves()
            if not legal:
                res = -board.side
                break
            mv = legal[rng.randrange(len(legal))]
        elif opponent == "greedy":
            mv = greedy_move(board, rng)
        else:
            best = eng.bestmove(iccs_hist)
            mv = iccs_to_move(best)
            if mv is None:          # 引擎认输/给出空着，按对手获胜处理
                res = -board.side
                break
            if mv not in board.legal_moves():   # 坐标解析对不上，判引擎负
                res = -board.side
                break
        iccs_hist.append(move_to_iccs(*mv))
        board.make(*mv)
    if eng:
        eng.close()
    return res


def run(model_path, opponent, sims, games, label):
    net = Net.load(model_path)
    rng = random.Random(1234)
    score = 0.0
    w = d = l = 0
    t0 = time.time()
    for i in range(games):
        side = RED if i % 2 == 0 else BLACK
        res = play_game(net, opponent, sims, side, rng)
        if res == 0:
            d += 1
            score += 0.5
        elif res == side:
            w += 1
            score += 1
        else:
            l += 1
    print(f"{label}: {games} 局 -> 胜 {w} / 和 {d} / 负 {l}  得分率 {score/games:.2f}  "
          f"({time.time()-t0:.0f}s, {sims} sims)", flush=True)
    return score / games


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--model", default=os.path.join(HERE, "..", "runs", "main", "best.npz"))
    ap.add_argument("--model2", default=os.path.join(HERE, "..", "runs", "main", "model.npz"))
    args = ap.parse_args()

    out = {}
    out["vs_random_best"] = run(args.model, "random", args.sims, args.games, "best  vs random")
    out["vs_greedy_best"] = run(args.model, "greedy", args.sims, max(6, args.games // 2), "best  vs greedy")
    out["vs_sf_best"] = run(args.model, "sf", args.sims, max(4, args.games // 3), "best  vs SF(skill0,20ms)")
    if os.path.exists(args.model2):
        out["vs_random_latest"] = run(args.model2, "random", args.sims, args.games, "最新  vs random")
    with open(os.path.join(HERE, "..", "runs", "main", "eval_result.json"), "w") as f:
        import json
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
