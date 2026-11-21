"""中象 AlphaZero 风格自博弈训练（单核）。

用法:
  python src/train.py --hours 7 --sims 120
  python src/train.py --smoke            # 2 分钟冒烟测试

产物 (runs/<id>/):
  model.npz / best.npz        最新权重 / 门控后的最佳权重
  stats.jsonl                 每轮指标（曲线数据源）
  status.json                 心跳（看板读它）
  games/*.json                抽样对局（看板回放）
  events.log                  人类可读日志
"""
import argparse, json, os, random, signal, sys, time, traceback
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xiangqi import (Board, START_FEN, encode_planes, encode_sparse, material,
                     move_to_index, mirror_move_index, index_to_move,
                     RED, BLACK, TYPE_CHARS, adjudicate, is_repetition)
from nn import Net, N_POL, N_IN
from mcts import search

TOP_K = 16


# ------------------------------------------------------------------ 落子/记录
def pick_move(visits, temperature, nprng):
    moves = list(visits.keys())
    counts = np.array([visits[m] for m in moves], dtype=np.float64)
    if temperature <= 1e-3 or len(moves) == 1:
        return moves[int(counts.argmax())]
    w = counts ** (1.0 / temperature)
    w /= w.sum()
    return moves[int(nprng.choice(len(moves), p=w))]


def play_self_game(net, sims, rng, nprng, noise=True, max_ply=160, use_q=False):
    board = Board(START_FEN)
    samples, moves_log = [], []
    while board.move_count < max_ply:
        res = board.result()
        if res is not None:
            return samples, moves_log, res, "checkmate"
        visits, _ = search(board, net, sims=sims,
                           dirichlet=0.3 if noise else 0.0, rng=nprng, use_q=use_q)
        if not visits:
            break
        temp = 1.0 if board.move_count < 30 else 0.05
        mv = pick_move(visits, temp, nprng)
        total = sum(visits.values())
        top = sorted(visits.items(), key=lambda kv: -kv[1])[:TOP_K]
        xs = encode_planes(board)
        samples.append((xs, [(move_to_index(f, t), n / total) for (f, t), n in top],
                        board.side, len(moves_log)))
        moves_log.append({"frm": mv[0], "to": mv[1],
                          "p": {f"{f},{t}": round(n / total, 3) for (f, t), n in top[:5]}})
        board.make(*mv)
    # 超步：按子力裁定（子力相同才判和）
    res = adjudicate(board)
    return samples, moves_log, res, ("adjudicated" if res else "draw")


# ------------------------------------------------------------------ 评估对局
def play_vs_random(net, sims, rng, net_side, use_q=False):
    board = Board(START_FEN)
    while True:
        res = board.result()
        if res is not None:
            return res
        if board.move_count >= 160:
            return adjudicate(board)
        if board.side == net_side:
            visits, _ = search(board, net, sims=sims, use_q=use_q, safety=True)
            if not visits:
                return -net_side
            mv = max(visits.items(), key=lambda kv: kv[1])[0]
        else:
            legal = board.legal_moves()
            if not legal:
                return -board.side
            mv = legal[rng.randrange(len(legal))]
        board.make(*mv)


def play_vs_model(net_a, net_b, sims, rng, a_side, use_q=False):
    """net_a 与 net_b 对局，返回 a 的结果 (1/0/-1)。"""
    board = Board(START_FEN)
    while True:
        res = board.result()
        if res is not None:
            return res if board.side != a_side else -res
        if board.move_count >= 160:
            return adjudicate(board)
        net = net_a if board.side == a_side else net_b
        visits, _ = search(board, net, sims=sims, use_q=use_q, safety=True)
        if not visits:
            return -board.side if board.side == a_side else board.side
        mv = max(visits.items(), key=lambda kv: kv[1])[0]
        board.make(*mv)


def gen_material_samples(n_positions=4000, seed=0):
    """随机局面 + 子力标签（既做价值热启动，也做训练时的"锚点"，防灾难性遗忘）。"""
    rng = random.Random(seed)
    X, Z = [], []
    guard = 0
    while len(X) < n_positions and guard < n_positions * 20:
        guard += 1
        b = Board(START_FEN)
        for _ in range(rng.randrange(4, 70)):
            legal = b.legal_moves()
            if not legal:
                break
            b.make(*legal[rng.randrange(len(legal))])
        if b.move_count < 4:
            continue
        d = material(b, RED) - material(b, BLACK)
        z = max(-1.0, min(1.0, d / 6.0))
        if b.side == BLACK:
            z = -z
        X.append(encode_planes(b))
        Z.append(z)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(Z, dtype=np.float32).reshape(-1, 1))


def material_warmstart(net, n_positions=4000, steps=300, batch=128, lr=2.5e-3, seed=0):
    """价值头热启动：先让它懂"多子=优"，给自博弈一个起点。"""
    X, Z = gen_material_samples(n_positions, seed)
    prng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        sub = prng.choice(len(X), min(batch, len(X)), replace=False)
        losses.append(net.train_value(X[sub], Z[sub], lr=lr))
    return losses


# ------------------------------------------------------------------ 训练器
class Trainer:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.nprng = np.random.default_rng(args.seed)
        self.net = Net(args.seed)
        self.best = Net(args.seed)
        self.buffer = deque(maxlen=args.buffer)
        self.iter = 0
        self.games = 0
        self.elo = 1000.0
        self.snapshots = []
        self.t0 = time.time()

        os.makedirs(args.run, exist_ok=True)
        os.makedirs(os.path.join(args.run, "games"), exist_ok=True)
        self.log_path = os.path.join(args.run, "events.log")
        if args.warmstart:
            t0 = time.time()
            losses = material_warmstart(self.net, steps=args.warmstart, seed=args.seed)
            self.log(f"热启动: {args.warmstart} 步价值训练, loss {losses[0]:.3f} -> {losses[-1]:.3f}"
                     f" ({time.time()-t0:.0f}s)")
            for k in self.net.p:
                self.best.p[k] = self.net.p[k].copy()
        self.anchor_X, self.anchor_Z = gen_material_samples(3000, args.seed + 1)
        # 断点续训：--resume 时从已有权重接着跑
        if getattr(args, "resume", False):
            mp = os.path.join(args.run, "model.npz")
            if os.path.exists(mp):
                self.net = Net.load(mp)
                self.log(f"resume from {mp}")

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    # ---------------------------------------------------------- 训练步
    def train(self):
        if len(self.buffer) < 200:
            return None
        a = self.args
        X = np.empty((a.batch, N_IN), dtype=np.float32)
        PI = np.zeros((a.batch, N_POL), dtype=np.float32)
        Z = np.empty((a.batch, 1), dtype=np.float32)
        PM = np.ones((a.batch, 1), dtype=np.float32)
        n = len(self.buffer)
        n_anchor = int(a.batch * a.anchor_prob)
        losses = []
        for _ in range(a.steps):
            idxs = self.nprng.integers(0, n, size=a.batch - n_anchor)
            PI[:] = 0.0
            PM[:] = 1.0
            for j, k in enumerate(idxs):
                xs, pi, z = self.buffer[k]
                X[j] = xs
                for mi, p in pi:
                    PI[j, mi] = p
                Z[j, 0] = z
            if n_anchor:
                ai = self.nprng.integers(0, len(self.anchor_X), n_anchor)
                X[a.batch - n_anchor:] = self.anchor_X[ai]
                Z[a.batch - n_anchor:, 0] = self.anchor_Z[ai, 0]
                PM[a.batch - n_anchor:] = 0.0        # 锚点只训练价值，不算 policy loss
            # top-k 截断后概率和 < 1，必须重新归一化，否则交叉熵目标不是分布（会不稳定）
            ssum = PI.sum(axis=1, keepdims=True)
            PI /= np.maximum(ssum, 1e-9)
            lp, lv = self.net.train_step(X, PI, Z, lr=a.lr,
                                         policy_weight=a.policy_weight, pmask=PM)
            losses.append((lp, lv))
        lp = float(np.mean([x[0] for x in losses]))
        lv = float(np.mean([x[1] for x in losses]))
        return lp, lv

    # ---------------------------------------------------------- 评估
    def eval_vs_random(self, n_games):
        wins = draws = 0
        for i in range(n_games):
            side = RED if i % 2 == 0 else BLACK
            r = play_vs_random(self.net, self.args.eval_sims, self.rng, side, use_q=self.args.use_q)
            wins += 1 if r == side else 0
            draws += 1 if r == 0 else 0
        return wins / n_games, draws / n_games

    def eval_vs_best(self, n_games):
        """挑战者(self.net) vs 卫冕者(self.best)，返回挑战者胜率（和棋算半分）。"""
        score = 0.0
        for i in range(n_games):
            side = RED if i % 2 == 0 else BLACK
            r = play_vs_model(self.net, self.best, self.args.eval_sims, self.rng, side, use_q=self.args.use_q)
            score += (r + 1) / 2 if r != 0 else 0.5
        wr = score / n_games
        # 简单 Elo 更新：期望胜率 = 1/(1+10^((Rb-Ra)/400))，这里 Rb 视为当前 elo
        self.elo += 24 * (wr - 0.5) * 2
        return wr

    # ---------------------------------------------------------- 主循环
    def run(self):
        _install_signal_save(self)
        a = self.args
        deadline = time.time() + a.hours * 3600
        self.log(f"start: sims={a.sims} hours={a.hours} run={a.run}")
        while time.time() < deadline:
            self.iter += 1
            t_start = time.time()
            it_samples, it_games, results = 0, 0, {"red": 0, "black": 0, "draw": 0}
            for _ in range(a.games_per_iter):
                samples, moves_log, res, tag = play_self_game(self.net, a.sims, self.rng, self.nprng, use_q=a.use_q)
                it_games += 1
                self.games += 1
                if not samples:
                    continue
                if res == 0:
                    z = 0.0
                    results["draw"] += 1
                else:
                    z = 1.0 if res == RED else -1.0
                    results["red" if res == RED else "black"] += 1
                for xs, pi, side, _ply in samples:
                    zz = z if side == RED else -z     # 以"该样本走子方"视角
                    xs8 = xs.astype(np.uint8)
                    self.buffer.append((xs8, pi, zz))
                    it_samples += 1
                    # 镜像增强
                    n_pl = len(xs8) // 90
                    xs_m = xs8.reshape(n_pl, 10, 9)[:, :, ::-1].reshape(-1).copy()
                    pi_m = [(mirror_move_index(mi), p) for mi, p in pi]
                    self.buffer.append((xs_m, pi_m, zz))
                    it_samples += 1
                if self.games % 40 == 0:
                    self.save_game(moves_log, res, tag)

            sp_time = time.time() - t_start
            loss = self.train()
            tr_time = time.time() - t_start - sp_time

            row = {"iter": self.iter, "games": self.games, "samples": it_samples,
                   "selfplay_s": round(sp_time, 1), "train_s": round(tr_time, 1),
                   "results": results, "loss_p": None, "loss_v": None,
                   "winrate_random": None, "winrate_best": None, "elo": round(self.elo, 1)}

            if loss:
                row["loss_p"], row["loss_v"] = round(loss[0], 4), round(loss[1], 4)

            if self.iter % a.eval_every == 0:
                wr, dr = self.eval_vs_random(a.eval_games)
                row["winrate_random"], row["draw_random"] = round(wr, 3), round(dr, 3)
                self.log(f"iter {self.iter}: games={self.games} vs-random win={wr:.2f} draw={dr:.2f} elo={self.elo:.0f}")

            if self.iter % a.gate_every == 0 and self.iter > 1:
                wr = self.eval_vs_best(a.gate_games)
                row["winrate_best"] = round(wr, 3)
                if wr >= 0.55:
                    for k in self.net.p:
                        self.best.p[k] = self.net.p[k].copy()
                    self.log(f"iter {self.iter}: 门控通过 (vs best {wr:.2f}) -> 新 best, elo~{self.elo:.0f}")
                else:
                    self.log(f"iter {self.iter}: 门控未过 (vs best {wr:.2f})")
                self.best.save(os.path.join(a.run, "best.npz"))

            self.write_status(row)
            with open(os.path.join(a.run, "stats.jsonl"), "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if self.iter % a.ckpt_every == 0:
                self.net.save(os.path.join(a.run, "model.npz"))
                self.log(f"iter {self.iter}: ckpt saved (games={self.games}, buffer={len(self.buffer)})")

        self.net.save(os.path.join(a.run, "model.npz"))
        self.best.save(os.path.join(a.run, "best.npz"))
        self.log(f"done: iters={self.iter} games={self.games} elo={self.elo:.0f}")

    def save_game(self, moves_log, res, tag):
        path = os.path.join(self.args.run, "games", f"g{self.games:06d}.json")
        with open(path, "w") as f:
            json.dump({"moves": moves_log, "result": res, "tag": tag,
                       "iter": self.iter, "games": self.games}, f, ensure_ascii=False)

    def write_status(self, row):
        status = dict(row)
        status["elapsed_min"] = round((time.time() - self.t0) / 60, 1)
        status["buffer"] = len(self.buffer)
        status["pid"] = os.getpid()
        with open(os.path.join(self.args.run, "status.json"), "w") as f:
            json.dump(status, f, ensure_ascii=False, indent=1)


def _install_signal_save(tr):
    def handler(signum, frame):
        tr.log(f"收到信号 {signum}，保存后退出")
        try:
            tr.net.save(os.path.join(tr.args.run, "model.npz"))
            tr.best.save(os.path.join(tr.args.run, "best.npz"))
        finally:
            raise SystemExit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "..", "runs", "main"))
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--eval-sims", type=int, default=60)
    ap.add_argument("--games-per-iter", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--buffer", type=int, default=120000)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--gate-every", type=int, default=40)
    ap.add_argument("--gate-games", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmstart", type=int, default=300)
    ap.add_argument("--policy-weight", type=float, default=0.1)
    ap.add_argument("--anchor-prob", type=float, default=0.3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use-q", action="store_true", help="叶子静息评估（吃子修正）")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.run = os.path.abspath(args.run)

    if args.smoke:
        args.hours = 0.02
        args.sims = 24
        args.eval_sims = 12
        args.games_per_iter = 1
        args.steps = 2
        args.buffer = 4000
        args.eval_every = 2
        args.eval_games = 2
        args.gate_every = 3
        args.gate_games = 2
        args.ckpt_every = 2

    Trainer(args).run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
