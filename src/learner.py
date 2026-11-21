"""learner：消费 actor 写下的样本，训练网络，发布权重。

与 actor 通过目录解耦：actor 写 queue/*.npz（先写 .tmp 再 rename），
learner 只读新文件，不删。
"""
import argparse
import glob
import json
import os
import random
import signal
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nn import Net, N_POL, N_IN
from xiangqi import (Board, START_FEN, encode_planes, material,
                     adjudicate, RED, BLACK)
from mcts import search
from train import gen_material_samples, material_warmstart, play_vs_random, play_vs_model


def load_queue_file(path):
    d = np.load(path)
    X, pi_idx, pi_p, Z = d["X"], d["pi_idx"], d["pi_p"], d["Z"]
    VW = d["VW"] if "VW" in d else np.ones(len(X), dtype=np.float32)
    Xm, pim = d["Xm"], d["pim"]
    out = []
    for i in range(len(X)):
        pi = [(m, p) for m, p in zip(pi_idx[i].tolist(), pi_p[i].tolist()) if m >= 0]
        out.append((X[i], pi, float(Z[i]), float(VW[i])))
        if len(Xm):
            pi_m = [(m, p) for m, p in zip(pim[i].tolist(), pi_p[i].tolist()) if m >= 0]
            out.append((Xm[i], pi_m, float(Z[i]), float(VW[i])))
    return out


class Learner:
    def __init__(self, args):
        self.a = args
        self.rng = random.Random(args.seed)
        self.nprng = np.random.default_rng(args.seed)
        self.net = Net(args.seed)
        self.best = Net(args.seed)
        self.buffer = deque(maxlen=args.buffer)
        self.seen = set()
        self.games = 0
        self.counts = {"red": 0, "black": 0, "draw": 0}
        self.iter = 0
        self.elo = 1000.0
        self.t0 = time.time()
        os.makedirs(args.run, exist_ok=True)
        os.makedirs(args.queue, exist_ok=True)
        self.log_path = os.path.join(args.run, "events.log")

        if args.warmstart:
            losses = material_warmstart(self.net, steps=args.warmstart, seed=args.seed)
            self.log(f"热启动 {args.warmstart} 步, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
        for k in self.net.p:
            self.best.p[k] = self.net.p[k].copy()
        self.anchor_X, self.anchor_Z = gen_material_samples(3000, args.seed + 1)
        self.publish()

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def publish(self):
        tmp = self.a.model + ".tmp.npz"
        self.net.save(tmp)
        os.replace(tmp, self.a.model)

    # ---------------------------------------------------------------- 训练
    def train(self):
        if len(self.buffer) < 300:
            return None
        a = self.a
        X = np.empty((a.batch, N_IN), dtype=np.float32)
        PI = np.zeros((a.batch, N_POL), dtype=np.float32)
        Z = np.empty((a.batch, 1), dtype=np.float32)
        PM = np.ones((a.batch, 1), dtype=np.float32)
        VM = np.ones((a.batch, 1), dtype=np.float32)
        n_anchor = int(a.batch * a.anchor_prob)
        losses = []
        n = len(self.buffer)
        for _ in range(a.steps):
            idxs = self.nprng.integers(0, n, size=a.batch - n_anchor)
            PI[:] = 0.0
            PM[:] = 1.0
            VM[:] = 1.0
            for j, k in enumerate(idxs):
                xs, pi, z, vw = self.buffer[k]
                VM[j, 0] = vw
                X[j] = xs
                for mi, p in pi:
                    PI[j, mi] = p
                Z[j, 0] = z
            if n_anchor:
                ai = self.nprng.integers(0, len(self.anchor_X), n_anchor)
                X[a.batch - n_anchor:] = self.anchor_X[ai]
                Z[a.batch - n_anchor:, 0] = self.anchor_Z[ai, 0]
                PM[a.batch - n_anchor:] = 0.0
            PI /= np.maximum(PI.sum(axis=1, keepdims=True), 1e-9)
            lp, lv = self.net.train_step(X, PI, Z, lr=a.lr,
                                         policy_weight=a.policy_weight, pmask=PM, vmask=VM)
            losses.append((lp, lv))
        return (float(np.mean([x[0] for x in losses])),
                float(np.mean([x[1] for x in losses])))

    # ---------------------------------------------------------------- 评估
    def eval_vs_random(self, n_games):
        wins = draws = 0
        for i in range(n_games):
            side = RED if i % 2 == 0 else BLACK
            r = play_vs_random(self.net, self.a.eval_sims, self.rng, side, use_q=self.a.use_q)
            wins += 1 if r == side else 0
            draws += 1 if r == 0 else 0
        return wins / n_games, draws / n_games

    def eval_vs_best(self, n_games):
        score = 0.0
        for i in range(n_games):
            side = RED if i % 2 == 0 else BLACK
            r = play_vs_model(self.net, self.best, self.a.eval_sims, self.rng, side, use_q=self.a.use_q)
            score += 0.5 if r == 0 else (1.0 if r == side else 0.0)
        wr = score / n_games
        self.elo += 48 * (wr - 0.5)
        return wr

    # ---------------------------------------------------------------- 主循环
    def run(self):
        def on_term(signum, frame):
            self.log(f"信号 {signum}，保存退出")
            self.publish()
            self.best.save(os.path.join(self.a.run, "best.npz"))
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)

        deadline = time.time() + self.a.hours * 3600
        last_train_games = 0
        while time.time() < deadline:
            # 消费新样本
            files = sorted(glob.glob(os.path.join(self.a.queue, "*.npz")))
            fresh = [f for f in files if os.path.basename(f) not in self.seen]
            if not fresh:
                time.sleep(1.0)
                self.write_status(None)
                continue
            for f in fresh:
                self.seen.add(os.path.basename(f))
                try:
                    d = np.load(f)
                    r = int(d["meta"][0])
                    self.counts["draw" if r == 0 else ("red" if r == 1 else "black")] += 1
                    for item in load_queue_file(f):
                        self.buffer.append(item)
                except Exception as e:
                    self.log(f"读样本失败 {f}: {e}")
            self.games += len(fresh)

            if self.games - last_train_games >= self.a.train_every:
                last_train_games = self.games
                self.iter += 1
                loss = self.train()
                row = {"iter": self.iter, "games": self.games,
                       "buffer": len(self.buffer), "elo": round(self.elo, 1),
                       "loss_p": round(loss[0], 4) if loss else None,
                       "loss_v": round(loss[1], 4) if loss else None,
                       "winrate_random": None, "winrate_best": None,
                       "results": dict(self.counts)}
                if self.iter % self.a.eval_every == 0:
                    wr, dr = self.eval_vs_random(self.a.eval_games)
                    row["winrate_random"], row["draw_random"] = round(wr, 3), round(dr, 3)
                    self.log(f"iter {self.iter}: games={self.games} vs-random {wr:.2f}")
                if self.iter % self.a.gate_every == 0 and self.iter > 1:
                    wr = self.eval_vs_best(self.a.gate_games)
                    row["winrate_best"] = round(wr, 3)
                    if wr >= 0.55:
                        for k in self.net.p:
                            self.best.p[k] = self.net.p[k].copy()
                        self.best.save(os.path.join(self.a.run, "best.npz"))
                        self.log(f"iter {self.iter}: 门控通过 {wr:.2f}")
                self.publish()
                with open(os.path.join(self.a.run, "stats.jsonl"), "a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.write_status(row)

        self.publish()
        self.best.save(os.path.join(self.a.run, "best.npz"))
        self.log(f"done: iters={self.iter} games={self.games}")

    def write_status(self, row):
        try:
            st = dict(row or {})
            st["elapsed_min"] = round((time.time() - self.t0) / 60, 1)
            st["buffer"] = len(self.buffer)
            st["pid"] = os.getpid()
            with open(os.path.join(self.a.run, "status.json"), "w") as f:
                json.dump(st, f, ensure_ascii=False)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--policy-weight", type=float, default=0.1)
    ap.add_argument("--anchor-prob", type=float, default=0.3)
    ap.add_argument("--buffer", type=int, default=30000)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-sims", type=int, default=120)
    ap.add_argument("--gate-every", type=int, default=20)
    ap.add_argument("--gate-games", type=int, default=20)
    ap.add_argument("--warmstart", type=int, default=300)
    ap.add_argument("--use-q", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    args.run = os.path.abspath(args.run)
    Learner(args).run()


if __name__ == "__main__":
    main()
