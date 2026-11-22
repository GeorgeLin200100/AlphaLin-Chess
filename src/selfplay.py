"""自博弈一局的公共实现，供 actor 进程调用。

产出：每局一个 npz（X / 策略前 16 / 价值目标），写进队列目录，
由 learner 进程消费。设计目标是"无锁"：actor 只写新文件，learner 只读不删。
"""
import os
import time

import numpy as np

from xiangqi import (Board, START_FEN, encode_planes, move_to_index,
                     mirror_move_index, adjudicate, RED, BLACK, is_repetition)
from mcts import search

TOP_K = 16
MAX_PLY = 300
MIRROR_TABLE = np.array([mirror_move_index(i) for i in range(90 * 90)], dtype=np.int32)


def play_self_game(net, sims, rng, nprng, noise=True, use_q=False,
                   temp_plies=60, max_ply=MAX_PLY):
    """返回 (X, PI_idx, PI_p, Z)，以及一局的元信息 dict。"""
    board = Board(START_FEN)
    xs, pi_idx, pi_p, z_root = [], [], [], []
    moves_log = []
    adjudicated = False
    while board.move_count < max_ply:
        res = board.result()
        if res is not None:
            break
        if is_repetition(board):
            res, adjudicated = 0, True
            break
        visits, root_value = search(board, net, sims=sims,
                                    dirichlet=0.3 if noise else 0.0,
                                    rng=nprng, use_q=use_q, safety=True)
        if not visits:
            # 搜索判成终局（无吃子 100 步 / 规则和棋），按子力裁定，不能当输局记
            res, adjudicated = adjudicate(board), True
            break
        total = sum(visits.values())
        temp = 1.0 if board.move_count < temp_plies else 0.05
        moves = list(visits.keys())
        counts = np.array([visits[m] for m in moves], dtype=np.float64)
        if temp <= 1e-3:
            mv = moves[int(counts.argmax())]
        else:
            w = counts ** (1.0 / temp)
            mv = moves[int(nprng.choice(len(moves), p=w / w.sum()))]

        top = sorted(visits.items(), key=lambda kv: -kv[1])[:TOP_K]
        xs.append(encode_planes(board).astype(np.uint8))
        ids = [move_to_index(f, t) for (f, t), _ in top]
        prs = [n / total for _, n in top]
        while len(ids) < TOP_K:            # 合法着法不足 16 个时补齐，保持矩形
            ids.append(-1)
            prs.append(0.0)
        pi_idx.append(ids)
        pi_p.append(prs)
        z_root.append(root_value)          # 搜索给出的价值，用来降方差
        moves_log.append({"frm": mv[0], "to": mv[1],
                          "p": {f"{f},{t}": round(n / total, 3) for (f, t), n in top[:5]}})
        board.make(*mv)
    else:
        res, adjudicated = adjudicate(board), True

    if not xs:
        return None

    z = 0.0 if res == 0 else (1.0 if res == RED else -1.0)
    X = np.stack(xs)                                    # (n, 1260) uint8
    pi_idx = np.asarray(pi_idx, dtype=np.int32)
    pi_p = np.asarray(pi_p, dtype=np.float32)

    # 每步的走子方，用来把胜负翻到"该步视角"
    sides, b2 = [], Board(START_FEN)
    for m in moves_log:
        sides.append(b2.side)
        b2.make(m["frm"], m["to"])
    z_signed = np.array([0.0 if res == 0 else (z if s == RED else -z)
                         for s in sides], dtype=np.float32)
    # 价值目标就是最终胜负（不做自指混合：搜索价值由网络自己给出，
    # 掺进去等于让网络拟合自己的判断，实测会让棋力缓慢下滑）
    z_final = z_signed

    # 左右镜像增强：同一局的另一份数据
    n_pl = X.shape[1] // 90
    Xm = X.reshape(-1, n_pl, 10, 9)[:, :, :, ::-1].reshape(len(X), -1).copy()
    pim = MIRROR_TABLE[pi_idx]

    VW = np.zeros(len(xs), dtype=np.float32) if adjudicated else np.ones(len(xs), dtype=np.float32)
    meta = {"result": res, "plies": len(xs), "moves": moves_log, "adjudicated": adjudicated}
    return (X, pi_idx, pi_p, z_final, Xm, pim, VW), meta


def save_game(path_dir, gid, payload, meta):
    X, pi_idx, pi_p, z, Xm, pim, vw = payload
    tmp = os.path.join(path_dir, f".tmp_{gid}.npz")
    final = os.path.join(path_dir, f"{gid}.npz")
    np.savez_compressed(tmp, X=X, pi_idx=pi_idx, pi_p=pi_p, Z=z, VW=vw,
                        Xm=Xm, pim=pim, meta=np.array([meta["result"], meta["plies"]]))
    os.replace(tmp, final)
    return final
