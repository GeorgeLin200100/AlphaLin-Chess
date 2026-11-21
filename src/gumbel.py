"""Gumbel 根节点搜索（Danheiser et al. 2022 的简化版）。

和 PUCT 的区别只有一点：根节点不把 160 次模拟摊在 42 个候选上，
而是先用 Gumbel 噪声抽出 m 个候选（人类式的"只看几种可能的走法"），
再把预算按"序列减半"分给它们：每轮给每个候选一小笔预算，
算完淘汰掉一半，剩下的拿更多预算。

产出两份东西：
  π' = softmax(log p + σ·q)   改进后的策略，直接当训练目标（比访问次数更稳）
  a  = argmax(g + log p + σ·q) 这一步实际走哪
"""
import math

import numpy as np

from mcts import Node, expand, select
from xiangqi import move_to_index


def _one_sim(b, node, net, use_q, c_puct=1.6):
    """从 node 出发跑一次模拟，统计只记在 node 及其子树，不碰根。"""
    path = []
    cur = node
    while cur.expanded and not cur.terminal:
        cur = select(cur, c_puct)
        b.make(*cur.move)
        path.append(cur)
    v = expand(cur, b, net, use_q) if not cur.expanded else cur.tv
    for nd in reversed(path):
        nd.N += 1
        nd.W += v
        v = -v
    for _ in path:
        b.unmake()
    node.N += 1
    node.W += v
    return v


def gumbel_search(board, net, sims=64, m=12, c_puct=1.6, use_q=False,
                  rng=None, sigma=1.0, force_boost=1.0):
    """返回 (π': {(frm,to): prob}, 根价值, 最佳着法, 最高概率)。"""
    rng = rng or np.random.default_rng()
    b = board.copy()
    root = Node()
    v_root = expand(root, b, net, use_q)
    if root.terminal:
        return {}, v_root, None, 0.0

    children = root.children
    n = len(children)
    # 强制着法优先：吃子和将军的先验放大（人类和引擎都会先算这些）
    if force_boost > 1.0:
        for ch in children:
            victim = b.sq[ch.move[1]]
            if victim is not None:
                ch.P *= force_boost

    priors = np.array([max(c.P, 1e-8) for c in children], dtype=np.float64)
    priors /= priors.sum()
    logp = np.log(priors)
    g = rng.gumbel(size=n)

    k = min(m, n)
    order = np.argsort(-(logp + g))[:k]
    cands = [children[i] for i in order]
    idx_of = {id(c): i for i, c in enumerate(children)}

    # 序列减半
    budget = sims
    while len(cands) > 1:
        phases = max(1, int(math.log2(len(cands))))
        per = max(1, budget // (len(cands) * phases))
        for c in cands:
            for _ in range(per):
                b.make(*c.move)
                _one_sim(b, c, net, use_q, c_puct)
                b.unmake()
        budget = max(0, budget - per * len(cands))

        def score(c):
            i = idx_of[id(c)]
            q = c.W / c.N if c.N else v_root
            return g[i] + logp[i] + sigma * q

        cands.sort(key=score, reverse=True)
        cands = cands[: max(1, len(cands) // 2)]

    # 改进后的策略：所有候选上 softmax(log p + σ q)
    qs = np.array([c.W / c.N if c.N else v_root for c in children], dtype=np.float64)
    logits = logp + sigma * qs
    logits -= logits.max()
    w = np.exp(logits)
    probs = w / w.sum()

    best = children[int(np.argmax(logits))]
    pi = {(c.move[0], c.move[1]): float(p) for c, p in zip(children, probs)}
    top1 = np.sort(probs)[-1] if len(probs) else 0.0
    root_value = float(root.W / root.N) if root.N else v_root
    return pi, root_value, (best.move[0], best.move[1]), float(top1)
