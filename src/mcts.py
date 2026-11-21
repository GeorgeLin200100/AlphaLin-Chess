"""PUCT MCTS：懒拷贝棋盘（只在下降时 make/unmake）+ 只对合法着法取网络 logits。

单线程；每节点成本 ≈ 合法着法生成(~150us) + 网络(~10us)。
"""
import math
import numpy as np

from xiangqi import (Board, encode_planes, encode_sparse, move_to_index,
                     adjudicate, is_repetition, RED)
import quiescence

MAX_PLY = 200
NO_CAPTURE_DRAW = 100          # 60 步无吃子判和，避免早期大量无意义长局


class Node:
    __slots__ = ("parent", "move", "children", "P", "N", "W",
                 "expanded", "terminal", "tv", "depth")

    def __init__(self, parent=None, move=None, prior=1.0, depth=0):
        self.parent = parent
        self.move = move
        self.children = []
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.expanded = False
        self.terminal = False
        self.tv = 0.0
        self.depth = depth

    def q(self):
        return self.W / self.N if self.N else 0.0


def expand(node, board, net, use_q=False):
    """在 board（就是 node 所在局面）上展开。返回该节点走子方视角的价值。"""
    moves = board.legal_moves()
    if not moves:
        node.terminal, node.tv = True, -1.0        # 将死/困毙
        return -1.0
    if is_repetition(board):
        node.terminal, node.tv = True, 0.0         # 三次重复判和
        return 0.0
    if board.move_count >= MAX_PLY or board.move_count - board.last_capture >= NO_CAPTURE_DRAW:
        res = adjudicate(board)                    # 子力裁定
        node.terminal, node.tv = True, (0.0 if res == 0 else (1.0 if res == board.side else -1.0))
        return node.tv

    idx = np.fromiter((move_to_index(f, t) for f, t in moves),
                      dtype=np.int64, count=len(moves))
    probs, value = net.evaluate_sparse(encode_sparse(board), idx)
    if use_q:
        value = quiescence.adjust(board, value)
    node.children = [Node(node, mv, float(p), node.depth + 1)
                     for mv, p in zip(moves, probs)]
    node.expanded = True
    return value


def select(node, c_puct):
    best, best_score = None, -1e9
    sqrt_total = math.sqrt(node.N + 1)
    for ch in node.children:
        score = -ch.q() + c_puct * ch.P * sqrt_total / (1 + ch.N)
        if score > best_score:
            best, best_score = ch, score
    return best


def search(board, net, sims=100, c_puct=1.6, dirichlet=0.0, rng=None, use_q=False,
           root_topk=0, force_boost=1.0, safety=False, safety_margin=1.0):
    """返回 (visits: {(frm,to): n}, root_value)。不会修改传入的 board。"""
    b = board.copy()
    root = Node()
    root_val = expand(root, b, net, use_q)
    if root.terminal:
        return {}, root_val

    if safety:
        # 硬规则过滤器：走完之后对方能白赚 margin 以上子力的着法直接剔除
        keep = [c for c in root.children
                if quiescence.forced_loss(b, c.move, safety_margin) < safety_margin]
        if len(keep) >= 3:
            dropped = len(root.children) - len(keep)
            if dropped:
                root.children = keep
                tot = sum(c.P for c in root.children) or 1.0
                for c in root.children:
                    c.P /= tot

    if force_boost > 1.0:
        for ch in root.children:
            if b.sq[ch.move[1]] is not None:      # 吃子着法先验放大
                ch.P *= force_boost

    if root_topk and len(root.children) > root_topk:
        # 不平均用力：只保留先验最高的 k 个候选，整体重归一化
        root.children.sort(key=lambda c: -c.P)
        root.children = root.children[:root_topk]
        tot = sum(c.P for c in root.children) or 1.0
        for c in root.children:
            c.P /= tot

    if dirichlet > 0:
        noise = (rng or np.random.default_rng()).dirichlet([dirichlet] * len(root.children))
        for ch, n in zip(root.children, noise):
            ch.P = 0.75 * ch.P + 0.25 * n

    for _ in range(sims):
        node, path = root, []
        while node.expanded and not node.terminal:
            node = select(node, c_puct)
            b.make(*node.move)
            path.append(node)
        v = expand(node, b, net, use_q) if not node.expanded else node.tv
        for nd in reversed(path):
            nd.N += 1
            nd.W += v
            v = -v
        root.N += 1
        root.W += v
        for _ in path:
            b.unmake()

    return {(ch.move[0], ch.move[1]): ch.N for ch in root.children}, root.q()
