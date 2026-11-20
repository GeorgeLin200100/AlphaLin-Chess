"""叶子静息评估（quiescence）：把"下一步能吃你一个子"直接算进叶子的价值。

为什么需要：网络只看静态局面，看不见"我这匹马下一手就没了"。
MCTS 在 160 次模拟时还探不到那一层，于是它会白送子。
这里在叶子做一次只含吃子的浅层交换（近似 SEE），
不改网络、不改训练，纯搜索侧的补丁。

实测（best.npz，160 sims，对一步贪心 6 局）：
  不加：0 胜 6 负
  加 1 层：2 胜 4 负（节点成本 +19%）
"""
from xiangqi import PIECE_VALUE

DAMP = 0.5          # 修正量的折扣
SCALE = 10.0        # 子力差 10 分（约一个车）对应价值 1.0
MAX_DEPTH = 1       # 交换深度：实测 2 层反而更差（512µs/node, 1胜5负）


def best_capture_gain(board, depth=MAX_DEPTH):
    """当前走子方在 depth 层强制吃子交换里能净赚多少子力（近似 SEE）。"""
    me = board.side
    sq = board.sq
    best = 0.0
    for frm, to in board.gen_pseudo():
        victim = sq[to]
        if victim is None:
            continue                       # 只看吃子
        gain = PIECE_VALUE[victim[0]]
        attacker = sq[frm]
        if attacker is not None and board.attacked(to, -me):
            gain -= PIECE_VALUE[attacker[0]]   # 会被反吃
            if depth > 1:
                board.make(frm, to)
                counter = best_capture_gain(board, depth - 1)
                board.unmake()
                gain -= counter                # 他反吃之后我能再赚回来的部分
        if gain > best:
            best = gain
    return best


def adjust(board, value):
    """把叶子的网络价值往"有子可吃"的方向修正（该节点走子方视角）。"""
    gain = best_capture_gain(board)
    if gain <= 0:
        return value
    return max(-1.0, min(1.0, value + DAMP * min(gain / SCALE, 1.0)))
