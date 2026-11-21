"""中国象棋规则引擎（纯 Python，单线程）。

设计目标：正确性优先（用 perft 对齐权威值），速度够跑单核自博弈。
棋盘 9 列 x 10 行，索引 = y*9 + x。红方在下（y=0..4），黑方在上（y=6..9），红先。

权威 perft（起始局面）:
  d1 = 44, d2 = 1,920, d3 = 79,666, d4 = 3,290,240
"""

RED, BLACK = 1, -1

# 棋子类型
ROOK, KNIGHT, BISHOP, ADVISOR, KING, CANNON, PAWN = range(7)
TYPE_CHARS = "RNBAKCP"          # 大写=红, 小写=黑
FEN_CHARS = {"R": ROOK, "N": KNIGHT, "B": BISHOP, "A": ADVISOR,
             "K": KING, "C": CANNON, "P": PAWN}

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

# ---------------------------------------------------------------- 预计算表
ORTHO = ((0, 1), (0, -1), (1, 0), (-1, 0))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))

# 马的 8 个走法: (dx, dy, 马腿 dx, 马腿 dy)
KNIGHT_MOVES = []
for dx, dy in ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
    if abs(dy) == 2:
        KNIGHT_MOVES.append((dx, dy, 0, dy // 2))
    else:
        KNIGHT_MOVES.append((dx, dy, dx // 2, 0))

# 九宫
PALACE_X = (3, 4, 5)


def in_palace(x, y, side):
    if not (3 <= x <= 5):
        return False
    return (0 <= y <= 2) if side == RED else (7 <= y <= 9)


def on_board(x, y):
    return 0 <= x <= 8 and 0 <= y <= 9


class Board:
    __slots__ = ("sq", "codes", "side", "kings", "history", "move_count", "last_capture",
                 "h", "h_list")

    def __init__(self, fen=START_FEN):
        self.sq = [None] * 90          # None 或 (type, side)
        self.codes = __import__("numpy").zeros(90, dtype=__import__("numpy").int8)
        self.side = RED
        self.kings = {RED: None, BLACK: None}
        self.history = []              # (from, to, captured)
        self.move_count = 0
        self.last_capture = 0
        self.h = 0
        self.h_list = []
        if fen:
            self.set_fen(fen)

    # ------------------------------------------------------------ FEN
    def set_fen(self, fen):
        parts = fen.split()
        rows = parts[0].split("/")
        self.sq = [None] * 90
        self.codes = __import__("numpy").zeros(90, dtype=__import__("numpy").int8)
        for i, row in enumerate(rows):
            y = 9 - i
            x = 0
            for ch in row:
                if ch.isdigit():
                    x += int(ch)
                else:
                    t = FEN_CHARS[ch.upper()]
                    side = RED if ch.isupper() else BLACK
                    self.sq[y * 9 + x] = (t, side)
                    self.codes[y * 9 + x] = t + 1 if side == RED else t + 8
                    if t == KING:
                        self.kings[side] = y * 9 + x
                    x += 1
        self.side = BLACK if len(parts) > 1 and parts[1] == "b" else RED
        self.history.clear()
        self.move_count = 0
        self.last_capture = 0
        self.h_list = [0]
        z = _zobrist()
        h = 0
        for i, c in enumerate(self.codes):
            if c:
                h ^= int(z[int(c), i])
        self.h = h
        self.h_list = [h]

    def fen(self):
        rows = []
        for y in range(9, -1, -1):
            row, empty = "", 0
            for x in range(9):
                p = self.sq[y * 9 + x]
                if p is None:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                ch = TYPE_CHARS[p[0]]
                row += ch if p[1] == RED else ch.lower()
            if empty:
                row += str(empty)
            rows.append(row)
        return "/".join(rows) + (" w" if self.side == RED else " b")

    def copy(self):
        b = Board(None)
        b.sq = self.sq[:]
        b.codes = self.codes.copy()
        b.side = self.side
        b.kings = dict(self.kings)
        b.move_count = self.move_count
        b.last_capture = self.last_capture
        b.h = self.h
        b.h_list = self.h_list[:]
        return b

    # ------------------------------------------------------------ 走子
    def make(self, frm, to):
        sq = self.sq
        piece = sq[frm]
        captured = sq[to]
        z = _zobrist()
        pc = self.codes[frm]
        h = self.h ^ int(z[int(pc), frm]) ^ int(z[int(pc), to])
        if captured is not None:
            h ^= int(z[int(self.codes[to]), to])
        sq[to] = piece
        sq[frm] = None
        self.codes[frm] = 0
        if piece[0] == KING:
            self.kings[piece[1]] = to
        self.codes[to] = piece[0] + 1 if piece[1] == RED else piece[0] + 8
        self.h = h
        self.h_list.append(h)
        self.history.append((frm, to, captured, self.last_capture))
        self.side = -self.side
        self.move_count += 1
        if captured is not None:
            self.last_capture = self.move_count
        return captured

    def unmake(self):
        frm, to, captured, prev_capture = self.history.pop()
        piece = self.sq[to]
        self.sq[frm] = piece
        self.sq[to] = captured
        self.codes[frm] = piece[0] + 1 if piece[1] == RED else piece[0] + 8
        self.codes[to] = 0 if captured is None else (captured[0] + 1 if captured[1] == RED else captured[0] + 8)
        if piece[0] == KING:
            self.kings[piece[1]] = frm
        self.side = -self.side
        self.move_count -= 1
        self.last_capture = prev_capture
        if self.h_list:
            self.h_list.pop()
        self.h = self.h_list[-1] if self.h_list else 0

    # ------------------------------------------------------------ 攻击判定
    def attacked(self, sq_idx, by_side):
        """sq_idx 是否被 by_side 方攻击（含飞将）。"""
        sq = self.sq
        kx, ky = sq_idx % 9, sq_idx // 9

        # 车/炮：沿 4 条直线扫描
        for dx, dy in ORTHO:
            x, y = kx + dx, ky + dy
            first = None
            while 0 <= x <= 8 and 0 <= y <= 9:
                p = sq[y * 9 + x]
                if p is not None:
                    if first is None:
                        if p[1] == by_side and p[0] in (ROOK, KING):
                            if p[0] == ROOK:
                                return True
                            # 将帅照面（同列相邻不算攻击，同列无子才算）
                            if p[0] == KING and dx == 0:
                                return True
                        first = p
                    elif p[1] == by_side and p[0] == CANNON:
                        return True
                    else:
                        break
                x += dx
                y += dy

        # 马（马腿要相对"马所在格"算）
        for dx, dy, lx, ly in KNIGHT_MOVES:
            kx2, ky2 = kx - dx, ky - dy      # 马可能所在的格
            if not on_board(kx2, ky2):
                continue
            p = sq[ky2 * 9 + kx2]
            if p is not None and p[0] == KNIGHT and p[1] == by_side:
                if sq[(ky2 + ly) * 9 + (kx2 + lx)] is None:   # 马腿没被塞
                    return True

        # 兵/卒
        if by_side == RED:
            # 红兵向上走，能吃它上方的目标，过河后还能横吃
            if ky + 1 <= 9:
                p = sq[(ky + 1) * 9 + kx]
                if p is not None and p[0] == PAWN and p[1] == RED:
                    return True
            if ky >= 5:   # 红兵已过河（在 y>=5）
                for dx in (1, -1):
                    x = kx + dx
                    if 0 <= x <= 8:
                        p = sq[ky * 9 + x]
                        if p is not None and p[0] == PAWN and p[1] == RED:
                            return True
        else:
            if ky - 1 >= 0:
                p = sq[(ky - 1) * 9 + kx]
                if p is not None and p[0] == PAWN and p[1] == BLACK:
                    return True
            if ky <= 4:
                for dx in (1, -1):
                    x = kx + dx
                    if 0 <= x <= 8:
                        p = sq[ky * 9 + x]
                        if p is not None and p[0] == PAWN and p[1] == BLACK:
                            return True
        return False

    def in_check(self, side=None):
        side = self.side if side is None else side
        k = self.kings[side]
        if k is None:
            return True
        return self.attacked(k, -side)

    # ------------------------------------------------------------ 着法生成
    def gen_pseudo(self):
        """生成伪合法着法 (from, to, piece)。"""
        sq = self.sq
        me = self.side
        out = []
        ap = out.append
        for idx in range(90):
            p = sq[idx]
            if p is None or p[1] != me:
                continue
            t = p[0]
            x, y = idx % 9, idx // 9
            if t == ROOK or t == CANNON:
                for dx, dy in ORTHO:
                    nx, ny = x + dx, y + dy
                    screen = 0
                    while 0 <= nx <= 8 and 0 <= ny <= 9:
                        nidx = ny * 9 + nx
                        q = sq[nidx]
                        if t == ROOK:
                            if q is None:
                                ap((idx, nidx))
                            else:
                                if q[1] != me:
                                    ap((idx, nidx))
                                break
                        else:   # CANNON
                            if screen == 0:
                                if q is None:
                                    ap((idx, nidx))
                                else:
                                    screen = 1
                            else:
                                if q is not None:
                                    if q[1] != me:
                                        ap((idx, nidx))
                                    break
                        nx += dx
                        ny += dy
            elif t == KNIGHT:
                for dx, dy, lx, ly in KNIGHT_MOVES:
                    nx, ny = x + dx, y + dy
                    if not on_board(nx, ny):
                        continue
                    if sq[(y + ly) * 9 + (x + lx)] is not None:
                        continue
                    q = sq[ny * 9 + nx]
                    if q is None or q[1] != me:
                        ap((idx, ny * 9 + nx))
            elif t == BISHOP:
                for dx, dy in DIAG:
                    nx, ny = x + 2 * dx, y + 2 * dy
                    if not on_board(nx, ny):
                        continue
                    if (ny > 4) if me == RED else (ny < 5):     # 不过河
                        continue
                    if sq[(y + dy) * 9 + (x + dx)] is not None:  # 塞象眼
                        continue
                    q = sq[ny * 9 + nx]
                    if q is None or q[1] != me:
                        ap((idx, ny * 9 + nx))
            elif t == ADVISOR:
                for dx, dy in DIAG:
                    nx, ny = x + dx, y + dy
                    if not in_palace(nx, ny, me):
                        continue
                    q = sq[ny * 9 + nx]
                    if q is None or q[1] != me:
                        ap((idx, ny * 9 + nx))
            elif t == KING:
                for dx, dy in ORTHO:
                    nx, ny = x + dx, y + dy
                    if not in_palace(nx, ny, me):
                        continue
                    q = sq[ny * 9 + nx]
                    if q is None or q[1] != me:
                        ap((idx, ny * 9 + nx))
            else:   # PAWN
                if me == RED:
                    fy = y + 1
                    if fy <= 9:
                        q = sq[fy * 9 + x]
                        if q is None or q[1] != me:
                            ap((idx, fy * 9 + x))
                    if y >= 5:      # 过河后可横走
                        for dx in (1, -1):
                            nx = x + dx
                            if 0 <= nx <= 8:
                                q = sq[y * 9 + nx]
                                if q is None or q[1] != me:
                                    ap((idx, y * 9 + nx))
                else:
                    fy = y - 1
                    if fy >= 0:
                        q = sq[fy * 9 + x]
                        if q is None or q[1] != me:
                            ap((idx, fy * 9 + x))
                    if y <= 4:
                        for dx in (1, -1):
                            nx = x + dx
                            if 0 <= nx <= 8:
                                q = sq[y * 9 + nx]
                                if q is None or q[1] != me:
                                    ap((idx, y * 9 + nx))
        return out

    def _king_threat_maps(self, me):
        """沿王的 4 条直线分析，返回 (被牵制的己方子, 不能落子的格)。

        炮需要正好 1 个炮架，所以：
          - 王与敌炮之间有 0 个子 -> 中间空格不能落子（落了就成炮架）
          - 王与敌炮之间正好 2 个子 -> 其中的己方子被牵制（走开一个就剩 1 个炮架）
        车/飞将：王与敌车（敌将，同列）之间正好 1 个己方子 -> 该子被牵制。
        炮架可以是任何颜色的子，所以扫描时不区分敌我，只统计个数。
        """
        k = self.kings[me]
        kx, ky = k % 9, k // 9
        pins, forbidden = set(), set()
        for dx, dy in ORTHO:
            pieces, empties = [], []
            x, y = kx + dx, ky + dy
            while 0 <= x <= 8 and 0 <= y <= 9 and len(pieces) < 3:
                idx = y * 9 + x
                p = self.sq[idx]
                if p is None:
                    empties.append(idx)
                else:
                    if p[1] != me:
                        t = p[0]
                        if t == ROOK and len(pieces) == 1 and pieces[0][1] == me:
                            pins.add(pieces[0][0])
                        elif t == CANNON and len(pieces) == 2:
                            for i, side in pieces:
                                if side == me:
                                    pins.add(i)
                        elif t == CANNON and len(pieces) == 0:
                            forbidden.update(empties)
                        elif t == KING and dx == 0 and len(pieces) == 1 and pieces[0][1] == me:
                            pins.add(pieces[0][0])
                    pieces.append((idx, p[1]))
                    empties = []
                x += dx
                y += dy
        return pins, forbidden

    def legal_moves(self):
        """合法着法 (from, to)。快路径：无将时只验王、被牵制子与炮架格。"""
        me = self.side
        pseudo = self.gen_pseudo()
        k = self.kings[me]
        if self.attacked(k, -me):
            check = None                      # 被将军：全部验证
        else:
            pins, forbidden = self._king_threat_maps(me)
            pins.add(k)
            check = pins
        out = []
        for frm, to in pseudo:
            if check is not None:
                # 王自己走进"炮架格"是合法的（王是目标不是炮架），所以只限制非王着法
                if to in forbidden and frm != k:
                    continue
                if frm not in check:
                    out.append((frm, to))
                    continue
            self.make(frm, to)
            if not self.attacked(self.kings[me], -me):
                out.append((frm, to))
            self.unmake()
        return out

    def is_checkmate(self):
        return not self.legal_moves()

    def result(self):
        """None=未结束, RED/BLACK=胜方。无子可走即负（象棋规则：困毙为负）。"""
        if self.legal_moves():
            return None
        return -self.side


# ---------------------------------------------------------------- 编码
_ZOB = None


def _zobrist():
    global _ZOB
    if _ZOB is None:
        rng = __import__("numpy").random.default_rng(20241119)
        _ZOB = rng.integers(1, 2**62, size=(15, 90), dtype=__import__("numpy").int64)
    return _ZOB


PIECE_VALUE = {ROOK: 9.0, KNIGHT: 4.0, BISHOP: 2.0, ADVISOR: 2.0,
               CANNON: 4.5, PAWN: 1.0, KING: 0.0}


def material(board, side):
    return sum(PIECE_VALUE[p[0]] for p in board.sq if p is not None and p[1] == side)


def is_repetition(board, times=3):
    """同一局面出现 >= times 次（含当前）视为和棋。"""
    if len(board.h_list) < times:
        return False
    h = board.h
    return board.h_list.count(h) >= times


def adjudicate(board, margin=1.0):
    """到达步数上限/长期无吃子时的裁定：子力多者胜，势均判和。

    这是给 RL 的"稠密化"终局信号：早期网络几乎不可能将死对方，
    纯靠和棋会没有梯度；按子力裁定相当于告诉它"吃子=好"。
    """
    d = material(board, RED) - material(board, BLACK)
    if d >= margin:
        return RED
    if d <= -margin:
        return BLACK
    return 0


def move_to_index(frm, to):
    return frm * 90 + to


def index_to_move(i):
    return i // 90, i % 90


N_PLANES = 18          # 14 棋子 + 4 威胁/意图平面


def _attack_targets(board, side):
    """某一方所有伪合法着法的落点（含被牵制的子），用来标"被攻击的格"。"""
    old = board.side
    board.side = side
    try:
        return {to for _, to in board.gen_pseudo()}
    finally:
        board.side = old


def feature_indices(board):
    """当前局面的稀疏特征下标（视角永远是"该走子的一方"）。

    平面 0-6: 我的 车马相仕帅炮兵；7-13: 对方的对应子
    平面 14 : 我的子正被对方攻击
    平面 15 : 对方的子正被我攻击
    平面 16 : 对方上一步的落点
    平面 17 : 对方上一步的起点
    """
    me = board.side
    out = []
    sq = board.sq
    for idx, piece in enumerate(sq):
        if piece is None:
            continue
        t, s = piece
        out.append((t if s == me else 7 + t) * 90 + idx)

    opp_targets = _attack_targets(board, -me)
    my_targets = _attack_targets(board, me)
    for idx, piece in enumerate(sq):
        if piece is None:
            continue
        side = piece[1]
        if side == me and idx in opp_targets:
            out.append(14 * 90 + idx)
        elif side != me and idx in my_targets:
            out.append(15 * 90 + idx)

    if board.history:
        frm, to, *_ = board.history[-1]
        out.append(16 * 90 + to)
        out.append(17 * 90 + frm)
    return out


def encode_sparse(board):
    """稀疏特征下标，供单样本快速推理。"""
    import numpy as np
    return np.asarray(feature_indices(board), dtype=np.int32)


def encode_planes(board):
    """稠密特征向量（训练用），长度 N_PLANES*90。"""
    import numpy as np
    v = np.zeros(N_PLANES * 90, dtype=np.float32)
    idx = feature_indices(board)
    if idx:
        v[np.asarray(idx, dtype=np.int32)] = 1.0
    return v


def mirror_index(idx):
    """左右对称：x -> 8-x。用于数据增强。"""
    x, y = idx % 9, idx // 9
    return y * 9 + (8 - x)


def mirror_move_index(mi):
    frm, to = mi // 90, mi % 90
    return mirror_index(frm) * 90 + mirror_index(to)


# ---------------------------------------------------------------- perft
def perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for frm, to in board.gen_pseudo():
        captured = board.make(frm, to)
        me = -board.side
        if captured is not None and captured[0] == KING:
            total += 1
        elif not board.attacked(board.kings[me], board.side):
            total += perft(board, depth - 1)
        board.unmake()
    return total


if __name__ == "__main__":
    import sys, time
    depths = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    expect = {1: 44, 2: 1920, 3: 79666, 4: 3290240, 5: 133312995}
    for d in depths:
        b = Board()
        t0 = time.perf_counter()
        n = perft(b, d)
        dt = time.perf_counter() - t0
        ok = "OK" if expect.get(d) == n else f"MISMATCH(expect {expect.get(d)})"
        print(f"perft({d}) = {n:,}  [{ok}]  {dt:.2f}s  {n/dt:,.0f} nodes/s")
