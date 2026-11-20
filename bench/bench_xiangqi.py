"""Xiangqi (中国象棋) movegen sanity + throughput using cchess.

Known perft from the standard start position (ICCS):
  d1=44, d2=1920, d3=79666, d4=3290240
"""
import sys, time, random
from cchess import ChessBoard


def perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for mv in board.create_moves():
        if not board.is_valid_move(mv, True):
            continue
        board.move(mv)
        total += perft(board, depth - 1)
        board.pop_fench()
    return total


def throughput(n_positions, seed=0):
    rng = random.Random(seed)
    board = ChessBoard()
    done, hist = 0, 0
    t0 = time.perf_counter()
    while done < n_positions:
        moves = [m for m in board.create_moves() if board.is_valid_move(m, True)]
        if not moves or board.is_checkmate():
            board = ChessBoard()
            continue
        board.move(rng.choice(moves))
        done += 1
    dt = time.perf_counter() - t0
    return n_positions / dt


if __name__ == "__main__":
    for d in (1, 2, 3):
        b = ChessBoard()
        t0 = time.perf_counter()
        n = perft(b, d)
        print(f"perft({d}) = {n:,}  ({time.perf_counter()-t0:.2f}s)")
    rate = throughput(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
    print(f"xiangqi movegen+random walk: {rate:,.0f} pos/s (single core)")
