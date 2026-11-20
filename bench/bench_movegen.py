"""Measure python-chess legal move generation + make/unmake throughput.

Run: python bench_movegen.py [n_positions]
Single-core number, then multiprocess scaling across available cores.
"""
import sys, time, random
import chess


def walk(n_positions, seed=0):
    rng = random.Random(seed)
    board = chess.Board()
    done = 0
    t0 = time.perf_counter()
    while done < n_positions:
        moves = list(board.legal_moves)
        # sample a random child instead of a full tree walk: keeps games diverse
        if not moves or board.is_game_over():
            board.reset()
            continue
        board.push(rng.choice(moves))
        done += 1
    dt = time.perf_counter() - t0
    return n_positions / dt


def worker(n, seed, q):
    q.put(walk(n, seed))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    t0 = time.perf_counter()
    rate1 = walk(n, 1)
    print(f"single-core movegen+selection: {rate1:,.0f} pos/s  ({n} pos in {time.perf_counter()-t0:.1f}s)")

    from multiprocessing import Process, Queue
    nproc = 4
    q = Queue()
    ps = [Process(target=worker, args=(n, 100 + i, q)) for i in range(nproc)]
    t0 = time.perf_counter()
    [p.start() for p in ps]
    [p.join() for p in ps]
    dt = time.perf_counter() - t0
    rates = [q.get() for _ in ps]
    print(f"{nproc} procs aggregate: {sum(rates):,.0f} pos/s (wall {dt:.1f}s, rates={[round(r) for r in rates]})")
