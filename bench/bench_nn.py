"""Measure numpy NN throughput on this box: forward + SGD backward.

Network shape mimics a deliberately tiny AlphaZero-style net:
  input  (12*8*8=768) -> 256 -> 256 -> {policy 4096, value 1}
Batch matmuls, tanh/relu, MSE-ish loss, manual backward. Pure numpy (BLAS).
"""
import time
import numpy as np

rng = np.random.default_rng(0)
D_IN, H1, H2, N_POL = 768, 256, 256, 4096


def init():
    return {
        "W1": rng.normal(0, 1 / np.sqrt(D_IN), (D_IN, H1)).astype(np.float32),
        "b1": np.zeros(H1, np.float32),
        "W2": rng.normal(0, 1 / np.sqrt(H1), (H1, H2)).astype(np.float32),
        "b2": np.zeros(H2, np.float32),
        "Wp": rng.normal(0, 1 / np.sqrt(H2), (H2, N_POL)).astype(np.float32),
        "bp": np.zeros(N_POL, np.float32),
        "Wv": rng.normal(0, 1 / np.sqrt(H2), (H2, 1)).astype(np.float32),
        "bv": np.zeros(1, np.float32),
    }


def fwd(p, x):
    z1 = x @ p["W1"] + p["b1"]
    h1 = np.maximum(z1, 0)
    z2 = h1 @ p["W2"] + p["b2"]
    h2 = np.maximum(z2, 0)
    logits = h2 @ p["Wp"] + p["bp"]
    v = np.tanh(h2 @ p["Wv"] + p["bv"])
    return z1, h1, z2, h2, logits, v


def step(p, x, pi, z_target, lr=1e-3):
    z1, h1, z2, h2, logits, v = fwd(p, x)
    b = x.shape[0]
    # policy: softmax cross-entropy
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    prob = e / e.sum(axis=1, keepdims=True)
    loss_p = -(pi * np.log(prob + 1e-9)).sum(axis=1).mean()
    dlogits = (prob - pi) / b
    # value: MSE
    dv = 2 * (v - z_target) / b
    dWp = h2.T @ dlogits
    dWv = h2.T @ (dv * (1 - v * v))
    dh2 = dlogits @ p["Wp"].T + (dv * (1 - v * v)) @ p["Wv"].T
    dz2 = dh2 * (h2 > 0)
    dW2 = h1.T @ dz2
    dh1 = dz2 @ p["W2"].T
    dz1 = dh1 * (h1 > 0)
    dW1 = x.T @ dz1
    for k, g in (("W1", dW1), ("W2", dW2), ("Wp", dWp), ("Wv", dWv)):
        p[k] -= lr * g
    p["b1"] -= lr * dz1.sum(0)
    p["b2"] -= lr * dz2.sum(0)
    p["bp"] -= lr * dlogits.sum(0)
    p["bv"] -= lr * (dv * (1 - v * v)).sum(0)
    return float(loss_p)


if __name__ == "__main__":
    p = init()
    x = rng.random((256, D_IN), dtype=np.float32)
    pi = rng.random((256, N_POL), dtype=np.float32)
    z = rng.random((256, 1), dtype=np.float32)

    # batched SGD (training-step throughput)
    for _ in range(2):
        step(p, x, pi, z)
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        step(p, x, pi, z)
    dt = time.perf_counter() - t0
    print(f"train: {iters*256/dt:,.0f} samples/s ({dt/iters*1000:.0f} ms per batch-256 step)")

    # inference (MCTS leaf eval) throughput, batch 1 and batch 64
    for bs in (1, 8, 64):
        xb = rng.random((bs, D_IN), dtype=np.float32)
        fwd(p, xb)
        t0 = time.perf_counter()
        n = max(1, 2000 // bs)
        for _ in range(n):
            fwd(p, xb)
        dt = time.perf_counter() - t0
        print(f"infer bs={bs:<3d}: {n*bs/dt:,.0f} evals/s")

    # raw matmul throughput reference
    a = rng.random((512, 512), dtype=np.float32)
    bb = rng.random((512, 512), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(200):
        a @ bb
    dt = time.perf_counter() - t0
    print(f"matmul 512x512: {200*2*512**3/dt/1e9:.1f} GFLOPS")
