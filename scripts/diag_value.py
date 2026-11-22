"""价值头体检：它到底比一个三行的子力函数强多少。

指标：
  Brier         (v - z)^2 的均值，z 是最终胜负（-1/0/1）。越低越好。
  常数基线      v ≡ 0 的 Brier = E[z^2]，约等于分胜负的比例
  子力基线      v = clip(a·子力差, -1, 1)，a 用最小二乘/网格搜出来
  校准表        按预测值分桶，看每桶的实际胜率
  与子力相关    corr(v, 子力差)，健康的价值头应该有明显正相关

用法: .venv/bin/python scripts/diag_value.py --queue runs/wide/queue --n 4000
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

W_PIECE = np.array([9.0, 4.0, 2.0, 2.0, 0.0, 4.5, 1.0], dtype=np.float32)  # R N B A K C P


def load(queue, want):
    xs, zs = [], []
    for f in sorted(glob.glob(os.path.join(queue, "*.npz")))[-150:]:
        d = np.load(f)
        for x, z in zip(d["X"], d["Z"]):
            xs.append(x), zs.append(z)
            if len(xs) >= want:
                return np.asarray(xs, dtype=np.float32), np.asarray(zs, dtype=np.float32)
    return np.asarray(xs, dtype=np.float32), np.asarray(zs, dtype=np.float32)


def material(X):
    c = X.reshape(len(X), 18, 90).sum(axis=2)
    return c[:, :7] @ W_PIECE - c[:, 7:14] @ W_PIECE


def brier(v, z):
    return float(((v - z) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="runs/wide/queue")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--ckpts", nargs="+",
                    default=["runs/main/model.npz", "runs/q1/model.npz", "runs/wide/model.npz"])
    args = ap.parse_args()

    X, Z = load(args.queue, args.n)
    M = material(X)
    print(f"样本 {len(X)} 条（{args.queue} 最近的对局）")
    print(f"分胜负比例 {np.mean(np.abs(Z) > 0).__float__():.3f}，"
          f"常数 0 预测的 Brier = {brier(np.zeros_like(Z), Z):.4f}")
    print(f"子力差分布：均值 {M.mean():+.2f}，标准差 {M.std():.2f}，"
          f"|差| > 2 的比例 {np.mean(np.abs(M) > 2):.3f}\n")

    # 子力基线：a 在网格上取最优（一个参数，过拟合可以忽略）
    grid = np.arange(0.02, 0.31, 0.005)
    best_a, best_b = None, 1e9
    for a in grid:
        b = brier(np.clip(a * M, -1, 1), Z)
        if b < best_b:
            best_a, best_b = a, b
    print(f"子力基线：v = clip({best_a:.3f}·子力差, -1, 1)，Brier = {best_b:.4f}  "
          f"（相关 {np.corrcoef(np.clip(best_a * M, -1, 1), Z)[0, 1]:+.3f}）\n")

    print("| checkpoint | Brier | 与 z 相关 | 与子力相关 | 平均 |v| | 比子力基线 |")
    print("|---|---|---|---|---|---|")
    vals = {}
    for path in args.ckpts:
        if not os.path.exists(path):
            continue
        with np.load(path) as d:
            w = d["W1p"].shape[1]
        os.environ["POL_H1"] = os.environ["POL_H2"] = str(w)
        for k in [m for m in list(sys.modules) if m in ("nn",)]:
            del sys.modules[k]
        from nn import Net  # noqa: E402
        net = Net.load(path)
        _, vv = net.forward(X)
        v = vv[:, 0]
        vals[path] = v
        print(f"| {path} | {brier(v, Z):.4f} | {np.corrcoef(v, Z)[0, 1]:+.3f} | "
              f"{np.corrcoef(v, M)[0, 1]:+.3f} | {np.abs(v).mean():.3f} | "
              f"{brier(v, Z) - best_b:.4f} |")

    # 校准表：拿最后一个 checkpoint
    path = args.ckpts[-1]
    if path in vals:
        v = vals[path]
        print(f"\n校准表（{path}）：")
        print("| 预测值区间 | 局面数 | 平均预测 | 实际得分率 |")
        print("|---|---|---|---|")
        edges = [-1.01, -0.5, -0.1, 0.1, 0.5, 1.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            if m.sum() == 0:
                continue
            print(f"| [{lo:+.1f}, {hi:+.1f}) | {int(m.sum())} | {v[m].mean():+.3f} | "
                  f"{(Z[m] + 1).mean() / 2:.3f} |")

        print("\n按 |子力差| 分组的 Brier（看价值头在哪类局面上更差）：")
        print("| |子力差| | 局面数 | 网络 Brier | 子力基线 Brier |")
        print("|---|---|---|---|")
        mb = np.clip(best_a * M, -1, 1)
        for lo, hi in ((0, 1.5), (1.5, 4.5), (4.5, 99)):
            m = (np.abs(M) >= lo) & (np.abs(M) < hi)
            if m.sum() == 0:
                continue
            print(f"| [{lo}, {hi}) | {int(m.sum())} | {brier(v[m], Z[m]):.4f} | "
                  f"{brier(mb[m], Z[m]):.4f} |")


if __name__ == "__main__":
    main()
