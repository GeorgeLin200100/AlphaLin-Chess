"""过拟合体检：拿一批固定数据，反复训同一批样本，看网络能不能把 loss 压下去。

这个测试分开了两件事：
  能压下去  -> 优化器和容量没问题，loss 高是数据/目标造成的（目标总在变、目标本身带噪声）
  压不下去  -> 优化本身有问题（学习率、初始化、梯度尺度），换目标或换容量都没用

用法: .venv/bin/python scripts/fit_probe.py --ckpt runs/q1/model.npz --steps 600 --weight 0.1
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/q1/model.npz")
    ap.add_argument("--queue", default="runs/q1/queue")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight", type=float, default=0.1)
    ap.add_argument("--report-every", type=int, default=100)
    args = ap.parse_args()

    with np.load(args.ckpt) as d:
        w = d["W1p"].shape[1]
    os.environ["POL_H1"] = os.environ["POL_H2"] = str(w)
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from nn import Net, N_POL  # noqa: E402

    X, IDX, PR, Z = [], [], [], []
    for f in sorted(glob.glob(os.path.join(args.queue, "*.npz")))[-120:]:
        d = np.load(f)
        for x, i, p, z in zip(d["X"], d["pi_idx"], d["pi_p"], d["Z"]):
            X.append(x), IDX.append(i), PR.append(p), Z.append(z)
            if len(X) >= args.n:
                break
        if len(X) >= args.n:
            break
    X = np.asarray(X, dtype=np.float32)
    IDX = np.asarray(IDX)
    PR = np.asarray(PR, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32).reshape(-1, 1)

    PI = np.zeros((len(X), N_POL), dtype=np.float32)
    rows = np.repeat(np.arange(len(X)), IDX.shape[1])
    valid = IDX.reshape(-1) >= 0
    PI[rows[valid], IDX.reshape(-1)[valid]] = PR.reshape(-1)[valid]
    PI /= np.maximum(PI.sum(1, keepdims=True), 1e-12)
    print(f"数据 {len(X)} 条，宽 {w}，lr {args.lr}，policy_weight {args.weight}，"
          f"训练 {args.steps} 步（同一批数据反复用）")

    net = Net.load(args.ckpt)
    rng = np.random.default_rng(7)

    def measure():
        logits, _ = net.forward(X)
        logits -= logits.max(1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        prob = np.exp(logp)
        sup = np.take_along_axis(prob, np.maximum(IDX, 0), axis=1)
        validm = IDX >= 0
        ps = np.where(validm, sup, 0.0)
        s = np.clip(ps.sum(1), 1e-12, 1.0)
        loss = -float((PI * logp).sum(1).mean())
        H = -float((PI * np.log(np.maximum(PI, 1e-12))).sum(1).mean())
        net_top = np.where(validm, ps, -1.0).argmax(1)
        pi_on_sup = np.take_along_axis(PI, np.maximum(IDX, 0), axis=1)
        pi_top = np.where(validm, pi_on_sup, -1.0).argmax(1)   # 都在"前 16 个槽位"这个坐标系里比
        top1 = float((net_top == pi_top).mean())
        # 网络在目标支撑集上的概率质量（1 - 尾巴）
        return loss, H, top1, float(s.mean())

    loss0, H, top10, s0 = measure()
    print(f"step    0  loss {loss0:.3f}  top1 {top10:.1%}  支撑集内质量 {s0:.3f}  "
          f"(目标熵 {H:.3f} 是下限)")

    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(X), size=args.batch)
        net.train_step(X[idx], PI[idx], Z[idx], lr=args.lr, policy_weight=args.weight)
        if step % args.report_every == 0:
            loss, _, top1, s = measure()
            print(f"step {step:4d}  loss {loss:.3f}  top1 {top1:.1%}  支撑集内质量 {s:.3f}")

    loss, _, top1, s = measure()
    print(f"\n结果：loss {loss0:.3f} -> {loss:.3f}（下限 {H:.3f}），"
          f"top1 {top10:.1%} -> {top1:.1%}，支撑集内质量 {s0:.3f} -> {s:.3f}")


if __name__ == "__main__":
    main()
