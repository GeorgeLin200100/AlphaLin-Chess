"""策略损失体检：把 loss_p 拆成三块，看它到底卡在哪。

trainer 的 loss 可以精确分解（不是近似）：
    loss_p = -Σ_top16 π·log p
           = H(π)                 目标自身的熵，只要目标不变，这项就不可能降
           + KL(π ‖ p/s)          网络在目标支撑集上的形状差多少
           + (−log s)             网络放到目标支撑集之外的概质量（s = p 在 top16 上的和）

loss 看不见、但直接决定棋力的是排序：
    top1 一致率、top5 覆盖率、网络给目标首选着法的概率

一个进程处理一个 checkpoint（因为主干宽度决定网络结构，必须在 import nn 前设好环境变量）。
批量用法见 scripts/diag_all.sh，或直接：
  .venv/bin/python scripts/diag_policy.py --ckpt runs/q1/model.npz --queue runs/q1/queue
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_samples(queue_dir, want):
    xs, idxs, ps, zs = [], [], [], []
    files = sorted(glob.glob(os.path.join(queue_dir, "*.npz")))[-150:]
    for f in files:
        d = np.load(f)
        for x, i, p, z in zip(d["X"], d["pi_idx"], d["pi_p"], d["Z"]):
            xs.append(x), idxs.append(i), ps.append(p), zs.append(z)
            if len(xs) >= want:
                return (np.asarray(xs, dtype=np.float32), np.asarray(idxs),
                        np.asarray(ps, dtype=np.float32), np.asarray(zs, dtype=np.float32))
    return (np.asarray(xs, dtype=np.float32), np.asarray(idxs),
            np.asarray(ps, dtype=np.float32), np.asarray(zs, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--queue", default="runs/wide/queue")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    # 先读宽度，再 import nn（nn 用环境变量决定主干宽度）
    with np.load(args.ckpt) as d:
        w1, w2 = d["W1p"].shape[1], d["W2p"].shape[0]
    os.environ["POL_H1"], os.environ["POL_H2"] = str(w1), str(w2)
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from nn import Net  # noqa: E402

    X, idx, pi, Z = load_samples(args.queue, args.n)
    net = Net.load(args.ckpt)

    logits, val = net.forward(X)
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    prob = np.exp(logp)

    valid = idx >= 0
    pi = pi * valid
    pi = pi / np.maximum(pi.sum(axis=1, keepdims=True), 1e-12)
    rows = np.arange(len(X))[:, None]
    p_sup = np.where(valid, prob[rows, np.maximum(idx, 0)], 0.0)
    s = np.clip(p_sup.sum(axis=1), 1e-12, 1.0)

    log_pi = np.where(valid, np.log(np.maximum(pi, 1e-12)), 0.0)
    log_ps = np.where(valid, np.log(np.maximum(p_sup, 1e-12)), 0.0)
    H = -(np.where(valid, pi * log_pi, 0.0)).sum(1)
    loss_total = -(np.where(valid, pi * log_ps, 0.0)).sum(1)
    kl = (np.where(valid, pi * (log_pi - (log_ps - np.log(s)[:, None])), 0.0)).sum(1)
    tail = -np.log(s)

    mask_net = np.where(valid, p_sup, -1.0)
    top1 = float((mask_net.argmax(1) == pi.argmax(1)).mean())
    order_pi = np.argsort(-pi, axis=1)
    order_net = np.argsort(-mask_net, axis=1)
    cov = float(np.mean([len(set(order_pi[r, :5]) & set(order_net[r, :5])) / 5.0
                         for r in range(len(X))]))
    p_mass = np.where(valid, p_sup, 0.0)

    v = val[:, 0]
    out = dict(ckpt=args.ckpt, n=int(len(X)), width=int(w1),
               loss_p=float(loss_total.mean()), H=float(H.mean()),
               kl=float(kl.mean()), tail=float(tail.mean()),
               top1=top1, top5=cov,
               pi_top1_mass=float(pi.max(1).mean()),
               net_on_target_top1=float(p_mass.max(1).mean()),
               value_brier=float(((v - Z) ** 2).mean()),
               value_corr=float(np.corrcoef(v, Z)[0, 1]) if v.std() > 1e-9 else 0.0,
               value_std=float(v.std()))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
