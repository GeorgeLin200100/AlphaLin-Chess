"""把一个 checkpoint 的策略主干加宽（net2wider 的常见做法），尽量不改变初始输出。

做法：
- 价值主干原样复制（宽度不变）
- 策略主干旧权重放到左上角子块里
- 新增隐层单元的出边（Wp 的新行）乘一个很小的系数，所以加宽后的初始策略几乎不变，
  同时这些单元的梯度不为零（不会变成死单元）

用法（新宽度由环境变量给，默认 512/512）：
  POL_H1=512 POL_H2=512 .venv/bin/python scripts/widen.py \
      --src runs/main/model.npz --dst runs/wide/model.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import nn  # noqa: E402
from nn import Net, _init  # noqa: E402

EPS_OUT = 0.2        # 新增隐层单元的出边系数
SEED = 4242


def widen(src, dst):
    old = np.load(src)
    net = Net(SEED)
    rng = np.random.default_rng(SEED)
    p, o = net.p, old

    h1_old, h1_new = o["W1p"].shape[1], nn.H_POL
    h2_old, h2_new = o["W2p"].shape[1], nn.H2_POL
    if h1_old == h1_new and h2_old == h2_new:
        raise SystemExit("新旧宽度相同，不需要加宽")
    if h1_old > h1_new or h2_old > h2_new:
        raise SystemExit(f"目标宽度更小（{h1_new}/{h2_new} < {h1_old}/{h2_old}），不支持缩窄")

    # 策略主干
    p["W1p"][:, :h1_old] = o["W1p"]
    p["W1p"][:, h1_old:] = _init(rng, nn.N_IN, (nn.N_IN, h1_new - h1_old))
    p["b1p"][:h1_old] = o["b1p"]

    p["W2p"][:h1_old, :h2_old] = o["W2p"]
    p["W2p"][h1_old:, :h2_old] = _init(rng, h1_new, (h1_new - h1_old, h2_old))
    p["W2p"][:, h2_old:] = _init(rng, h1_new, (h1_new, h2_new - h2_old))
    p["b2p"][:h2_old] = o["b2p"]

    p["Wp"][:h2_old, :] = o["Wp"]
    p["Wp"][h2_old:, :] = (rng.standard_normal((h2_new - h2_old, nn.N_POL)) * 0.01 * EPS_OUT).astype(np.float32)
    p["bp"][:] = o["bp"]

    # 价值主干：形状没变，整体复制
    for k in ("W1v", "b1v", "W2v", "b2v", "Wv", "bv"):
        if p[k].shape != o[k].shape:
            raise SystemExit(f"{k} 形状变了（{p[k].shape} vs {o[k].shape}），价值主干应保持不变")
        p[k] = o[k].astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    net.save(dst)
    return dict(h1=(h1_old, h1_new), h2=(h2_old, h2_new))


def policy_shift(src, dst, n=256):
    """量一下加宽前后在同一批输入上的策略差异。"""
    import learner

    old = Net.load(src)
    new = Net.load(dst)
    files = sorted(__import__("glob").glob(os.path.join(HERE, "..", "runs", "main", "queue", "*.npz")))[-3:]
    xs = []
    for f in files:
        for item in learner.load_queue_file(f):
            xs.append(item[0])
            if len(xs) >= n:
                break
        if len(xs) >= n:
            break
    X = np.asarray(xs, dtype=np.float32)
    lo, _ = old.forward(X)
    ln, _ = new.forward(X)
    po = np.exp(lo - lo.max(1, keepdims=True))
    po /= po.sum(1, keepdims=True)
    pn = np.exp(ln - ln.max(1, keepdims=True))
    pn /= pn.sum(1, keepdims=True)
    kl = float((po * (np.log(po + 1e-9) - np.log(pn + 1e-9))).sum(1).mean())
    top1 = float((lo.argmax(1) == ln.argmax(1)).mean())
    # 价值头应完全一致
    _, vo = old.forward(X)
    _, vn = new.forward(X)
    vmax = float(np.abs(vo - vn).max())
    return dict(n=len(X), policy_kl=kl, top1_agree=top1, value_maxdiff=vmax)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/main/model.npz")
    ap.add_argument("--dst", default="runs/wide/model.npz")
    args = ap.parse_args()
    info = widen(args.src, args.dst)
    print("加宽完成:", info)
    print("策略偏移:", policy_shift(args.src, args.dst))
