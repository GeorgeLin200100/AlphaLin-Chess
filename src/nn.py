"""numpy 网络：双主干（策略一套隐层，价值一套隐层，互不干扰）。

输入 18 平面 × 90 = 1620 维。

为什么拆：之前两个头共用一个隐层，策略头有 8100 个输出，梯度量级远大于价值头
（1 个输出）。训练步数一多，主干被改写成"专门服务策略"的形状，价值头读到的
特征被冲掉。实测价值与子力的相关性从 +0.42 掉到 -0.07，棋力反而比热启动差。
拆开之后两条梯度各走各的路，互不覆盖。
"""
import os

import numpy as np

# 策略主干宽度可用环境变量覆盖（POL_H1 / POL_H2），默认 256 保持旧 checkpoint 兼容。
N_IN = 18 * 90
H_POL = int(os.environ.get("POL_H1", 256))
H2_POL = int(os.environ.get("POL_H2", 256))
H_VAL, H2_VAL = 192, 128                     # 价值主干
N_POL = 90 * 90


def _init(rng, fan_in, shape):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


class Net:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            # 策略主干
            "W1p": _init(rng, N_IN, (N_IN, H_POL)), "b1p": np.zeros(H_POL, np.float32),
            "W2p": _init(rng, H_POL, (H_POL, H2_POL)), "b2p": np.zeros(H2_POL, np.float32),
            "Wp": (rng.standard_normal((H2_POL, N_POL)) * 0.01).astype(np.float32),
            "bp": np.zeros(N_POL, np.float32),
            # 价值主干
            "W1v": _init(rng, N_IN, (N_IN, H_VAL)), "b1v": np.zeros(H_VAL, np.float32),
            "W2v": _init(rng, H_VAL, (H_VAL, H2_VAL)), "b2v": np.zeros(H2_VAL, np.float32),
            "Wv": _init(rng, H2_VAL, (H2_VAL, 1)), "bv": np.zeros(1, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ------------------------------------------------------------ 前向
    def forward(self, x):
        """x: (B, 1620) -> logits (B, 8100), value (B, 1)"""
        p = self.p
        h1 = np.maximum(x @ p["W1p"] + p["b1p"], 0)
        h2 = np.maximum(h1 @ p["W2p"] + p["b2p"], 0)
        logits = h2 @ p["Wp"] + p["bp"]
        g1 = np.maximum(x @ p["W1v"] + p["b1v"], 0)
        g2 = np.maximum(g1 @ p["W2v"] + p["b2v"], 0)
        return logits, np.tanh(g2 @ p["Wv"] + p["bv"])

    def evaluate_sparse(self, feats, idx):
        """稀疏输入推理：只 gather 活跃行。返回 (probs over idx, value)。"""
        p = self.p
        h1 = p["W1p"][feats].sum(0) + p["b1p"]
        np.maximum(h1, 0, out=h1)
        h2 = np.maximum(h1 @ p["W2p"] + p["b2p"], 0)
        sub = h2 @ p["Wp"][:, idx] + p["bp"][idx]
        sub = sub - sub.max()
        e = np.exp(sub)
        probs = e / e.sum()
        g1 = p["W1v"][feats].sum(0) + p["b1v"]
        np.maximum(g1, 0, out=g1)
        g2 = np.maximum(g1 @ p["W2v"] + p["b2v"], 0)
        value = float(np.tanh(g2 @ p["Wv"] + p["bv"])[0])
        return probs, value

    # ------------------------------------------------------------ 训练
    def _adam(self, name, g, lr):
        b1_, b2_ = 0.9, 0.999
        t = self.t
        lr_t = lr * np.sqrt(1 - b2_ ** t) / (1 - b1_ ** t)
        self.m[name] = b1_ * self.m[name] + (1 - b1_) * g
        self.v[name] = b2_ * self.v[name] + (1 - b2_) * g * g
        self.p[name] -= (lr_t * self.m[name] / (np.sqrt(self.v[name]) + 1e-8)).astype(np.float32)

    def train_step(self, x, pi, z, lr=1e-3, weight_decay=1e-4, policy_weight=1.0,
                   pmask=None, vmask=None):
        """策略梯度只走策略主干，价值梯度只走价值主干。"""
        p = self.p
        B = x.shape[0]
        self.t += 1

        # ---- 策略分支
        z1 = x @ p["W1p"] + p["b1p"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ p["W2p"] + p["b2p"]
        h2 = np.maximum(z2, 0)
        logits = h2 @ p["Wp"] + p["bp"]
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        prob = e / e.sum(axis=1, keepdims=True)
        loss_p = float(-(pi * np.log(prob + 1e-9)).sum(axis=1).mean())
        dlogits = (prob - pi) * (policy_weight / B)
        if pmask is not None:
            dlogits *= pmask.reshape(-1, 1)

        gWp = h2.T @ dlogits + weight_decay * p["Wp"]
        gbp = dlogits.sum(0)
        dh2 = dlogits @ p["Wp"].T
        dz2 = dh2 * (z2 > 0)
        gW2 = h1.T @ dz2 + weight_decay * p["W2p"]
        gb2 = dz2.sum(0)
        dh1 = dz2 @ p["W2p"].T
        dz1 = dh1 * (z1 > 0)
        gW1 = x.T @ dz1 + weight_decay * p["W1p"]
        gb1 = dz1.sum(0)
        for name, g in (("W1p", gW1), ("b1p", gb1), ("W2p", gW2), ("b2p", gb2),
                        ("Wp", gWp), ("bp", gbp)):
            self._adam(name, g, lr)

        # ---- 价值分支
        y1 = x @ p["W1v"] + p["b1v"]
        k1 = np.maximum(y1, 0)
        y2 = k1 @ p["W2v"] + p["b2v"]
        k2 = np.maximum(y2, 0)
        vval = np.tanh(k2 @ p["Wv"] + p["bv"])
        loss_v = float(((vval - z) ** 2).mean())
        dv = (vval - z) * (1 - vval ** 2) * 2 / B
        if vmask is not None:
            dv *= vmask.reshape(-1, 1)
        gWv = k2.T @ dv + weight_decay * p["Wv"]
        gbv = dv.sum(0)
        dk2 = dv @ p["Wv"].T
        dy2 = dk2 * (y2 > 0)
        gW2v = k1.T @ dy2 + weight_decay * p["W2v"]
        gb2v = dy2.sum(0)
        dk1 = dy2 @ p["W2v"].T
        dy1 = dk1 * (y1 > 0)
        gW1v = x.T @ dy1 + weight_decay * p["W1v"]
        gb1v = dy1.sum(0)
        for name, g in (("W1v", gW1v), ("b1v", gb1v), ("W2v", gW2v), ("b2v", gb2v),
                        ("Wv", gWv), ("bv", gbv)):
            self._adam(name, g, lr)
        return loss_p, loss_v

    def train_value(self, x, z, lr=2e-3):
        """只训练价值主干（热启动用）。"""
        p = self.p
        B = x.shape[0]
        self.t += 1
        y1 = x @ p["W1v"] + p["b1v"]
        k1 = np.maximum(y1, 0)
        y2 = k1 @ p["W2v"] + p["b2v"]
        k2 = np.maximum(y2, 0)
        vval = np.tanh(k2 @ p["Wv"] + p["bv"])
        loss = float(((vval - z) ** 2).mean())
        dv = (vval - z) * (1 - vval ** 2) * 2 / B
        gWv = k2.T @ dv
        gbv = dv.sum(0)
        dk2 = dv @ p["Wv"].T
        dy2 = dk2 * (y2 > 0)
        gW2v = k1.T @ dy2
        gb2v = dy2.sum(0)
        dk1 = dy2 @ p["W2v"].T
        dy1 = dk1 * (y1 > 0)
        gW1v = x.T @ dy1
        gb1v = dy1.sum(0)
        for name, g in (("W1v", gW1v), ("b1v", gb1v), ("W2v", gW2v), ("b2v", gb2v),
                        ("Wv", gWv), ("bv", gbv)):
            self._adam(name, g, lr)
        return loss

    # ------------------------------------------------------------ 存取
    def save(self, path):
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path):
        net = cls(0)
        d = np.load(path)
        for k in net.p:
            net.p[k] = d[k].astype(np.float32)
        return net
