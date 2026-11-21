"""极简 numpy 神经网络：14x90 平面 -> 256 -> 256 -> {policy 8100, value 1}

单核友好：纯 numpy 矩阵乘（AVX-512），Adam 优化器，自己写反向传播。
参数量约 2.5M，单样本前向 ~2.1 MMAC。
"""
import numpy as np

N_IN, H1, H2, N_POL = 18 * 90, 256, 256, 90 * 90   # 18 平面（14 棋子 + 4 威胁）


def zeros_init(rng, fan_in, shape):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


class Net:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": zeros_init(rng, N_IN, (N_IN, H1)), "b1": np.zeros(H1, np.float32),
            "W2": zeros_init(rng, H1, (H1, H2)), "b2": np.zeros(H2, np.float32),
            "Wp": (rng.standard_normal((H2, N_POL)) * 0.01).astype(np.float32),
            "bp": np.zeros(N_POL, np.float32),
            "Wv": zeros_init(rng, H2, (H2, 1)), "bv": np.zeros(1, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ------------------------------------------------------------ 前向
    def forward(self, x):
        """x: (B, 1260) -> logits (B, 8100), value (B, 1)"""
        p = self.p
        h1 = np.maximum(x @ p["W1"] + p["b1"], 0)
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        return h2 @ p["Wp"] + p["bp"], np.tanh(h2 @ p["Wv"] + p["bv"])

    def evaluate(self, x1):
        """单样本推理，返回 (logits 8100, value 标量)"""
        p = self.p
        h1 = np.maximum(x1 @ p["W1"] + p["b1"], 0)
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        return h2 @ p["Wp"] + p["bp"], float(np.tanh(h2 @ p["Wv"])[0])

    def evaluate_moves(self, x1, idx):
        """只算给定合法着法下标上的 logits（推理快 100 倍）；返回 (probs, value)。"""
        p = self.p
        h1 = np.maximum(x1 @ p["W1"] + p["b1"], 0)
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        sub = h2 @ p["Wp"][:, idx] + p["bp"][idx]
        sub = sub - sub.max()
        e = np.exp(sub)
        probs = e / e.sum()
        value = float(np.tanh(h2 @ p["Wv"] + p["bv"])[0])
        return probs, value

    def evaluate_sparse(self, feats, idx):
        """稀疏输入推理：只 gather 约 32 行 W1，避免读 1.3MB 权重。"""
        p = self.p
        h1 = p["W1"][feats].sum(0) + p["b1"]
        np.maximum(h1, 0, out=h1)
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        sub = h2 @ p["Wp"][:, idx] + p["bp"][idx]
        sub = sub - sub.max()
        e = np.exp(sub)
        probs = e / e.sum()
        value = float(np.tanh(h2 @ p["Wv"] + p["bv"])[0])
        return probs, value

    # ------------------------------------------------------------ 训练
    def train_step(self, x, pi, z, lr=1e-3, weight_decay=1e-4, policy_weight=1.0, pmask=None, vmask=None):
        """一步 Adam。pi: (B,8100) 稀疏后仍为稠密小批量；z: (B,1) in [-1,1]"""
        p, m, v = self.p, self.m, self.v
        B = x.shape[0]
        z1 = x @ p["W1"] + p["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.maximum(z2, 0)
        logits = h2 @ p["Wp"] + p["bp"]
        vval = np.tanh(h2 @ p["Wv"] + p["bv"])

        # softmax 交叉熵
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        prob = e / e.sum(axis=1, keepdims=True)
        loss_p = float(-(pi * np.log(prob + 1e-9)).sum(axis=1).mean())
        dlogits = (prob - pi) * (policy_weight / B)
        if pmask is not None:
            dlogits *= pmask.reshape(-1, 1)

        dv = (vval - z) * (1 - vval ** 2) * 2 / B      # tanh 导数
        if vmask is not None:
            dv *= vmask.reshape(-1, 1)
        loss_v = float(((vval - z) ** 2).mean())

        gWp = h2.T @ dlogits + weight_decay * p["Wp"]
        gbp = dlogits.sum(0)
        gWv = h2.T @ dv + weight_decay * p["Wv"]
        gbv = dv.sum(0)

        dh2 = dlogits @ p["Wp"].T + dv @ p["Wv"].T
        dz2 = dh2 * (z2 > 0)
        gW2 = h1.T @ dz2 + weight_decay * p["W2"]
        gb2 = dz2.sum(0)
        dh1 = dz2 @ p["W2"].T
        dz1 = dh1 * (z1 > 0)
        gW1 = x.T @ dz1 + weight_decay * p["W1"]
        gb1 = dz1.sum(0)

        self.t += 1
        t = self.t
        b1_, b2_ = 0.9, 0.999
        lr_t = lr * np.sqrt(1 - b2_ ** t) / (1 - b1_ ** t)
        for name, g in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2),
                        ("Wp", gWp), ("bp", gbp), ("Wv", gWv), ("bv", gbv)):
            m[name] = b1_ * m[name] + (1 - b1_) * g
            v[name] = b2_ * v[name] + (1 - b2_) * g * g
            p[name] -= (lr_t * m[name] / (np.sqrt(v[name]) + 1e-8)).astype(np.float32)
        return loss_p, loss_v

    def train_value(self, x, z, lr=2e-3):
        """只训练价值头（用于"子力先验"热启动）。"""
        p, m, v = self.p, self.m, self.v
        B = x.shape[0]
        z1 = x @ p["W1"] + p["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.maximum(z2, 0)
        vval = np.tanh(h2 @ p["Wv"] + p["bv"])
        loss = float(((vval - z) ** 2).mean())
        dv = (vval - z) * (1 - vval ** 2) * 2 / B
        gWv = h2.T @ dv
        gbv = dv.sum(0)
        dh2 = dv @ p["Wv"].T
        dz2 = dh2 * (z2 > 0)
        gW2 = h1.T @ dz2
        gb2 = dz2.sum(0)
        dh1 = dz2 @ p["W2"].T
        dz1 = dh1 * (z1 > 0)
        gW1 = x.T @ dz1
        gb1 = dz1.sum(0)
        self.t += 1
        t = self.t
        b1_, b2_ = 0.9, 0.999
        lr_t = lr * np.sqrt(1 - b2_ ** t) / (1 - b1_ ** t)
        for name, g in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2),
                        ("Wv", gWv), ("bv", gbv)):
            m[name] = b1_ * m[name] + (1 - b1_) * g
            v[name] = b2_ * v[name] + (1 - b2_) * g * g
            p[name] -= (lr_t * m[name] / (np.sqrt(v[name]) + 1e-8)).astype(np.float32)
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
