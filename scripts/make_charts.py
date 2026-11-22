"""给实验记录文档生成配图（中文标签用文泉驿正黑）。"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "reports")
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

if os.path.exists(FONT):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#999"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
BLUE, ORANGE, GREY = "#2b6cb0", "#c05621", "#888"


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def series(row_list, key="score"):
    if key == "greedy":                     # 早期文件是 {"greedy": {...}, "sf": {...}}
        return [r["greedy"]["score"] for r in row_list]
    return [r[key] for r in row_list]


def line_panel(ax, xs, ys, band, title, ylim=(0, 1)):
    ax.plot(xs, ys, "o-", color=BLUE, lw=1.8, ms=5)
    ax.fill_between(xs, [y - band for y in ys], [y + band for y in ys],
                    color=BLUE, alpha=0.15, lw=0)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_ylabel("得分（0.5 打平）", fontsize=9)


def chart_yardstick():
    old = rows(os.path.join(ROOT, "runs/main/gauntlet.jsonl"))
    new = rows(os.path.join(ROOT, "runs/main/gauntlet2.jsonl"))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), dpi=140)
    if old:
        line_panel(axes[0], list(range(len(old))), series(old, "greedy"), 0.346,
                   "旧尺子：8 局一点，阴影是 95% 区间", ylim=(0, 1))
        axes[0].set_xlabel("第几次评测", fontsize=9)
    if new:
        line_panel(axes[1], list(range(len(new))), series(new), 0.079,
                   "新尺子：40 局一点，阴影是 ±1 标准误", ylim=(0, 1))
        axes[1].set_xlabel("第几个评测点", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "chart_yardstick.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_loss():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), dpi=140)
    ax = axes[0]
    names = ["q1\n(240 sims)", "wide\n(512 主干)"]
    H = [1.362, 1.362]
    KL = [1.373, 1.425]
    TAIL = [1.032, 1.061]
    ax.bar(names, H, color="#cbd5e0", label="目标自身的熵")
    ax.bar(names, KL, bottom=H, color=BLUE, label="网络与目标的形状差")
    bottom2 = [a + b for a, b in zip(H, KL)]
    ax.bar(names, TAIL, bottom=bottom2, color=ORANGE, label="落在目标前 16 手之外")
    for i in range(2):
        ax.text(i, bottom2[i] + TAIL[i] / 2, f"{TAIL[i]:.2f}", ha="center", va="center",
                fontsize=8, color="white")
        ax.text(i, H[i] + KL[i] / 2, f"{KL[i]:.2f}", ha="center", va="center",
                fontsize=8, color="white")
    ax.set_title("一份损失拆成三块", fontsize=11)
    ax.set_ylim(0, 4.4)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    ax = axes[1]
    labels = ["240", "480", "800", "关噪声\n240"]
    vals = [3.96, 4.67, 4.21, 0.0]
    ax.bar(labels, vals, color=[BLUE, BLUE, BLUE, GREY])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title("同一局面搜两次，两份目标的差距（nats）", fontsize=11)
    ax.set_ylim(0, 5.4)
    ax.set_xlabel("每次模拟次数", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "chart_loss.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_curve(run, fname, title):
    r = rows(os.path.join(ROOT, f"runs/{run}/gauntlet2.jsonl"))
    if not r:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=140)
    line_panel(ax, list(range(len(r))), series(r), 0.079, title)
    for i, row in enumerate(r):
        ax.annotate(f"{row['score']:.3f}", (i, row["score"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xlabel("第几个评测点（每点 40 局、240 次模拟）", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_depth():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), dpi=140)
    ax = axes[0]
    labels = ["60", "120", "240", "480"]
    vals = [0.362, 0.525, 0.588, 0.700]
    ax.bar(labels, vals, color=BLUE)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_title("同一份权重，只改每步模拟次数", fontsize=11)
    ax.set_xlabel("模拟次数", fontsize=9)
    ax.set_ylabel("得分", fontsize=9)

    ax = axes[1]
    labels = ["纯策略\n不搜索", "策略\n+硬规则", "再搜索\n240 次"]
    vals = [0.013, 0.300, 0.588]
    ax.bar(labels, vals, color=[GREY, ORANGE, BLUE])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_title("棋力来自哪里", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "chart_depth.png")
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    made = [chart_yardstick(), chart_loss(),
            chart_curve("q1", "chart_q1.png", "打开叶子吃子修正 + 240 次模拟"),
            chart_curve("wide", "chart_wide.png", "策略主干 256 → 512"),
            chart_depth()]
    for p in made:
        print(p, os.path.getsize(p) if p else "缺失")
