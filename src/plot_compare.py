"""三版横向对比图：对三个对手的得分率 + 版本间对局。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if os.path.exists(font):
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

data = {
    "night1（1356 局，1 进程）": {"随机": 0.60, "贪心": 0.10, "Stockfish": 0.00},
    "night2（6482 局，3 进程）": {"随机": 0.80, "贪心": 0.10, "Stockfish": 0.00},
    "night3（8880 局，修正后）": {"随机": 0.85, "贪心": 0.10, "Stockfish": 0.00},
}

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
fig.patch.set_facecolor("#0f1216")
ax.set_facecolor("#171c22")
names = list(data)
opps = ["随机", "贪心", "Stockfish"]
colors = ["#3fb950", "#d29922", "#e5534b"]
width = 0.26
for i, opp in enumerate(opps):
    vals = [data[n][opp] for n in names]
    xs = [j + (i - 1) * width for j in range(len(names))]
    bars = ax.bar(xs, vals, width, label=opp, color=colors[i])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", color="#e6edf3", fontsize=8)

ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("（", "\n（") for n in names], color="#8b98a5", fontsize=8)
ax.set_ylim(0, 1.0)
ax.set_ylabel("得分率（和棋算半分）", color="#8b98a5", fontsize=9)
ax.tick_params(colors="#8b98a5", labelsize=8)
for s in ax.spines.values():
    s.set_color("#2a323c")
ax.grid(axis="y", alpha=0.15, color="#8b98a5", linewidth=0.6)
ax.set_title("三版对比：局数翻 6 倍，赢不了会看威胁的对手", color="#e6edf3", fontsize=11)
ax.legend(facecolor="#171c22", edgecolor="#2a323c", labelcolor="#e6edf3", fontsize=8)
fig.text(0.5, 0.01, "版本间对局：night3 对 night2 得分 0.25，对 night1 得分 0.50（各 20 局）",
         ha="center", color="#8b98a5", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("reports/compare.png", facecolor=fig.get_facecolor())
print("saved reports/compare.png")
