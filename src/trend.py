"""读 gauntlet2.jsonl，判断"棋力有没有在涨"，给继续/停止的结论。

判定阈值（事先定好，不事后解释）：
  成功：末点 - 首点 >= +0.10  且  同开局配对符号检验 p < 0.10
  退步：末点 - 首点 <= -0.10
  无效：其余情况（含 p >= 0.10）—— 说明这套配置 3 小时里没有可测的进步

用法: .venv/bin/python src/trend.py [--run runs/main]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DELTA_OK = 0.10
P_OK = 0.10


def load(run_dir):
    p = os.path.join(run_dir, "gauntlet2.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def binom_two_sided(k, n):
    """符号检验：n 次配对里 k 次朝同一方向，双侧 p 值。"""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _score(r):
    """单局得分：赢 1 / 和 0.5 / 输 0（r 是 res，r['side'] 是我们执的一方）。"""
    if r["res"] == 0:
        return 0.5
    return 1.0 if r["res"] == r["side"] else 0.0


def paired(first, last):
    a = {(r["o"], r["side"]): _score(r) for r in first["per_game"]}
    b = {(r["o"], r["side"]): _score(r) for r in last["per_game"]}
    common = sorted(set(a) & set(b))
    better = worse = same = 0
    for k in common:
        if b[k] > a[k]:
            better += 1
        elif b[k] < a[k]:
            worse += 1
        else:
            same += 1
    return better, worse, same, len(common)


def slope_per_hour(pts):
    """pts: [(minutes, score)] -> 每小时得分变化（最小二乘）。"""
    n = len(pts)
    if n < 2:
        return 0.0
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den == 0:
        return 0.0
    return num / den * 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "..", "runs", "main"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run)
    rows = load(run_dir)
    if not rows:
        print(f"{run_dir}/gauntlet2.jsonl 没有数据")
        return

    t0 = rows[0]["ts"]
    pts = [((r["ts"] - t0) / 60, r["score"]) for r in rows]
    scores = [r["score"] for r in rows]
    n_games = rows[0]["games"]
    se = math.sqrt(0.25 / n_games)

    lines = []
    add = lines.append
    add(f"# 棋力趋势（对手：一步贪心，固定 {len(rows[0]['per_game']) // 2} 个开局，"
        f"每点 {n_games} 局）")
    add("")
    add("| 时刻 | 分钟 | 自博弈局数 | 胜/和/负 | 得分 |")
    add("|---|---|---|---|---|")
    for (m, s), r in zip(pts, rows):
        add(f"| {r['t']} | {m:.0f} | {r.get('selfplay_games')} | "
            f"{r['win']}/{r['draw']}/{r['loss']} | {s:.3f} |")
    add("")

    if len(rows) < 2:
        verdict = "数据不足：只有一个评测点，无法判断趋势。"
        add(verdict)
    else:
        delta = scores[-1] - scores[0]
        b, w, same, common = paired(rows[0], rows[-1])
        n_dec = b + w
        p = binom_two_sided(max(b, w), n_dec) if n_dec else 1.0
        sl = slope_per_hour(pts)
        add(f"- 起点 {scores[0]:.3f} -> 末点 {scores[-1]:.3f}，变化 {delta:+.3f}"
            f"（单点噪声 ±{se:.3f}）")
        add(f"- 最小二乘斜率 {sl:+.3f}/小时；局内一致性 ±{se:.3f}")
        add(f"- 同开局配对：{common} 局可比，{b} 局变好 / {w} 局变坏 / {same} 局相同，"
            f"符号检验 p = {p:.3f}")
        add("")
        if delta >= DELTA_OK and p < P_OK:
            verdict = (f"**结论：在学。** 得分涨了 {delta:+.3f} 且配对检验显著（p={p:.3f}），"
                       f"按事先阈值判为有效进步，建议继续跑同配置、把时长拉长。")
        elif delta <= -DELTA_OK:
            verdict = (f"**结论：退步。** 得分掉了 {delta:+.3f}。先别加时长，"
                       f"该查学习率/门控/安全过滤是否把策略带偏。")
        else:
            verdict = (f"**结论：无效。** 变化 {delta:+.3f} 落在噪声带内、配对 p={p:.3f}，"
                       f"这套配置 3 小时里没有可测的进步。按事先约定：停止当前配置，"
                       f"换方向（提高 sims、加大自博弈量级，或换更强的对照与训练目标）。")
        add(verdict)

    text = "\n".join(lines)
    os.makedirs(os.path.join(HERE, "..", "reports"), exist_ok=True)
    md = os.path.join(HERE, "..", "reports", "trend.md")
    with open(md, "w") as f:
        f.write(text + "\n")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
            xs = [p[0] for p in pts]
            ax.plot(xs, scores, "o-", color="#2b6cb0", lw=1.8, ms=5, label="vs 贪心")
            ax.fill_between(xs, [s - se for s in scores], [s + se for s in scores],
                            color="#2b6cb0", alpha=0.15, lw=0)
            ax.axhline(0.5, color="#999", ls="--", lw=1)
            ax.set_xlabel("训练时间 (min)")
            ax.set_ylabel("得分率")
            ax.set_title("固定开局评测（每点 %d 局，阴影 ±1 SE）" % n_games)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            png = os.path.join(HERE, "..", "reports", "trend.png")
            fig.savefig(png)
            print(png)
        except Exception as e:
            print("画图失败:", repr(e))

    print(text)


if __name__ == "__main__":
    main()
