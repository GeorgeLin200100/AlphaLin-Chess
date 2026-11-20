"""生成训练报告：曲线图（PNG）+ Markdown 摘要 + 可离线打开的看板快照。

用法:
    .venv/bin/python src/report.py [--run runs/main]
产物:
    reports/curves.png          4 联图（发飞书用）
    reports/report.md           文字报告
    web/snapshot.html           内嵌数据的单文件看板
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 中文字体
for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False


def load_rows(run, include_phase0=False):
    rows = []
    names = ("stats_phase0.jsonl", "stats.jsonl") if include_phase0 else ("stats.jsonl",)
    for name in names:
        p = os.path.join(run, name)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return rows


def load_games(run, limit=400):
    d = os.path.join(run, "games")
    out = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d))[-limit:]:
            try:
                with open(os.path.join(d, name)) as f:
                    g = json.load(f)
                g["name"] = name
                out.append(g)
            except Exception:
                pass
    return out


def plot(rows, games, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=140)
    fig.patch.set_facecolor("#0f1216")
    for ax in axes.ravel():
        ax.set_facecolor("#171c22")
        ax.tick_params(colors="#8b98a5", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2a323c")
        ax.grid(alpha=0.15, color="#8b98a5", linewidth=0.6)
        ax.title.set_color("#e6edf3")

    # 1) 对随机棋手胜率
    ax = axes[0][0]
    wr = [(r["games"], r["winrate_random"]) for r in rows if r.get("winrate_random") is not None]
    if len(wr) == 1:
        ax.scatter([wr[0][0]], [wr[0][1]], color="#3fb950", s=18)
        ax.set_xlim(wr[0][0] - 5, wr[0][0] + 5)
    elif wr:
        ax.plot(*zip(*wr), color="#3fb950", lw=1.8, label="胜率")
        dr = [(r["games"], r.get("draw_random", 0)) for r in rows if r.get("winrate_random") is not None]
        ax.plot(*zip(*dr), color="#8b98a5", lw=1.2, ls="--", label="和棋率")
        ax.set_ylim(0, 1.05)
    ax.set_title("对随机棋手的胜率（和棋算半分）")
    ax.legend(facecolor="#171c22", edgecolor="#2a323c", labelcolor="#e6edf3", fontsize=8)

    # 2) 损失
    ax = axes[0][1]
    lp = [(r["games"], r["loss_p"]) for r in rows if r.get("loss_p")]
    lv = [(r["games"], r["loss_v"]) for r in rows if r.get("loss_v") is not None]
    if lp:
        ax.plot(*zip(*lp), color="#58a6ff", lw=1.4, label="policy loss")
    if lv:
        ax.plot(*zip(*lv), color="#d29922", lw=1.4, label="value loss")
    ax.set_title("训练损失")
    ax.legend(facecolor="#171c22", edgecolor="#2a323c", labelcolor="#e6edf3", fontsize=8)

    # 3) 对局数 + 结果构成
    ax = axes[1][0]
    if rows:
        ax.plot([r["iter"] for r in rows], [r["games"] for r in rows], color="#58a6ff", lw=1.4)
        ax.set_title("累计自博弈局数")
    # 4) 结果构成（柱状）
    ax = axes[1][1]
    red = sum(r["results"].get("red", 0) for r in rows)
    black = sum(r["results"].get("black", 0) for r in rows)
    draw = sum(r["results"].get("draw", 0) for r in rows)
    ax.bar(["红胜", "黑胜", "和棋"], [red, black, draw],
           color=["#e5534b", "#58a6ff", "#8b98a5"])
    ax.set_title(f"结果构成（共 {red+black+draw} 局）")

    fig.suptitle("中国象棋自博弈训练", color="#e6edf3", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_snapshot(rows, games, out_html):
    src = os.path.join(ROOT, "web", "index.html")
    if not os.path.exists(src):
        return
    with open(src, encoding="utf-8") as f:
        html = f.read()
    payload = {
        "rows": rows[-2000:],
        "status": rows[-1] if rows else {},
        "alive": False,
        "games": games[-12:],
    }
    embed = ("<script>window.__EMBEDDED__ = " +
             json.dumps(payload, ensure_ascii=False) + ";</script>\n")
    html = html.replace("<script>\nconst CH", embed + "<script>\nconst CH")
    html = html.replace("const r = await fetch('/api/stats').then(r=>r.json());",
                        "const r = (window.__EMBEDDED__ && !location.protocol.startsWith('http'))"
                        " ? window.__EMBEDDED__ : await fetch('/api/stats').then(r=>r.json());")
    html = html.replace("games = await fetch('/api/games').then(r=>r.json());",
                        "games = window.__EMBEDDED__ ? window.__EMBEDDED__.games"
                        " : await fetch('/api/games').then(r=>r.json());")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "main"))
    ap.add_argument("--include-phase0", action="store_true")
    args = ap.parse_args()
    rows = load_rows(args.run, args.include_phase0)
    games = load_games(args.run)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    png = os.path.join(ROOT, "reports", "curves.png")
    plot(rows, games, png)
    build_snapshot(rows, games, os.path.join(ROOT, "web", "snapshot.html"))

    last = rows[-1] if rows else {}
    total = sum(r["results"].get("red", 0) + r["results"].get("black", 0) +
                r["results"].get("draw", 0) for r in rows)
    hours = sum(r.get("selfplay_s", 0) + r.get("train_s", 0) for r in rows) / 3600
    evals = [r["winrate_random"] for r in rows if r.get("winrate_random") is not None]
    import statistics
    first = statistics.mean(evals[:8]) if evals else float("nan")
    mid = statistics.mean(evals[len(evals)//2-4:len(evals)//2+4]) if len(evals) > 8 else float("nan")
    tail = statistics.mean(evals[-8:]) if evals else float("nan")
    adjud = last.get("adjudicated", None)
    lines = [
        "# 中象自博弈训练报告",
        "",
        f"- 总迭代 {last.get('iter', 0)}　总对局 {total}　用时 {hours:.1f} 小时（{total/max(hours,1e-6):.0f} 局/小时）",
        f"- 对随机棋手胜率：前 8 次 {first:.2f} → 中段 {mid:.2f} → 最近 8 次 {tail:.2f}",
        f"- 结果构成：红胜 {last['results'].get('red', 0)} / 黑胜 {last['results'].get('black', 0)} / 和棋 {last['results'].get('draw', 0)}（最后一轮）",
        f"- 最近 policy/value loss: {last.get('loss_p')} / {last.get('loss_v')}",
        f"- 曲线图: reports/curves.png　离线看板: web/snapshot.html",
    ]
    with open(os.path.join(ROOT, "reports", "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"charts -> {png}")


if __name__ == "__main__":
    main()
