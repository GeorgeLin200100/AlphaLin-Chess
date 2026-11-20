# rl-chess —— 中国象棋自博弈（AlphaZero 风格）学习项目

目标不是造强引擎，而是**把"越下越好"这件事完整跑通并看清楚**。
原理说明：`docs/一个晚上看懂AlphaZero.md`；可行性评估：`reports/可行性评估.md`。

## 快速开始

```bash
cd /root/rl-chess
export PIP_CONFIG_FILE=/root/rl-chess/pip.conf    # 本机 IPv6 无路由，pip 建议走清华镜像

# 1) 规则引擎自检（必须 44 / 1,920 / 79,666 / 3,290,240 全对）
.venv/bin/python src/xiangqi.py 1 2 3 4

# 2) 训练（单核；--hours 控制时长；--smoke 约 1 分钟冒烟）
OMP_NUM_THREADS=1 .venv/bin/python src/train.py --hours 5.7 --sims 160 --eval-sims 100 \
  --steps 8 --lr 1e-3 --policy-weight 0.1 --anchor-prob 0.3 --warmstart 300 --run runs/main

# 3) 看板 + 人机对弈 + 报告
.venv/bin/python src/serve.py     # http://127.0.0.1:8000/ 与 /play
.venv/bin/python src/report.py    # reports/curves.png + web/snapshot.html
```

## 目录

```
src/xiangqi.py   规则引擎（9x10 全规则；perft 对齐权威值；~200k 节点/s）
src/mcts.py      PUCT 搜索（单线程，懒拷贝棋盘，~3k 节点/s）
src/nn.py        numpy 手写 MLP + Adam（1260→256→256→{8100,1}，约 2.5M 参数）
src/train.py     自博弈 + 训练 + 门控 + 统计
src/serve.py     看板 / 对弈服务器（标准库，无外部依赖）
src/report.py    曲线图 / 报告 / 离线看板快照
web/index.html   训练看板（胜率、损失、局数、对局回放）
web/play.html    和模型下棋
runs/main/       model.npz（最新）best.npz（门控后最佳）stats.jsonl status.json games/
bench/           吞吐与正确性基准（国象 / numpy 网络 / cchess 对照）
docs/            原理说明
reports/         可行性评估 + 生成的曲线
logs/            训练日志
```

## 关键设计（为什么这样能学会）

- **搜索当老师**：MCTS 访问次数分布 π 比网络直觉更强，网络去模仿它 —— 这个循环就是"越下越好"。
- **子力裁定**：160 步或 100 步无吃子时按子力判胜负。没有它，弱网络之间几乎全是和棋（实测第一版 24/26 局和棋），`z=0` 没有梯度。
- **门控**：新模型对旧模型胜率 ≥55% 才替换，防止回退。
- **镜像增强**：左右对称，样本翻倍。
- **稀疏策略目标**：每步只存访问次数前 16 的着法，缓冲区省 30 倍内存。

## 已知坑

- **pip 能用但直连 pypi 慢**（实测 1.15 KB/s，IPv6 AAAA 会让 pip 偶尔挂死）；已全局配置 `~/.config/pip/pip.conf` 走清华镜像（实测 17 MB/s）。
- **后台 shell 里的 curl 会走 `127.0.0.1` 代理并失败**；下载文件请放前台跑。
- **torch 未安装**：`torch-2.9.1+cpu-cp314` wheel 184 MB，`/tmp/torch_cpu.whl` 只到 125 MB。本项目的网络是 numpy 手写，不需要 torch：
  `curl -4 -o /tmp/torch_cpu.whl "https://download.pytorch.org/whl/cpu/torch-2.9.1%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl" && .venv/bin/pip install --no-index /tmp/torch_cpu.whl`
- **`cchess` 库不能用**：perft(3) 75,652 ≠ 79,666（少 5% 合法着法）。本项目的 `src/xiangqi.py` 已重写并通过 perft 1–4 验证。
- 无 GPU；单核约 3k 节点/s，一夜约 4,000 局，棋力上限见评估文档。
