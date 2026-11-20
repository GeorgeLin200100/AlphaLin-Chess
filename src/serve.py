"""本地看板 + 人机对弈服务器（标准库 http.server，无外部依赖）。

用法:
    .venv/bin/python src/serve.py            # http://0.0.0.0:8000
    .venv/bin/python src/serve.py --port 9000 --sims 200

页面:
    /            训练看板（曲线 + 状态 + 对局回放）
    /play        和模型下棋（点两下走子）
"""
import argparse
import json
import os
import sys
import threading

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
from xiangqi import Board, START_FEN, move_to_index, index_to_move
from nn import Net
from mcts import search

WEB = os.path.join(ROOT, "web")
RUN = os.path.join(ROOT, "runs", "main")

_lock = threading.Lock()
_model_path = {"p": os.path.join(RUN, "best.npz")}
_net_cache = {"mtime": None, "net": None}
_game = {"board": Board(START_FEN), "last": None, "engine_value": None, "engine_visits": None}


def get_net():
    """按文件 mtime 热加载模型。"""
    path = _model_path["p"]
    if not os.path.exists(path):
        path = os.path.join(RUN, "model.npz")
    if not os.path.exists(path):
        return Net(0)
    mt = os.path.getmtime(path)
    if _net_cache["mtime"] != mt or _net_cache["net"] is None:
        _net_cache["net"] = Net.load(path)
        _net_cache["mtime"] = mt
    return _net_cache["net"]


def board_json(board, extra=None):
    d = {
        "fen": board.fen(),
        "side": "red" if board.side == 1 else "black",
        "legal": [[f, t] for f, t in board.legal_moves()],
        "move_count": board.move_count,
        "over": False,
        "winner": None,
    }
    res = board.result()
    if res is not None:
        d["over"] = True
        d["winner"] = "red" if res == 1 else "black"
    if extra:
        d.update(extra)
    return d


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    # ---------------------------------------------------------- utils
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, name, ctype):
        path = os.path.join(WEB, name)
        if not os.path.exists(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    # ---------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if p == "/play":
            return self._file("play.html", "text/html; charset=utf-8")
        if p == "/api/stats":
            return self._send(200, self.stats())
        if p == "/api/board":
            with _lock:
                return self._send(200, board_json(_game["board"], {
                    "last": _game["last"],
                    "engine_value": _game["engine_value"],
                    "engine_visits": _game["engine_visits"],
                }))
        if p == "/api/games":
            return self._send(200, self.games())
        if p == "/api/checkpoint":
            q = parse_qs(u.query)
            which = q.get("name", ["best"])[0]
            _model_path["p"] = os.path.join(RUN, f"{which}.npz")
            _net_cache["net"] = None
            return self._send(200, {"ok": True, "model": _model_path["p"]})
        return self._send(404, {"error": "no route"})

    # ---------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/new":
            with _lock:
                _game["board"] = Board(START_FEN)
                _game["last"] = None
                _game["engine_value"] = None
                _game["engine_visits"] = None
                return self._send(200, board_json(_game["board"]))
        if u.path == "/api/undo":
            with _lock:
                b = _game["board"]
                for _ in range(2):
                    if b.history:
                        b.unmake()
                _game["last"] = None
                return self._send(200, board_json(b))
        if u.path == "/api/play":
            return self.play(body)
        if u.path == "/api/hint":
            return self.hint(body)
        return self._send(404, {"error": "no route"})

    # ---------------------------------------------------------- 逻辑
    def play(self, body):
        frm, to = int(body["frm"]), int(body["to"])
        sims = int(body.get("sims", 200))
        with _lock:
            b = _game["board"]
            if (frm, to) not in set(b.legal_moves()):
                return self._send(400, {"error": "illegal move"})
            b.make(frm, to)
            _game["last"] = [frm, to]
            _game["engine_visits"] = None
            _game["engine_value"] = None
            if b.result() is not None:
                return self._send(200, board_json(b, {"last": _game["last"]}))
            net = get_net()
            visits, value = search(b, net, sims=sims, rng=np.random.default_rng())
            if not visits:
                return self._send(200, board_json(b, {"last": _game["last"]}))
            mv = max(visits.items(), key=lambda kv: kv[1])[0]
            top = sorted(visits.items(), key=lambda kv: -kv[1])[:5]
            total = sum(visits.values())
            _game["engine_visits"] = [[f, t, n / total] for (f, t), n in top]
            _game["engine_value"] = value
            b.make(*mv)
            _game["last"] = [mv[0], mv[1]]
            return self._send(200, board_json(b, {
                "last": _game["last"],
                "engine_move": [mv[0], mv[1]],
                "engine_value": value,
                "engine_visits": _game["engine_visits"],
            }))

    def hint(self, body):
        sims = int(body.get("sims", 400))
        with _lock:
            b = _game["board"]
            net = get_net()
            visits, value = search(b, net, sims=sims, rng=np.random.default_rng())
            total = max(1, sum(visits.values()))
            top = sorted(visits.items(), key=lambda kv: -kv[1])[:5]
            return self._send(200, {"value": value,
                                    "visits": [[f, t, n / total] for (f, t), n in top]})

    def stats(self):
        path = os.path.join(RUN, "stats.jsonl")
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        status = {}
        sp = os.path.join(RUN, "status.json")
        if os.path.exists(sp):
            with open(sp) as f:
                status = json.load(f)
        alive = False
        pid = status.get("pid")
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        return {"rows": rows[-2000:], "status": status, "alive": alive}

    def games(self):
        d = os.path.join(RUN, "games")
        out = []
        if os.path.isdir(d):
            files = sorted(os.listdir(d))[-12:]
            for name in files:
                try:
                    with open(os.path.join(d, name)) as f:
                        g = json.load(f)
                    g["name"] = name
                    out.append(g)
                except Exception:
                    pass
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"看板:  http://127.0.0.1:{args.port}/")
    print(f"对弈:  http://127.0.0.1:{args.port}/play")
    srv.serve_forever()


if __name__ == "__main__":
    main()
