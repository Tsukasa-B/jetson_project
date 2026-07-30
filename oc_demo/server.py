"""HTTPサーバ。Python標準ライブラリのみ（pipインストール不要）。

テレメトリ配信は WebSocket ではなく SSE (Server-Sent Events)。
ブラウザ標準の EventSource で受けられ、サーバ側は http.server だけで書ける。
配信は片方向で足りる（操作は POST /api/start などの通常リクエスト）ため。

起動:
    python3 -m oc_demo.server --midi-dir midi --models-dir models --mock
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from . import adapter
from .policy import discover_models
from .runner import RunConfig, Runner, build_run_target
from .score import Score, scan_midi_dir

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


class Settings:
    midi_dir: Path = Path("midi")
    models_dir: Path = Path("models")
    port_name: str = "/dev/ttyUSB0"
    use_hardware: bool = True
    default_model: Optional[str] = None
    drive: bool = True
    bpm_min: float = 60.0
    bpm_max: float = 180.0


SETTINGS = Settings()
SONGS: dict = {}
MODELS: list = []
LABELS: dict = {}
RUNNER: Optional[Runner] = None
_START_LOCK = threading.Lock()


# ---------------------------------------------------------------------
def reload_assets() -> None:
    """MIDI・モデル・表示名を読み直す。

    表示名は <midi_dir>/labels.json（{"01_yonuchi": "四分打ち"} 形式）で上書きできる。
    """
    global SONGS, MODELS, LABELS
    SONGS = {s.name: s for s in scan_midi_dir(SETTINGS.midi_dir)}
    MODELS = discover_models(SETTINGS.models_dir)
    LABELS = {}
    lp = Path(SETTINGS.midi_dir) / "labels.json"
    if lp.exists():
        try:
            LABELS = {str(k): str(v) for k, v in
                      json.loads(lp.read_text(encoding="utf-8")).items()}
        except Exception as exc:  # noqa: BLE001
            print(f"[oc_demo] labels.json を読めませんでした: {exc}")


def song_payload(s: Score, bpm: float, lead_in: float = 1.5) -> dict:
    """譜面 + 目標力軌道。どちらも lead_in ぶんずらした「演奏時刻」で返す。"""
    played = s.rescaled(bpm)
    target, _ = build_run_target(s, bpm, lead_in, 1.0)
    d = played.to_dict()
    for n in d["notes"]:
        n["t"] = round(n["t"] + lead_in, 4)
    d["label"] = LABELS.get(s.name, s.name)
    d["lead_in"] = lead_in
    d["dt"] = adapter.CONTROL_DT
    d["target"] = [round(float(v), 3) for v in target]
    return d


def do_start(body: dict) -> tuple[int, dict]:
    global RUNNER
    with _START_LOCK:
        if RUNNER is not None and RUNNER.state in ("running", "arming"):
            return 409, {"error": "already running"}
        name = body.get("song")
        s = SONGS.get(name)
        if s is None:
            return 404, {"error": f"unknown song: {name}"}

        bpm = float(body.get("bpm") or s.nominal_bpm)
        bpm = max(SETTINGS.bpm_min, min(SETTINGS.bpm_max, bpm))
        lead_in = float(body.get("lead_in", 1.5))

        model_path = None
        model_id = body.get("model") or SETTINGS.default_model
        if model_id:
            hit = next((m for m in MODELS
                        if m["id"] == model_id or m["label"] == model_id), None)
            if hit is None:
                return 404, {"error": f"unknown model: {model_id}"}
            model_path = hit["path"]

        RUNNER = Runner(RunConfig(
            score=s, bpm=bpm, model_path=model_path, port=SETTINGS.port_name,
            use_hardware=SETTINGS.use_hardware, lead_in=lead_in,
            drive=SETTINGS.drive,
        ))
        RUNNER.start()
        return 200, {"ok": True, "bpm": bpm, "model": model_id,
                     "score": song_payload(s, bpm, lead_in)}


# ---------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "oc_demo"

    def log_message(self, fmt, *args) -> None:  # noqa: A003
        pass  # 20Hz配信でログが溢れるので黙らせる

    # -- 返信ヘルパ ---------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: Path, ctype: str) -> None:
        try:
            self._send(200, path.read_bytes(), ctype)
        except FileNotFoundError:
            self._json({"error": "not found"}, 404)

    # -- GET ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        route = u.path
        q = parse_qs(u.query)

        if route in ("/", "/index.html"):
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")

        if route.startswith("/static/"):
            name = Path(route[len("/static/"):]).name  # ディレクトリ抜けを防ぐ
            ctype = ("text/html; charset=utf-8" if name.endswith(".html")
                     else "text/css" if name.endswith(".css")
                     else "application/javascript" if name.endswith(".js")
                     else "application/octet-stream")
            return self._file(STATIC / name, ctype)

        if route == "/api/health":
            return self._json({
                "ok": True,
                "hardware": SETTINGS.use_hardware,
                "drive": SETTINGS.drive,
                "port": SETTINGS.port_name,
                "n_songs": len(SONGS),
                "n_models": len(MODELS),
                "adapter": dict(adapter.STATUS),
                "bpm_range": [SETTINGS.bpm_min, SETTINGS.bpm_max],
                "state": RUNNER.state if RUNNER else "idle",
            })

        if route == "/api/songs":
            return self._json([
                {"name": s.name, "label": LABELS.get(s.name, s.name),
                 "nominal_bpm": round(s.nominal_bpm, 1),
                 "duration": round(s.duration, 2),
                 "n_notes": len(s.notes), "lanes": s.lanes}
                for s in SONGS.values()
            ])

        if route == "/api/models":
            return self._json(MODELS)

        if route.startswith("/api/song/"):
            name = unquote(route[len("/api/song/"):])
            s = SONGS.get(name)
            if s is None:
                return self._json({"error": f"unknown song: {name}"}, 404)
            bpm = float(q.get("bpm", [s.nominal_bpm])[0])
            lead_in = float(q.get("lead_in", [1.5])[0])
            return self._json(song_payload(s, bpm, lead_in))

        if route == "/events":
            return self._sse()

        self._json({"error": "not found"}, 404)

    # -- POST ---------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if route == "/api/start":
            code, payload = do_start(body)
            return self._json(payload, code)

        if route == "/api/stop":
            if RUNNER is not None:
                RUNNER.stop()
            return self._json({"ok": True, "state": RUNNER.state if RUNNER else "idle"})

        if route == "/api/reload":
            reload_assets()
            return self._json({"ok": True, "n_songs": len(SONGS)})

        self._json({"error": "not found"}, 404)

    # -- SSE ----------------------------------------------------------
    def _sse(self) -> None:
        """20Hzでテレメトリを流し続ける。ブラウザが切れても演奏は止めない。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        cursor = 0
        seen_runner = None
        seen_run = None
        try:
            while True:
                if RUNNER is None:
                    payload = {"type": "idle"}
                else:
                    if seen_runner is not RUNNER or seen_run != RUNNER.run_id:
                        cursor, seen_runner, seen_run = 0, RUNNER, RUNNER.run_id
                    samples, hits, cursor = RUNNER.drain(cursor)
                    payload = {"type": "tick", "s": samples, "hits": hits,
                               "run_id": RUNNER.run_id, **RUNNER.snapshot()}
                line = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return  # ブラウザを閉じただけ。演奏は続ける


# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PAM robot OC demo server")
    ap.add_argument("--midi-dir", default=os.environ.get("OC_MIDI_DIR", "midi"))
    ap.add_argument("--models-dir", default=os.environ.get("OC_MODELS_DIR", "models"))
    ap.add_argument("--port-name", default=os.environ.get("OC_SERIAL_PORT", "/dev/ttyUSB0"))
    ap.add_argument("--model", default=os.environ.get("OC_DEFAULT_MODEL"),
                    help="既定モデル（例 IROS/modelB.onnx）")
    ap.add_argument("--mock", action="store_true", help="実機に繋がずシミュレーションで動かす")
    ap.add_argument("--no-drive", action="store_true",
                    help="実シリアルに繋ぐが圧力0だけを送る（ロボットを動かさない検証モード）")
    ap.add_argument("--bpm-min", type=float, default=60.0)
    ap.add_argument("--bpm-max", type=float, default=180.0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http-port", type=int, default=8080)
    args = ap.parse_args()

    SETTINGS.midi_dir = Path(args.midi_dir)
    SETTINGS.models_dir = Path(args.models_dir)
    SETTINGS.port_name = args.port_name
    SETTINGS.use_hardware = not args.mock
    SETTINGS.default_model = args.model
    SETTINGS.drive = not args.no_drive
    SETTINGS.bpm_min = args.bpm_min
    SETTINGS.bpm_max = args.bpm_max

    reload_assets()
    print(f"[oc_demo] songs={len(SONGS)} models={len(MODELS)} "
          f"mode={'MOCK' if args.mock else 'HARDWARE ' + args.port_name}"
          f"{'  [駆動なし検証モード]' if args.no_drive else ''}")
    print(f"[oc_demo] ブラウザで http://localhost:{args.http_port}/ を開いてください")
    if SETTINGS.use_hardware and SETTINGS.default_model is None:
        print("[oc_demo] 警告: --model が未指定です。スクリプト動作（学習済みでない）になります")

    httpd = ThreadingHTTPServer((args.host, args.http_port), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[oc_demo] 終了します")
    finally:
        if RUNNER is not None:
            RUNNER.stop()
            RUNNER.join(timeout=2.0)
        httpd.server_close()


if __name__ == "__main__":
    main()
