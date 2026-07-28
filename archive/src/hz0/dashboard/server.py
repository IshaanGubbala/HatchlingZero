"""
Real-time training dashboard server (stdlib-only).

Reads:
  * outputs/phase14-current.log              (trailing log lines for footer)
  * outputs/training/<run>/step_*.metrics.json (JSON metrics per checkpoint)

Serves:
  GET /                                 -> HTML dashboard
  GET /api/state                        -> JSON snapshot (live poll)

No external dependencies (no FastAPI/Flask/uvicorn required). The browser
polls /api/state every 2 s; this is plenty fast for cross-token step
updates and keeps the dep footprint at zero.
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]  # project root
OUT_DIR = ROOT / "outputs"
TRAIN_DIR = OUT_DIR / "training"
LOG_PATH = OUT_DIR / "phase14-current.log"

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

_LOG_TAIL_N = 8


def _read_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for f in sorted(run_dir.glob("step_*.metrics.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def _pick_latest_run() -> Optional[Path]:
    if not TRAIN_DIR.exists():
        return None
    candidates = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def _tail_log(n: int = _LOG_TAIL_N) -> List[str]:
    if not LOG_PATH.exists():
        return []
    try:
        data = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return data[-n:]


def _summary(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {
            "step": None, "train_loss": None, "val_loss": None,
            "throughput_tok_s": None, "tokens": None,
            "elapsed_min": None, "lr": None, "best_val_loss": None,
        }
    last = metrics[-1]
    val_losses = [m["val_loss"] for m in metrics
                  if isinstance(m.get("val_loss"), (int, float))]
    best_val = min(val_losses) if val_losses else None
    elapsed_min = None
    if isinstance(last.get("wall_time"), (int, float)):
        elapsed_min = float(last["wall_time"]) / 60.0
    return {
        "step": last.get("step"),
        "train_loss": last.get("train_loss"),
        "val_loss": last.get("val_loss"),
        "throughput_tok_s": last.get("throughput_tok_s"),
        "tokens": last.get("tokens"),
        "elapsed_min": elapsed_min,
        "lr": last.get("lr"),
        "best_val_loss": best_val,
    }


# 1-second TTL cache protected by a threading lock. ThreadingTCPServer
# is multi-threaded so without the lock two requests arriving near the
# expiry boundary both miss the cache, both rebuild, both write back.
_SNAPSHOT_CACHE: Dict[str, Any] = {"ts": 0.0, "payload": None}
_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOT_TTL_S = 1.0


def snapshot() -> Dict[str, Any]:
    with _SNAPSHOT_LOCK:
        now = time.time()
        cached = _SNAPSHOT_CACHE["payload"]
        if cached is not None and (now - _SNAPSHOT_CACHE["ts"]) < _SNAPSHOT_TTL_S:
            return cached
        run_dir = _pick_latest_run()
        metrics = _read_metrics(run_dir) if run_dir else []
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "run": run_dir.name if run_dir else None,
            "metrics": metrics,
            "summary": _summary(metrics),
            "log": _tail_log(),
        }
        _SNAPSHOT_CACHE["ts"] = payload["ts"]
        _SNAPSHOT_CACHE["payload"] = payload
        return payload


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # silence access log
        pass

    def _send_bytes(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/state"):
            body = json.dumps(snapshot(), default=float).encode("utf-8")
            self._send_bytes(200, body, "application/json")
            return
        if path in ("/", "/index", "/index.html"):
            try:
                body = INDEX_HTML.read_bytes()
                self._send_bytes(200, body, "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send_bytes(404, b"index.html not found",
                                 "text/plain; charset=utf-8")
            return
        self._send_bytes(404, b"not found", "text/plain; charset=utf-8")


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    port = int(os.environ.get("PORT", "8014"))
    _ = snapshot()  # warm filesystem caches, fail loud if paths are bad
    httpd = _ReusableTCPServer(("127.0.0.1", port), _Handler)
    print(f"[dashboard] serving on http://127.0.0.1:{port}  "
          f"(root={ROOT})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] shutting down")


if __name__ == "__main__":
    main()
