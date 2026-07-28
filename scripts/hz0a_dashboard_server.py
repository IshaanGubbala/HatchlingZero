"""Real-time HZ-0A training dashboard server (stdlib-only).

Ported from the pre-restart archive/src/hz0/dashboard/server.py and
re-pointed at the current restart's actual metrics format:

  * outputs/**/native_metal_memory.jsonl  (per-chunk step/loss/lr/memory,
    written by scripts/hz0a_native_stage_runner.py)
  * outputs/**/native_metal_checkpoint/state.json  (sparse validation_loss,
    only recorded every --validation-interval steps)
  * outputs/<run-name>.log  (the tee'd stdout of the run, by convention)

Serves:
  GET /                 -> HTML dashboard
  GET /api/state?run=X  -> JSON snapshot for run X (default: most-recently-
                           written run, i.e. whichever job is actively training)
  GET /api/runs         -> list of discovered run names

No external dependencies (stdlib http.server only); the browser pulls
Chart.js from a CDN. The old dashboard's field names (train_loss, val_loss,
throughput_tok_s, tokens, wall_time, lr) are preserved here so the existing
templates/index.html front-end works unmodified.
"""
from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
TEMPLATES_DIR = Path(__file__).resolve().parent / "dashboard_templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

_LOG_TAIL_N = 12
_MAX_METRIC_POINTS = 2000


def _discover_runs() -> List[Path]:
    if not OUT_DIR.exists():
        return []
    return [p.parent for p in OUT_DIR.rglob("native_metal_memory.jsonl")]


def _pick_latest_run() -> Optional[Path]:
    runs = _discover_runs()
    if not runs:
        return None
    runs.sort(key=lambda d: (d / "native_metal_memory.jsonl").stat().st_mtime, reverse=True)
    return runs[0]


def _find_run(name: str) -> Optional[Path]:
    for run_dir in _discover_runs():
        if run_dir.name == name or str(run_dir.relative_to(OUT_DIR)) == name:
            return run_dir
    return None


def _read_validation_losses(run_dir: Path) -> Dict[int, float]:
    state_path = run_dir / "native_metal_checkpoint" / "state.json"
    if not state_path.is_file():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for item in payload.get("metrics", []):
        if isinstance(item.get("validation_loss"), (int, float)) and isinstance(item.get("step"), int):
            out[item["step"]] = item["validation_loss"]
    return out


def _read_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    jsonl_path = run_dir / "native_metal_memory.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()[-_MAX_METRIC_POINTS:]
    validation_losses = _read_validation_losses(run_dir)
    records = []
    previous = None
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        throughput = None
        if previous is not None and "wall_time" in row and "wall_time" in previous:
            dt = row["wall_time"] - previous["wall_time"]
            dtok = row["tokens_seen"] - previous["tokens_seen"]
            if dt > 0:
                throughput = dtok / dt
        records.append({
            "step": row.get("step"),
            "train_loss": row.get("loss"),
            "val_loss": validation_losses.get(row.get("step")),
            "throughput_tok_s": throughput,
            "tokens": row.get("tokens_seen"),
            "wall_time": row.get("wall_time"),
            "lr": row.get("lr"),
            "gradient_norm": row.get("gradient_norm"),
            "peak_memory_gb": (row["peak_memory_bytes"] / 1e9) if row.get("peak_memory_bytes") is not None else None,
        })
        previous = row
    return records


def _tail_log(run_dir: Path, n: int = _LOG_TAIL_N) -> List[str]:
    log_path = run_dir.parent / f"{run_dir.name}.log"
    if not log_path.is_file():
        return []
    try:
        return log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
    except Exception:
        return []


def _summary(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {"step": None, "train_loss": None, "val_loss": None, "throughput_tok_s": None,
                "tokens": None, "elapsed_min": None, "lr": None, "best_val_loss": None}
    last = metrics[-1]
    val_losses = [m["val_loss"] for m in metrics if isinstance(m.get("val_loss"), (int, float))]
    throughputs = [m["throughput_tok_s"] for m in metrics[-20:] if isinstance(m.get("throughput_tok_s"), (int, float))]
    return {
        "step": last.get("step"),
        "train_loss": last.get("train_loss"),
        "val_loss": val_losses[-1] if val_losses else None,
        "throughput_tok_s": (sum(throughputs) / len(throughputs)) if throughputs else None,
        "tokens": last.get("tokens"),
        "elapsed_min": (last["wall_time"] / 60.0) if isinstance(last.get("wall_time"), (int, float)) else None,
        "lr": last.get("lr"),
        "best_val_loss": min(val_losses) if val_losses else None,
    }


_SNAPSHOT_CACHE: Dict[str, Dict[str, Any]] = {}
_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOT_TTL_S = 1.5


def snapshot(requested_run: Optional[str]) -> Dict[str, Any]:
    with _SNAPSHOT_LOCK:
        cache_key = requested_run or "__latest__"
        cached = _SNAPSHOT_CACHE.get(cache_key)
        now = time.time()
        if cached is not None and (now - cached["ts"]) < _SNAPSHOT_TTL_S:
            return cached
        run_dir = _find_run(requested_run) if requested_run else _pick_latest_run()
        metrics = _read_metrics(run_dir) if run_dir else []
        payload: Dict[str, Any] = {
            "ts": now,
            "run": run_dir.name if run_dir else None,
            "metrics": metrics,
            "summary": _summary(metrics),
            "log": _tail_log(run_dir) if run_dir else [],
        }
        _SNAPSHOT_CACHE[cache_key] = payload
        return payload


def _list_runs() -> List[str]:
    return sorted({run_dir.name for run_dir in _discover_runs()})


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass

    def _send_bytes(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path.startswith("/api/runs"):
            self._send_bytes(200, json.dumps(_list_runs()).encode("utf-8"), "application/json")
            return
        if path.startswith("/api/state"):
            requested_run = query.get("run", [None])[0]
            body = json.dumps(snapshot(requested_run), default=float).encode("utf-8")
            self._send_bytes(200, body, "application/json")
            return
        if path in ("/", "/index", "/index.html"):
            try:
                self._send_bytes(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send_bytes(404, b"index.html not found", "text/plain; charset=utf-8")
            return
        self._send_bytes(404, b"not found", "text/plain; charset=utf-8")


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8014)
    args = parser.parse_args()
    print(f"[dashboard] runs discovered: {_list_runs()}", flush=True)
    httpd = _ReusableTCPServer(("127.0.0.1", args.port), _Handler)
    print(f"[dashboard] serving on http://127.0.0.1:{args.port}  (root={ROOT})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] shutting down")


if __name__ == "__main__":
    main()
