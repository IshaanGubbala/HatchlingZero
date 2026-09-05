#!/usr/bin/env python3
"""Real-time Hatchling World viewer -- a tiny local HTTP server that
serves a live-updating page rendering whatever a running rollout
(scripts/hz_world_rollout_demo.py today, the real HZ policy once
Phase 2 lands) is currently doing in the world. No external
dependencies beyond the standard library.

How it works: the rollout script writes one JSON snapshot to
--state-file after every environment step (see hz_world_rollout_demo.py
for the exact schema). This server just serves that file's current
content at GET /state, and a single static HTML/JS page that polls
/state every --poll-ms milliseconds and redraws an SVG of the room
graph (agent position, doors, lock colors, inventory) plus a small
recent-reward/success sparkline. Open http://localhost:<port> in a
browser while a rollout is running.
"""
from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from pathlib import Path

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Hatchling World -- live</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f1115; color: #e8e8ec; margin: 0; padding: 24px; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: #8a8f98; font-size: 13px; margin-bottom: 20px; }
  .row { display: flex; gap: 24px; flex-wrap: wrap; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 10px; padding: 16px; }
  .stat { font-size: 12px; color: #8a8f98; text-transform: uppercase; letter-spacing: .04em; }
  .val { font-size: 22px; font-weight: 600; margin-top: 2px; }
  svg { background: #10131a; border-radius: 8px; }
  .legend { font-size: 12px; color: #8a8f98; margin-top: 8px; line-height: 1.6; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
  #log { font-size: 12px; color: #b8bcc4; white-space: pre-wrap; max-height: 160px; overflow-y: auto; }
</style></head>
<body>
  <h1>Hatchling World -- live</h1>
  <div class="sub" id="subtitle">waiting for a rollout to start...</div>
  <div class="row">
    <div class="card"><svg id="world" width="440" height="440" viewBox="0 0 440 440"></svg></div>
    <div class="card" style="min-width:220px">
      <div class="stat">Episode</div><div class="val" id="episode">-</div>
      <div class="stat" style="margin-top:12px">Step</div><div class="val" id="step">-</div>
      <div class="stat" style="margin-top:12px">Last reward</div><div class="val" id="reward">-</div>
      <div class="stat" style="margin-top:12px">Episode return</div><div class="val" id="return">-</div>
      <div class="stat" style="margin-top:12px">Recent success rate</div><div class="val" id="success">-</div>
      <div class="stat" style="margin-top:12px">Agent</div><div class="val" id="agent_type">-</div>
    </div>
    <div class="card" style="min-width:260px">
      <div class="stat">Inventory</div>
      <div id="inventory" style="margin-top:6px"></div>
      <div class="stat" style="margin-top:16px">Recent actions</div>
      <div id="log"></div>
    </div>
  </div>
<script>
const COLORS = ["#ff6b6b", "#4dabf7", "#69db7c", "#ffd43b", "#da77f2", "#ff922b"];
let recentLog = [];

function render(s) {
  document.getElementById('subtitle').textContent =
    s.agent_type + " agent, room graph n=" + s.n_rooms + ", colors=" + s.n_colors;
  document.getElementById('episode').textContent = s.episode;
  document.getElementById('step').textContent = s.step;
  document.getElementById('reward').textContent = s.last_reward.toFixed(3);
  document.getElementById('return').textContent = s.episode_return.toFixed(3);
  document.getElementById('success').textContent = (s.recent_success_rate * 100).toFixed(0) + "%";
  document.getElementById('agent_type').textContent = s.agent_type;

  const inv = document.getElementById('inventory');
  inv.innerHTML = s.inventory.map((c, i) =>
    `<span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span>${c}&nbsp;&nbsp;`
  ).join('');

  if (s.last_action_str) {
    recentLog.unshift(`[${s.step}] ${s.last_action_str} -> r=${s.last_reward.toFixed(2)}`);
    recentLog = recentLog.slice(0, 12);
  }
  document.getElementById('log').textContent = recentLog.join('\\n');

  drawWorld(s);
}

function drawWorld(s) {
  const svg = document.getElementById('world');
  svg.innerHTML = '';
  const R = s.n_rooms;
  const cx = 220, cy = 220, radius = 160;
  const pos = [];
  for (let i = 0; i < R; i++) {
    const a = (2 * Math.PI * i) / R - Math.PI / 2;
    pos.push([cx + radius * Math.cos(a), cy + radius * Math.sin(a)]);
  }
  const ns = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {
    const e = document.createElementNS(ns, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  // doors
  for (const d of s.doors) {
    const [x1, y1] = pos[d.a], [x2, y2] = pos[d.b];
    const color = d.locked ? COLORS[d.color % COLORS.length] : '#3a4150';
    const width = d.locked ? 3 : 2;
    const dash = d.locked ? '6,4' : 'none';
    svg.appendChild(el('line', { x1, y1, x2, y2, stroke: color, 'stroke-width': width, 'stroke-dasharray': dash }));
  }
  // rooms
  for (let i = 0; i < R; i++) {
    const [x, y] = pos[i];
    const isGoal = i === s.goal_room, isAgent = i === s.agent_room;
    svg.appendChild(el('circle', {
      cx: x, cy: y, r: isGoal ? 20 : 16,
      fill: isGoal ? '#2b8a3e' : '#20242e',
      stroke: isAgent ? '#fff' : '#4a5060', 'stroke-width': isAgent ? 3 : 1.5,
    }));
    const label = el('text', { x, y: y + 4, 'text-anchor': 'middle', fill: '#cfd3da', 'font-size': 11 });
    label.textContent = i;
    svg.appendChild(label);
    const keyCount = s.room_keys[i].reduce((a, b) => a + b, 0);
    if (keyCount > 0) {
      svg.appendChild(el('circle', { cx: x + 14, cy: y - 14, r: 6, fill: '#ffd43b' }));
    }
    if (isAgent) {
      svg.appendChild(el('circle', { cx: x, cy: y, r: 7, fill: '#fff' }));
    }
  }
}

async function poll() {
  try {
    const res = await fetch('/state');
    if (res.ok) render(await res.json());
  } catch (e) {}
  setTimeout(poll, __POLL_MS__);
}
poll();
</script>
</body></html>"""


def make_handler(state_file: Path, poll_ms: int):
    page = PAGE.replace("__POLL_MS__", str(poll_ms))

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/state":
                try:
                    body = state_file.read_bytes()
                except FileNotFoundError:
                    body = b'{"error": "no rollout running yet"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(page.encode("utf-8"))

        def log_message(self, format, *args):
            pass  # quiet -- don't spam stdout with every poll

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/hz_world_live_state.json"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll-ms", type=int, default=300)
    args = parser.parse_args()

    handler = make_handler(args.state_file, args.poll_ms)
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"[hz_world_live_view] serving http://localhost:{args.port} "
              f"(reading {args.state_file})", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
