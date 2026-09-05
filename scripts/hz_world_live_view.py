#!/usr/bin/env python3
"""Real-time Hatchling World viewer -- a tiny local HTTP server that
serves a live-updating page rendering whatever a running rollout
(scripts/hz_world_rollout_demo.py today, the real HZ policy once
Phase 2 lands) is currently doing in the world. No external
dependencies beyond the standard library (the page itself pulls one
Google Font over the network, same as any normal webpage would).

How it works: the rollout script writes one JSON snapshot to
--state-file after every environment step (see hz_world_rollout_demo.py
for the exact schema). This server just serves that file's current
content at GET /state, and a single static HTML/JS page that polls
/state every --poll-ms milliseconds, animates the agent smoothly
between rooms, flashes doors when they unlock, and keeps a running
success/return history chart. Open http://localhost:<port> in a
browser while a rollout is running.
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hatchling World -- live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #08090d;
    --bg-grid: #0d1017;
    --panel: #10141c;
    --panel-2: #141a24;
    --border: #1e2530;
    --text: #dde3ee;
    --text-dim: #6b7690;
    --dim: #707893;
    --accent: #2fe3c6;
    --accent-glow: rgba(47,227,198,0.45);
    --amber: #ffb84d;
    --danger: #ff6b6b;
    --success: #45e08f;
    --lock-colors: #ff6b6b,#4d9dff,#ffd166,#c77dff,#45e08f,#ff9f4d;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    background:
      radial-gradient(circle at 15% -10%, rgba(47,227,198,0.06), transparent 40%),
      radial-gradient(circle at 100% 10%, rgba(255,184,77,0.05), transparent 35%),
      var(--bg);
    background-attachment: fixed;
    color: var(--text);
    margin: 0;
    padding: 28px 32px 48px;
    min-height: 100vh;
  }
  .topbar { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 22px; }
  .brand { display: flex; align-items: center; gap: 14px; }
  h1 {
    font-size: 22px; font-weight: 800; letter-spacing: 0.14em; margin: 0;
    text-transform: uppercase; color: var(--text);
    text-shadow: 0 0 18px var(--accent-glow);
  }
  h1 span { color: var(--accent); }
  .live-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 12px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--panel); font-size: 11px; letter-spacing: 0.08em; color: var(--dim);
    text-transform: uppercase;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); transition: background .3s; }
  .dot.on { background: var(--success); box-shadow: 0 0 10px var(--success); animation: pulse 1.4s ease-in-out infinite; }
  .dot.stale { background: var(--amber); box-shadow: 0 0 8px var(--amber); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
  .subtitle { color: var(--dim); font-size: 12.5px; margin-top: 6px; letter-spacing: .02em; }
  .subtitle b { color: var(--text); font-weight: 600; }

  .layout { display: grid; grid-template-columns: minmax(420px, 620px) minmax(300px, 380px); gap: 20px; align-items: start; }
  @media (max-width: 1020px) { .layout { grid-template-columns: 1fr; } }

  .card {
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--border); border-radius: 14px; padding: 18px;
    box-shadow: 0 12px 30px -18px rgba(0,0,0,0.7);
  }
  .card + .card { margin-top: 18px; }
  .card-title {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--dim);
    margin: 0 0 12px; display: flex; justify-content: space-between; align-items: center;
  }

  svg#world { display: block; width: 100%; height: auto; }
  .room-label { font-size: 12px; fill: #cfd6e4; font-weight: 600; }
  .goal-badge { font-size: 9px; fill: #08130d; font-weight: 800; letter-spacing: .06em; }

  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; font-size: 11px; color: var(--dim); }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .swatch.line { width: 18px; height: 2px; border-radius: 0; }

  .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
  .stat { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
  .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: var(--dim); }
  .stat-val { font-size: 21px; font-weight: 700; margin-top: 3px; color: var(--text); }
  .stat-val.accent { color: var(--accent); }
  .stat-val.good { color: var(--success); }
  .stat-val.bad { color: var(--danger); }

  .inv-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
  .key-chip {
    display: flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 8px;
    background: rgba(255,255,255,0.03); border: 1px solid var(--border); font-size: 12px;
    transition: transform .2s;
  }
  .key-chip.zero { opacity: .3; }
  .key-icon { width: 10px; height: 10px; border-radius: 3px 3px 3px 0; transform: rotate(-8deg); }

  .chart-wrap { position: relative; }
  #chart { width: 100%; height: 64px; display: block; }
  .ticks { display: flex; gap: 3px; margin-top: 8px; }
  .tick { flex: 1; height: 14px; border-radius: 2px; background: var(--border); }
  .tick.win { background: var(--success); }
  .tick.loss { background: var(--danger); }

  #log { display: flex; flex-direction: column-reverse; gap: 6px; max-height: 260px; overflow-y: auto; padding-right: 4px; }
  #log::-webkit-scrollbar { width: 6px; }
  #log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .log-entry {
    display: flex; gap: 8px; align-items: baseline; font-size: 12px; padding: 6px 8px;
    border-radius: 8px; background: rgba(255,255,255,0.015); border-left: 2px solid var(--border);
    animation: fadein .25s ease-out;
  }
  .log-entry.move { border-left-color: #4d9dff; }
  .log-entry.use_key { border-left-color: var(--amber); }
  .log-entry.pickup { border-left-color: var(--success); }
  .log-entry.inspect { border-left-color: var(--dim); }
  .log-icon { width: 16px; text-align: center; }
  .log-text { color: var(--text); flex: 1; }
  .log-reward { color: var(--dim); font-size: 11px; }
  @keyframes fadein { from { opacity: 0; transform: translateY(-3px); } to { opacity: 1; transform: translateY(0); } }

  .badge { font-size: 10px; padding: 2px 8px; border-radius: 999px; background: rgba(47,227,198,0.12); color: var(--accent); border: 1px solid rgba(47,227,198,0.3); }

  /* -- Language Nursery view -- */
  .stage-track { display: flex; gap: 8px; margin-bottom: 4px; }
  .stage-pip {
    flex: 1; text-align: center; padding: 7px 4px; border-radius: 8px; font-size: 11px; font-weight: 700;
    letter-spacing: .04em; background: rgba(255,255,255,0.02); border: 1px solid var(--border); color: var(--dim);
  }
  .stage-pip.done { color: var(--success); border-color: rgba(69,224,143,0.35); background: rgba(69,224,143,0.06); }
  .stage-pip.active {
    color: var(--bg); background: var(--accent); border-color: var(--accent);
    box-shadow: 0 0 16px var(--accent-glow);
  }
  .instruction-banner {
    font-size: 19px; font-weight: 700; color: var(--text); padding: 14px 16px; border-radius: 10px;
    background: rgba(47,227,198,0.06); border: 1px solid rgba(47,227,198,0.22); margin: 14px 0 2px;
    text-shadow: 0 0 14px var(--accent-glow);
  }
  .obj-grid {
    display: flex; flex-wrap: wrap; gap: 22px; align-items: flex-end; justify-content: flex-start;
    padding: 28px 18px 22px; margin-top: 10px; border-radius: 12px; min-height: 120px;
    background:
      linear-gradient(180deg, transparent 0%, transparent 62%, rgba(47,227,198,0.05) 62%, rgba(47,227,198,0.09) 100%),
      radial-gradient(ellipse at 30% 15%, rgba(255,184,77,0.05), transparent 55%),
      var(--panel);
    border: 1px solid var(--border);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), inset 0 -18px 30px -20px rgba(47,227,198,0.12);
    position: relative;
    overflow: hidden;
  }
  .obj-grid::before {
    content: ''; position: absolute; left: 0; right: 0; bottom: 38%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(47,227,198,0.25), transparent);
  }
  .nursery-obj { display: flex; flex-direction: column; align-items: center; gap: 6px; position: relative; }
  .obj-shape { transition: box-shadow .25s, transform .25s; box-shadow: 0 6px 16px -6px rgba(0,0,0,0.6); }
  .obj-shape.ring-target { box-shadow: 0 0 0 3px var(--amber), 0 0 18px rgba(255,184,77,0.55); }
  .obj-shape.ring-pred { box-shadow: 0 0 0 3px var(--accent), 0 0 16px var(--accent-glow); outline: 3px dashed rgba(47,227,198,0.5); outline-offset: 3px; }
  .obj-shape.ring-correct { box-shadow: 0 0 0 3px var(--success), 0 0 20px rgba(69,224,143,0.6); }
  .obj-shape.ring-match { outline: 2px dashed var(--accent); outline-offset: 3px; }
  .obj-label { font-size: 10px; color: var(--dim); text-transform: uppercase; letter-spacing: .03em; text-align: center; max-width: 92px; }
  .obj-tag {
    font-size: 9px; font-weight: 800; letter-spacing: .06em; padding: 2px 7px; border-radius: 999px;
  }
  .tag-target { background: rgba(255,184,77,0.16); color: var(--amber); border: 1px solid rgba(255,184,77,0.4); }
  .tag-pred { background: rgba(47,227,198,0.16); color: var(--accent); border: 1px solid rgba(47,227,198,0.4); }

  .token-row { display: flex; flex-wrap: wrap; gap: 6px; padding: 16px 4px 4px; }
  .token-chip {
    font-size: 13px; padding: 4px 9px; border-radius: 6px; background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
  }
  .token-chip.ok { color: var(--success); border-color: rgba(69,224,143,0.35); }
  .token-chip.bad { color: var(--danger); border-color: rgba(255,107,107,0.35); }

  .consequence-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
  .cons-chip { padding: 6px 10px; border-radius: 8px; font-size: 11px; border: 1px solid var(--border); background: rgba(255,255,255,0.02); }
  .cons-chip.match { border-color: rgba(69,224,143,0.4); color: var(--success); }
  .cons-chip.mismatch { border-color: rgba(255,107,107,0.4); color: var(--danger); }

  .verify-banner {
    display: flex; align-items: center; gap: 12px; margin-top: 14px; padding: 10px 14px;
    border-radius: 10px; border: 1px solid var(--border); background: rgba(255,255,255,0.02); font-size: 13px;
  }
  .verify-banner.match { border-color: rgba(69,224,143,0.4); }
  .verify-banner.mismatch { border-color: rgba(255,107,107,0.4); }
  .verify-verdict { font-weight: 800; letter-spacing: .04em; padding: 3px 10px; border-radius: 999px; font-size: 11px; }
  .verify-verdict.yes { background: rgba(69,224,143,0.14); color: var(--success); }
  .verify-verdict.no { background: rgba(255,107,107,0.14); color: var(--danger); }

  .passage-list { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
  .passage-line {
    font-size: 12.5px; padding: 7px 11px; border-radius: 8px; color: var(--dim);
    background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-left: 2px solid var(--border);
  }
  .passage-line .idx { color: var(--text-dim); margin-right: 8px; font-weight: 700; }

  #nursery-chart { width: 100%; height: 90px; display: block; }

  #arch-diagram { width: 100%; height: auto; display: block; }
  .arch-note { font-size: 10.5px; color: var(--dim); margin-top: 8px; letter-spacing: .01em; }
  .arch-label { font-size: 9px; fill: var(--dim); letter-spacing: .06em; text-transform: uppercase; }
  .arch-val { font-size: 9px; fill: var(--accent); font-weight: 700; }
  .arch-flow { stroke: var(--border); stroke-width: 1.5; fill: none; }
  .arch-pulse { stroke: var(--accent); stroke-width: 2; fill: none; filter: drop-shadow(0 0 4px var(--accent-glow)); }
  #arch-gate-badge { font-variant-numeric: tabular-nums; }
  .nursery-legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 11px; color: var(--dim); }
</style></head>
<body>
  <div class="topbar">
    <div>
      <div class="brand">
        <h1>HATCHLING <span>WORLD</span></h1>
        <span class="live-pill"><span class="dot" id="dot"></span><span id="live-text">connecting</span></span>
      </div>
      <div class="subtitle">
        Agent: <b id="agent_type">-</b> &nbsp;|&nbsp; School level: <b id="school_level">-</b>
        &nbsp;|&nbsp; Episode <b id="episode">-</b> &nbsp;/&nbsp; step <b id="step">-</b> of <b id="plan_len">-</b>
      </div>
    </div>
  </div>

  <div class="layout" id="room-view">
    <div>
      <div class="card">
        <div class="card-title">
          <span>WORLD MAP</span>
          <span class="badge" id="room-count-badge">-</span>
        </div>
        <svg id="world" viewBox="0 0 520 480">
          <defs>
            <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="6" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
        </svg>
        <div class="legend" id="legend"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="card-title">STATUS</div>
        <div class="stat-grid">
          <div class="stat"><div class="stat-label">Last reward</div><div class="stat-val" id="reward">-</div></div>
          <div class="stat"><div class="stat-label">Episode return</div><div class="stat-val accent" id="return">-</div></div>
          <div class="stat"><div class="stat-label">Success rate (recent)</div><div class="stat-val good" id="success">-</div></div>
          <div class="stat"><div class="stat-label">Last action</div><div class="stat-val" style="font-size:13px" id="last_action_type">-</div></div>
        </div>
        <div class="card-title" style="margin-top:16px">INVENTORY</div>
        <div class="inv-row" id="inventory"></div>
      </div>

      <div class="card">
        <div class="card-title">RECENT EPISODES <span id="chart-n" class="badge">n=0</span></div>
        <div class="chart-wrap"><svg id="chart" viewBox="0 0 300 64" preserveAspectRatio="none"></svg></div>
        <div class="ticks" id="ticks"></div>
      </div>

      <div class="card">
        <div class="card-title">ACTION LOG</div>
        <div id="log"></div>
      </div>
    </div>
  </div>

  <div class="layout" id="nursery-view" style="display:none">
    <div>
      <div class="card">
        <div class="card-title">
          <span>LANGUAGE NURSERY</span>
          <span class="badge" id="nursery-stage-badge">-</span>
        </div>
        <div class="stage-track" id="stage-track"></div>
        <div class="passage-list" id="nursery-passage"></div>
        <div class="instruction-banner" id="nursery-instruction">-</div>
        <div class="obj-grid" id="nursery-objects"></div>
        <div class="token-row" id="nursery-tokens"></div>
        <div class="consequence-row" id="nursery-consequence"></div>
        <div class="verify-banner" id="nursery-verify" style="display:none"></div>
        <div class="verify-banner" id="nursery-recall" style="display:none"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="card-title">PROGRESS <span class="badge" id="nursery-step-badge">-</span></div>
        <div class="stat-grid" id="nursery-stats"></div>
      </div>

      <div class="card">
        <div class="card-title">HELD-OUT METRICS</div>
        <svg id="nursery-chart" viewBox="0 0 300 90" preserveAspectRatio="none"></svg>
        <div class="nursery-legend" id="nursery-chart-legend"></div>
      </div>

      <div class="card" id="arch-card">
        <div class="card-title">
          <span>NEURAL ARCHITECTURE &mdash; LIVE</span>
          <span class="badge" id="arch-gate-badge">-</span>
        </div>
        <svg id="arch-diagram" viewBox="0 0 320 230"></svg>
        <div class="arch-note" id="arch-note">real per-slot activity from this step's actual S/H tensors</div>
      </div>
    </div>
  </div>

<script>
const ACTUAL_LOCK_COLORS = ["#ff6b6b","#4d9dff","#ffd166","#c77dff","#45e08f","#ff9f4d"];
const ICONS = { move: "&#8594;", use_key: "&#128273;", pickup: "&#9733;", inspect: "&#128065;" };

let agentPos = null;       // {x,y} currently rendered
let agentTarget = null;    // {x,y} to animate toward
let animStart = 0, animFrom = null, animTo = null;
let lastStepKey = null;
let lastFetchOk = 0;

function lerp(a,b,t){ return a+(b-a)*t; }

function roomPositions(R) {
  const cx=260, cy=230, radius=190, pos=[];
  for (let i=0;i<R;i++){
    const a = (2*Math.PI*i)/R - Math.PI/2;
    pos.push([cx+radius*Math.cos(a), cy+radius*Math.sin(a)]);
  }
  return pos;
}

function el(tag, attrs, ns=true) {
  const e = ns ? document.createElementNS('http://www.w3.org/2000/svg', tag) : document.createElement(tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function renderLegend(nColors) {
  const legend = document.getElementById('legend');
  let html = '';
  for (let c=0;c<nColors;c++){
    html += `<div class="legend-item"><span class="swatch" style="background:${ACTUAL_LOCK_COLORS[c%6]}"></span>Key ${String.fromCharCode(65+c)}</div>`;
  }
  html += `<div class="legend-item"><span class="swatch line" style="background:#3a4150;border-top:2px dashed #3a4150"></span>open passage</div>`;
  html += `<div class="legend-item"><span class="swatch line" style="background:transparent;border-top:2px dashed #ff6b6b"></span>locked (needs key)</div>`;
  html += `<div class="legend-item">&#9679;&nbsp;agent&nbsp;&nbsp;&#9678;&nbsp;goal</div>`;
  legend.innerHTML = html;
}

function drawWorld(s) {
  const svg = document.getElementById('world');
  svg.innerHTML = '<defs><filter id="glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>';
  const R = s.n_rooms;
  const pos = roomPositions(R);

  for (const d of s.doors) {
    const [x1,y1] = pos[d.a], [x2,y2] = pos[d.b];
    const justUnlocked = window.__lastDoors && window.__lastDoors.some(pd => pd.a===d.a && pd.b===d.b && pd.locked && !d.locked);
    const color = d.locked ? ACTUAL_LOCK_COLORS[d.color % 6] : '#333c4a';
    const line = el('line', {x1,y1,x2,y2, stroke: color, 'stroke-width': d.locked?3:2, 'stroke-dasharray': d.locked?'7,5':'none', opacity: justUnlocked?0:1});
    if (justUnlocked) {
      line.style.transition = 'opacity 1.2s ease-out .1s';
      requestAnimationFrame(()=>requestAnimationFrame(()=>line.setAttribute('opacity','0.15')));
    }
    svg.appendChild(line);
    if (d.locked) {
      const mx=(x1+x2)/2, my=(y1+y2)/2;
      const lockSvg = el('text', {x:mx,y:my+4,'text-anchor':'middle','font-size':11,fill:color});
      lockSvg.textContent = '\u{1F512}';
      svg.appendChild(lockSvg);
    }
  }
  window.__lastDoors = s.doors;

  for (let i=0;i<R;i++){
    const [x,y] = pos[i];
    const isGoal = i===s.goal_room;
    const g = el('g', {});
    if (isGoal) {
      const pulse = el('circle', {cx:x,cy:y,r:26,fill:'none',stroke:'var(--success)','stroke-width':2,opacity:0.5});
      pulse.innerHTML = '<animate attributeName="r" values="22;30;22" dur="2s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.6;0.05;0.6" dur="2s" repeatCount="indefinite"/>';
      g.appendChild(pulse);
    }
    g.appendChild(el('circle', {cx:x,cy:y,r:19, fill: isGoal?'#123322':'#161c27', stroke: isGoal?'#45e08f':'#2a3140', 'stroke-width':1.5}));
    const label = el('text', {x,y:y+4,'text-anchor':'middle', class:'room-label'});
    label.textContent = i;
    g.appendChild(label);
    const keyCount = s.room_keys[i].reduce((a,b)=>a+b,0);
    if (keyCount>0) {
      const kx=x+15, ky=y-15;
      g.appendChild(el('circle',{cx:kx,cy:ky,r:7,fill:'#ffd166',filter:'url(#glow)'}));
      const kt = el('text', {x:kx,y:ky+3,'text-anchor':'middle','font-size':9,fill:'#3a2c00','font-weight':700});
      kt.textContent = keyCount;
      g.appendChild(kt);
    }
    svg.appendChild(g);
  }

  // agent -- smooth animated marker, drawn last so it's on top
  const targetXY = pos[s.agent_room];
  const prevXY = pos[s.prev_room !== undefined ? s.prev_room : s.agent_room];
  const stepKey = s.episode + ':' + s.step;
  if (stepKey !== lastStepKey) {
    animFrom = agentPos || prevXY;
    animTo = targetXY;
    animStart = performance.now();
    lastStepKey = stepKey;
  }
  if (!agentPos) agentPos = targetXY;

  const halo = el('circle', {cx:agentPos[0], cy:agentPos[1], r:14, fill:'var(--accent)', opacity:0.25, filter:'url(#glow)', id:'agent-halo'});
  const dot = el('circle', {cx:agentPos[0], cy:agentPos[1], r:8, fill:'#eafffb', stroke:'var(--accent)', 'stroke-width':2, id:'agent-dot'});
  svg.appendChild(halo);
  svg.appendChild(dot);
}

function animateAgent() {
  if (animFrom && animTo) {
    const t = Math.min(1, (performance.now()-animStart)/380);
    const ease = 1 - Math.pow(1-t, 3);
    agentPos = [lerp(animFrom[0],animTo[0],ease), lerp(animFrom[1],animTo[1],ease)];
    const dot = document.getElementById('agent-dot');
    const halo = document.getElementById('agent-halo');
    if (dot && halo) {
      dot.setAttribute('cx', agentPos[0]); dot.setAttribute('cy', agentPos[1]);
      halo.setAttribute('cx', agentPos[0]); halo.setAttribute('cy', agentPos[1]);
    }
    if (t >= 1) { animFrom = null; }
  }
  requestAnimationFrame(animateAgent);
}
requestAnimationFrame(animateAgent);

function drawChart(returns) {
  const svg = document.getElementById('chart');
  svg.innerHTML = '';
  if (!returns.length) return;
  const min = Math.min(...returns, 0), max = Math.max(...returns, 1);
  const range = (max-min) || 1;
  const w = 300, h = 64, n = returns.length;
  const pts = returns.map((r,i) => {
    const x = n===1 ? w : (i/(n-1))*w;
    const y = h - ((r-min)/range)*h;
    return [x,y];
  });
  const path = 'M ' + pts.map(p=>p.join(',')).join(' L ');
  const area = path + ` L ${w},${h} L 0,${h} Z`;
  svg.appendChild(el('path', {d:area, fill:'rgba(47,227,198,0.12)', stroke:'none'}));
  svg.appendChild(el('path', {d:path, fill:'none', stroke:'var(--accent)', 'stroke-width':2}));
}

const NURSERY_COLOR_HEX = { red: '#ff6b6b', blue: '#4d9dff', green: '#45e08f', yellow: '#ffd166' };
const NURSERY_SHAPE_RADIUS = { ball: '50%', box: '8px', block: '8px', object: '8px' };
const NURSERY_PALETTE = ['#2fe3c6', '#ffb84d', '#ff6b6b', '#c77dff', '#45e08f'];
const NURSERY_STAGES = ['L0', 'L1', 'L2', 'L3', 'L4-logic', 'L4-count',
                         'L5', 'L5-stress', 'L6', 'Sch-arith', 'Sch-rule'];

function renderStageTrack(s) {
  const track = document.getElementById('stage-track');
  track.innerHTML = NURSERY_STAGES.map((name, i) => {
    let cls = '';
    if (i < s.stage_idx) cls = 'done';
    else if (i === s.stage_idx) cls = 'active';
    return `<div class="stage-pip ${cls}">${name}</div>`;
  }).join('');
}

function renderNurseryObjects(s) {
  const wrap = document.getElementById('nursery-objects');
  wrap.innerHTML = '';
  const matching = s.matching_indices || [];
  (s.objects || []).forEach((o, i) => {
    const isTarget = i === s.target_idx;
    const isPred = i === s.pred_idx;
    let ringClass = '';
    if (isTarget && isPred) ringClass = 'ring-correct';
    else if (isTarget) ringClass = 'ring-target';
    else if (isPred) ringClass = 'ring-pred';
    else if (matching.includes(i)) ringClass = 'ring-match';
    const size = o.size === 'large' ? 74 : 46;
    const rotate = o.type === 'object' ? 'transform:rotate(45deg);' : '';
    const div = document.createElement('div');
    div.className = 'nursery-obj';
    let tags = '';
    if (isTarget) tags += '<div class="obj-tag tag-target">TARGET</div>';
    if (isPred) tags += `<div class="obj-tag tag-pred">${isTarget ? 'MODEL &#10003;' : 'MODEL'}</div>`;
    const stateBits = [o.held ? 'held' : '', o.opened ? 'open' : ''].filter(Boolean).join(', ');
    div.innerHTML =
      `<div class="obj-shape ${ringClass}" style="width:${size}px;height:${size}px;background:${NURSERY_COLOR_HEX[o.color] || '#888'};border-radius:${NURSERY_SHAPE_RADIUS[o.type] || '8px'};${rotate}"></div>` +
      `<div class="obj-label">${o.size} ${o.color} ${o.type}${stateBits ? ' &middot; ' + stateBits : ''}</div>` +
      tags;
    wrap.appendChild(div);
  });
}

function renderNurseryTokens(s) {
  const row = document.getElementById('nursery-tokens');
  if (!s.tokens) { row.innerHTML = ''; return; }
  row.innerHTML = s.tokens.map(t => `<span class="token-chip ${t.correct ? 'ok' : 'bad'}">${t.word}</span>`).join('');
}

function renderNurseryConsequence(s) {
  const row = document.getElementById('nursery-consequence');
  if (!s.consequence_true || !s.consequence_pred) { row.innerHTML = ''; return; }
  const keys = [['position_right', 'position: right?'], ['held', 'held?'], ['opened', 'opened?']];
  row.innerHTML = `<div class="cons-chip">verb: <b style="color:var(--text)">${s.verb}</b></div>` +
    keys.map(([k, label]) => {
      const match = s.consequence_true[k] === s.consequence_pred[k];
      return `<div class="cons-chip ${match ? 'match' : 'mismatch'}">${label} true=${s.consequence_true[k]} pred=${s.consequence_pred[k]}</div>`;
    }).join('');
}

function renderNurseryVerification(s) {
  const banner = document.getElementById('nursery-verify');
  if (s.verify_true === null || s.verify_true === undefined) { banner.style.display = 'none'; return; }
  const match = s.verify_true === s.verify_pred;
  banner.style.display = 'flex';
  banner.className = 'verify-banner ' + (match ? 'match' : 'mismatch');
  const chip = (label, val) => `<span class="verify-verdict ${val ? 'yes' : 'no'}">${label}: ${val ? 'YES' : 'NO'}</span>`;
  banner.innerHTML = `<span style="color:var(--dim)">verify statement:</span> ${chip('actual', s.verify_true)} ${chip('model', s.verify_pred)}` +
    `<span style="margin-left:auto;color:${match ? 'var(--success)' : 'var(--danger)'}">${match ? '&#10003; match' : '&#10007; mismatch'}</span>`;
}

function renderNurseryPassage(s) {
  const wrap = document.getElementById('nursery-passage');
  if (!s.passage || !s.passage.length) { wrap.innerHTML = ''; return; }
  wrap.innerHTML = s.passage.map((line, i) =>
    `<div class="passage-line"><span class="idx">${i + 1}.</span>${line}</div>`
  ).join('');
}

function renderNurseryRecall(s) {
  const banner = document.getElementById('nursery-recall');
  if (s.recall_true === null || s.recall_true === undefined) { banner.style.display = 'none'; return; }
  const match = s.recall_true === s.recall_pred;
  banner.style.display = 'flex';
  banner.className = 'verify-banner ' + (match ? 'match' : 'mismatch');
  banner.innerHTML =
    `<span style="color:var(--dim)">recall:</span> ` +
    `<span class="verify-verdict yes">actual: ${s.recall_true}</span> ` +
    `<span class="verify-verdict ${match ? 'yes' : 'no'}">model: ${s.recall_pred}</span>` +
    `<span style="margin-left:auto;color:${match ? 'var(--success)' : 'var(--danger)'}">${match ? '&#10003; match' : '&#10007; mismatch'}</span>`;
}

function renderNurseryStats(s) {
  const grid = document.getElementById('nursery-stats');
  const cur = s.metrics.current || {};
  const entries = Object.entries(cur);
  grid.innerHTML = entries.map(([name, val]) => `
    <div class="stat"><div class="stat-label">${name}</div><div class="stat-val accent">${(val*100).toFixed(1)}%</div></div>
  `).join('') || '<div class="stat"><div class="stat-label">metrics</div><div class="stat-val">-</div></div>';
}

function renderArchitecture(s) {
  const svg = document.getElementById('arch-diagram');
  const badge = document.getElementById('arch-gate-badge');
  const note = document.getElementById('arch-note');
  svg.innerHTML = '';

  if (!s.memory_slots) {
    badge.textContent = 'n/a';
    note.textContent = 'not instrumented for this stage yet (L5 / L5-stress only, real per-slot tensors)';
    const txt = el('text', { x: 160, y: 40, 'text-anchor': 'middle', class: 'arch-label' });
    txt.textContent = 'architecture introspection not wired for this stage';
    svg.appendChild(txt);
    return;
  }

  badge.textContent = 'write gate ' + (s.mean_gate * 100).toFixed(0) + '%';
  note.textContent = "real per-slot L2-norm activity from this step's actual S/H tensors";

  const W = 320;
  svg.innerHTML = '<defs><marker id="archArrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">' +
    '<path d="M0,0 L6,3 L0,6 Z" fill="var(--dim)"/></marker></defs>';
  const arrow = (x1, y1, x2, y2) => el('line', { x1, y1, x2, y2, class: 'arch-flow', 'marker-end': 'url(#archArrow)' });
  const label = (x, y, text, cls, anchor) => {
    const t = el('text', { x, y, class: cls || 'arch-label', 'text-anchor': anchor || 'start' });
    t.textContent = text;
    return t;
  };

  svg.appendChild(label(W / 2, 10, 'INPUT', 'arch-label', 'middle'));
  svg.appendChild(arrow(W / 2, 14, W / 2, 26));

  const sY = 58;
  svg.appendChild(label(8, 40, `S · PERSISTENT MEMORY (${s.memory_slots.length} slots)`));
  const sN = s.memory_slots.length;
  const sSpacing = (W - 40) / Math.max(sN - 1, 1);
  s.memory_slots.forEach((v, i) => {
    const x = 20 + i * sSpacing;
    const r = 6 + v * 7;
    svg.appendChild(el('circle', { cx: x, cy: sY, r: r + 4, fill: 'var(--accent)', opacity: (v * 0.35).toFixed(2) }));
    svg.appendChild(el('circle', {
      cx: x, cy: sY, r, fill: 'var(--accent)', opacity: (0.35 + v * 0.65).toFixed(2),
      stroke: 'var(--border)', 'stroke-width': 1,
    }));
  });

  svg.appendChild(arrow(W / 2, sY + 14, W / 2, sY + 26));

  const hSlots = s.hidden_slots || [];
  const hLblY = sY + 42;
  svg.appendChild(label(8, hLblY, `H · REASONING WORKSPACE (${hSlots.length} slots)`));
  const cols = 8;
  const rows = Math.ceil(hSlots.length / cols) || 1;
  const cellW = (W - 40) / cols;
  const cellH = 10;
  const hTop = hLblY + 10;
  hSlots.forEach((v, i) => {
    const col = i % cols, row = Math.floor(i / cols);
    const x = 20 + col * cellW;
    const y = hTop + row * (cellH + 4);
    svg.appendChild(el('rect', {
      x, y, width: cellW - 4, height: cellH, rx: 2,
      fill: 'var(--amber)', opacity: (0.25 + v * 0.7).toFixed(2),
    }));
  });

  const afterH = hTop + rows * (cellH + 4) + 8;
  svg.appendChild(arrow(W / 2, afterH, W / 2, afterH + 16));
  svg.appendChild(label(W / 2, afterH + 30, 'READOUT → PREDICTION', 'arch-label', 'middle'));
  svg.appendChild(el('circle', { cx: W / 2, cy: afterH + 38, r: 5, fill: 'var(--success)' }));
}

function drawNurseryChart(metrics) {
  const svg = document.getElementById('nursery-chart');
  svg.innerHTML = '';
  const history = metrics.history || {};
  const names = Object.keys(history).filter(k => (history[k] || []).length > 0);
  const allVals = [];
  names.forEach(k => allVals.push(...history[k]));
  Object.values(metrics.chance || {}).forEach(v => allVals.push(v));
  Object.values(metrics.baseline || {}).forEach(v => allVals.push(v));
  if (!allVals.length) return;
  const min = Math.min(...allVals, 0), max = Math.max(...allVals);
  const range = (max - min) || 1;
  const w = 300, h = 90;

  const refs = [];
  Object.values(metrics.chance || {}).forEach(v => refs.push({ v, color: '#707893', dash: '4,4' }));
  Object.values(metrics.baseline || {}).forEach(v => refs.push({ v, color: '#ff6b6b', dash: '2,3' }));
  refs.forEach(r => {
    const y = h - ((r.v - min) / range) * h;
    svg.appendChild(el('line', { x1: 0, y1: y, x2: w, y2: y, stroke: r.color, 'stroke-width': 1, 'stroke-dasharray': r.dash, opacity: 0.7 }));
  });

  names.forEach((name, idx) => {
    const data = history[name];
    const n = data.length;
    const pts = data.map((v, i) => { const x = n === 1 ? w : (i / (n - 1)) * w; const y = h - ((v - min) / range) * h; return [x, y]; });
    const path = 'M ' + pts.map(p => p.join(',')).join(' L ');
    svg.appendChild(el('path', { d: path, fill: 'none', stroke: NURSERY_PALETTE[idx % NURSERY_PALETTE.length], 'stroke-width': 2 }));
  });

  const legend = document.getElementById('nursery-chart-legend');
  let html = '';
  names.forEach((name, idx) => {
    const data = history[name];
    const cur = data[data.length - 1];
    html += `<div class="legend-item"><span class="swatch line" style="background:${NURSERY_PALETTE[idx % NURSERY_PALETTE.length]}"></span>${name}: <b style="color:var(--text)">${(cur*100).toFixed(1)}%</b></div>`;
  });
  Object.entries(metrics.chance || {}).forEach(([k, v]) => {
    html += `<div class="legend-item" style="opacity:.65">chance: ${(v*100).toFixed(1)}%</div>`;
  });
  Object.entries(metrics.baseline || {}).forEach(([k, v]) => {
    html += `<div class="legend-item" style="opacity:.85;color:var(--danger)">shortcut baseline: ${(v*100).toFixed(1)}%</div>`;
  });
  legend.innerHTML = html;
}

function renderNursery(s) {
  lastFetchOk = Date.now();
  document.getElementById('dot').className = 'dot on';
  document.getElementById('live-text').textContent = 'live';
  document.getElementById('room-view').style.display = 'none';
  document.getElementById('nursery-view').style.display = 'grid';

  document.getElementById('nursery-stage-badge').textContent = s.stage;
  document.getElementById('nursery-step-badge').textContent = s.step + ' / ' + s.total_steps;
  renderStageTrack(s);
  renderNurseryPassage(s);
  document.getElementById('nursery-instruction').textContent = s.instruction || '-';
  renderNurseryObjects(s);
  renderNurseryTokens(s);
  renderNurseryConsequence(s);
  renderNurseryVerification(s);
  renderNurseryRecall(s);
  renderNurseryStats(s);
  drawNurseryChart(s.metrics);
  renderArchitecture(s);
}

function render(s) {
  document.getElementById('room-view').style.display = 'grid';
  document.getElementById('nursery-view').style.display = 'none';
  lastFetchOk = Date.now();
  document.getElementById('dot').className = 'dot on';
  document.getElementById('live-text').textContent = 'live';
  document.getElementById('agent_type').textContent = s.agent_type;
  document.getElementById('school_level').textContent = s.school_level || '-';
  document.getElementById('episode').textContent = s.episode;
  document.getElementById('step').textContent = s.step;
  document.getElementById('plan_len').textContent = s.plan_len;
  document.getElementById('room-count-badge').textContent = s.n_rooms + ' rooms';

  document.getElementById('reward').textContent = s.last_reward.toFixed(3);
  document.getElementById('reward').className = 'stat-val ' + (s.last_reward > 0 ? 'good' : (s.last_reward < 0 ? 'bad' : ''));
  document.getElementById('return').textContent = s.episode_return.toFixed(3);
  document.getElementById('success').textContent = (s.recent_success_rate*100).toFixed(0) + '%';
  document.getElementById('last_action_type').innerHTML = s.last_action ? (ICONS[s.last_action.type]+' '+s.last_action.text) : '&mdash;';

  const inv = document.getElementById('inventory');
  inv.innerHTML = s.inventory.map((c,i) =>
    `<div class="key-chip ${c===0?'zero':''}"><span class="key-icon" style="background:${ACTUAL_LOCK_COLORS[i%6]}"></span>Key ${String.fromCharCode(65+i)} &times; ${c}</div>`
  ).join('');

  renderLegend(s.n_colors);
  drawWorld(s);
  drawChart(s.return_history || []);
  document.getElementById('chart-n').textContent = 'n=' + (s.return_history||[]).length;

  const ticks = document.getElementById('ticks');
  ticks.innerHTML = (s.success_history||[]).map(v => `<div class="tick ${v?'win':'loss'}"></div>`).join('');

  if (s.last_action && s._new_step) {
    const log = document.getElementById('log');
    const entry = document.createElement('div');
    entry.className = 'log-entry ' + s.last_action.type;
    entry.innerHTML = `<span class="log-icon">${ICONS[s.last_action.type]}</span><span class="log-text">ep${s.episode}.${s.step} ${s.last_action.text}</span><span class="log-reward">${s.last_reward>=0?'+':''}${s.last_reward.toFixed(2)}</span>`;
    log.appendChild(entry);
    while (log.children.length > 40) log.removeChild(log.firstChild);
  }
}

let lastKey = null;
async function poll() {
  try {
    const res = await fetch('/state');
    if (res.ok) {
      const s = await res.json();
      if (!s.error) {
        if (s.kind === 'nursery') {
          renderNursery(s);
        } else {
          const key = s.episode + ':' + s.step;
          s._new_step = key !== lastKey;
          lastKey = key;
          render(s);
        }
      }
    }
  } catch (e) {}
  if (Date.now() - lastFetchOk > 3000 && lastFetchOk > 0) {
    document.getElementById('dot').className = 'dot stale';
    document.getElementById('live-text').textContent = 'stale';
  }
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
