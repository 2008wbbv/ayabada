"""The dashboard single page. Served embedded so self-hosting stays one file.

Design notes (dataviz method):
- Palette is the validated reference set. Categorical slots 1–3 carry the
  three metric series in the z-chart (CVD worst-adjacent ΔE 47 light / 41
  dark). Aqua and yellow sit below 3:1 on the light surface, so the chart
  ships relief: an always-visible current-value readout row plus the
  all-series tooltip — identity never rides on color alone.
- Status colors are reserved for gate state (asleep→good, candidate→warning,
  cooldown→serious, awake→critical) and always appear as dot + text label.
- One axis per chart. Raw metrics have wildly different scales, so they are
  small multiples (single blue series each), never a dual axis.
- Text wears text tokens; marks wear series color. Hairline solid gridlines.
- Crosshair + tooltip on every plot; tooltips enhance, the incidents table
  and readout rows keep everything reachable without hovering.
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ayabada — SRE agent dashboard</title>
<style>
  :root {
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --grid: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;  /* blue   — traffic */
    --series-2: #1baf7a;  /* aqua   — error_rate */
    --series-3: #eda100;  /* yellow — p95_latency_ms */
    --series-4: #008300;  /* green  — spare slot */
    --status-good: #0ca30c;
    --status-warning: #fab219;
    --status-serious: #ec835a;
    --status-critical: #d03b3b;
    --wash-opacity: 0.10;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --grid: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-2: #199e70;
      --series-3: #c98500;
      --series-4: #008300;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--text-primary);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 20px 24px 48px; }
  header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 6px; }
  header h1 { font-size: 20px; font-weight: 650; margin: 0; }
  header .sub { color: var(--text-secondary); }
  header .clock { margin-left: auto; color: var(--text-muted); font-variant-numeric: tabular-nums; }

  .filters { display: flex; gap: 8px; align-items: center; margin: 14px 0 16px; }
  .filters .label { color: var(--text-muted); margin-right: 2px; }
  .chip {
    border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
    border-radius: 999px; padding: 4px 12px; cursor: pointer; font: inherit;
  }
  .chip:hover { color: var(--text-primary); }
  .chip[aria-pressed="true"] { color: var(--text-primary); border-color: var(--text-secondary); font-weight: 600; }

  .cards { display: grid; gap: 14px; }
  .kpis { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .tile .label { color: var(--text-secondary); font-size: 13px; }
  .tile .value { font-size: 30px; font-weight: 650; margin-top: 2px; }
  .tile .note { color: var(--text-muted); font-size: 12px; margin-top: 2px; min-height: 16px; }
  .statusdot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 7px; vertical-align: baseline; }

  .card h2 { font-size: 14px; font-weight: 650; margin: 0 0 2px; }
  .card .desc { color: var(--text-muted); font-size: 12px; margin: 0 0 10px; }
  .chartwrap { position: relative; }
  svg { display: block; width: 100%; height: auto; }
  .smalls { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-top: 14px; }

  .readout { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px; }
  .readout .item { display: flex; align-items: center; gap: 7px; color: var(--text-secondary); font-size: 13px; }
  .readout .key { width: 14px; height: 0; border-top: 3px solid; border-radius: 2px; }
  .readout .val { color: var(--text-primary); font-weight: 600; font-variant-numeric: tabular-nums; }

  .tooltip {
    position: absolute; pointer-events: none; display: none; z-index: 5;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.13); padding: 8px 11px; min-width: 150px;
  }
  .tooltip .t-when { color: var(--text-muted); font-size: 12px; margin-bottom: 5px; }
  .tooltip .t-row { display: flex; align-items: center; gap: 7px; margin: 2px 0; }
  .tooltip .t-key { width: 12px; height: 0; border-top: 3px solid; border-radius: 2px; flex: none; }
  .tooltip .t-val { font-weight: 650; font-variant-numeric: tabular-nums; }
  .tooltip .t-name { color: var(--text-secondary); font-size: 12px; }

  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--text-muted); font-weight: 500; font-size: 12px; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
  td { padding: 8px 10px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
  tr.inc { cursor: pointer; }
  tr.inc:hover td { background: color-mix(in srgb, var(--text-muted) 7%, transparent); }
  tr.inc[aria-selected="true"] td { background: color-mix(in srgb, var(--series-1) 10%, transparent); }
  .badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; }
  .empty { color: var(--text-muted); padding: 14px 10px; }

  .handoff { display: none; margin-top: 14px; }
  .handoff pre {
    margin: 0; white-space: pre-wrap; overflow-wrap: anywhere;
    font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    color: var(--text-secondary);
  }
  .handoff .close { float: right; }
  section { margin-top: 14px; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 6px; }
  .legend .item { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); font-size: 12.5px; }
  .legend .key { width: 16px; height: 0; border-top: 3px solid; border-radius: 2px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Ayabada</h1>
    <span class="sub">self-hosted SRE agent — heartbeat &amp; incidents</span>
    <span class="clock" id="clock">—</span>
  </header>

  <div class="filters" role="group" aria-label="Time range">
    <span class="label">Range</span>
    <button class="chip" data-range="24">6&nbsp;h</button>
    <button class="chip" data-range="48">12&nbsp;h</button>
    <button class="chip" data-range="96" aria-pressed="true">24&nbsp;h</button>
    <button class="chip" data-range="192">48&nbsp;h</button>
  </div>

  <div class="cards kpis" id="kpis"></div>

  <section class="card">
    <h2>Seasonal deviation</h2>
    <p class="desc">Robust z-score of each metric against its own (weekday × 15-min) bucket — what the detector actually judges. Hairlines mark the corroboration (±3) and severity (±6) thresholds; ▲ marks a wake.</p>
    <div class="legend" id="zlegend"></div>
    <div class="chartwrap" id="zchart"></div>
    <div class="readout" id="zreadout"></div>
  </section>

  <div class="cards smalls" id="smalls"></div>

  <section class="card">
    <h2>Self-host services</h2>
    <p class="desc">Health of the stack itself: the dashboard, the feed driving the wake gate, the brain that runs on wakes, and the action-layer tools available on this host.</p>
    <div id="services"></div>
  </section>

  <section class="card">
    <h2>Incidents</h2>
    <p class="desc">Every wake, what the brain concluded, and the handoff written for a human. Click a row to read the handoff.</p>
    <div id="incidents"></div>
    <div class="handoff card" id="handoff">
      <button class="chip close" id="handoff-close">close</button>
      <pre id="handoff-body"></pre>
    </div>
  </section>
</div>

<script>
"use strict";
const SVGNS = "http://www.w3.org/2000/svg";
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4"];
const STATUS = {
  asleep:    { color: "--status-good",     label: "Asleep",    icon: "●" },
  candidate: { color: "--status-warning",  label: "Candidate", icon: "◐" },
  awake:     { color: "--status-critical", label: "Awake",     icon: "▲" },
  cooldown:  { color: "--status-serious",  label: "Cooldown",  icon: "◍" },
};

let rangeIntervals = 96;
let selectedIncident = null;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}
function svgEl(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== null) node.textContent = text;
  return node;
}
const fmt = v => {
  if (v === null || v === undefined) return "—";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + "K";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2).replace(/\.?0+$/, "");
  return v.toPrecision(2);
};
const fmtTime = iso => iso ? iso.slice(11, 16) : "";
const fmtDay = iso => iso ? iso.slice(5, 10) + " " + iso.slice(11, 16) : "";

function niceTicks(lo, hi, n = 4) {
  if (!(isFinite(lo) && isFinite(hi)) || lo === hi) { hi = lo + 1; }
  const span = hi - lo;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + 1) || mag * 10;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

/* Generic line chart.
   cfg: { mount, series: [{name, cssVar, points:[[iso,val],…]}], height,
          refLines: [{y, label}], markers: [iso…], area: bool, yLabel } */
function lineChart(cfg) {
  const W = cfg.width || 960, H = cfg.height || 230;
  const M = { top: 14, right: 18, bottom: 24, left: 48 };
  const iw = W - M.left - M.right, ih = H - M.top - M.bottom;
  const mount = cfg.mount;
  mount.textContent = "";

  const xs = cfg.series[0] ? cfg.series[0].points.map(p => p[0]) : [];
  const n = xs.length;
  if (!n) { mount.append(el("div", { class: "empty", text: "waiting for data…" })); return; }

  let lo = Infinity, hi = -Infinity;
  for (const s of cfg.series) for (const p of s.points) {
    if (p[1] === null) continue;
    lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]);
  }
  for (const r of cfg.refLines || []) { lo = Math.min(lo, r.y); hi = Math.max(hi, r.y); }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;

  const X = i => M.left + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const Y = v => M.top + ih - ((v - lo) / (hi - lo)) * ih;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  for (const t of niceTicks(lo, hi)) {
    svg.append(svgEl("line", { x1: M.left, x2: M.left + iw, y1: Y(t), y2: Y(t), stroke: css("--grid"), "stroke-width": 1 }));
    svg.append(svgEl("text", { x: M.left - 8, y: Y(t) + 4, "text-anchor": "end", fill: css("--text-muted"), "font-size": 11 }, fmt(t)));
  }
  const nTicks = Math.max(3, Math.round(iw / 160));
  const tickEvery = Math.max(1, Math.round(n / nTicks));
  for (let i = 0; i < n; i += tickEvery) {
    svg.append(svgEl("text", { x: X(i), y: H - 6, "text-anchor": "middle", fill: css("--text-muted"), "font-size": 11 }, fmtTime(xs[i])));
  }
  svg.append(svgEl("line", { x1: M.left, x2: M.left + iw, y1: Y(Math.max(lo, Math.min(0, hi))), y2: Y(Math.max(lo, Math.min(0, hi))), stroke: css("--baseline"), "stroke-width": 1 }));

  // Threshold labels sit inside-left, above their hairline, and skip when
  // they'd collide with an already-placed label (the lines bunch together
  // whenever an incident stretches the z-range).
  const usedLabelYs = [];
  for (const ref of cfg.refLines || []) {
    svg.append(svgEl("line", { x1: M.left, x2: M.left + iw, y1: Y(ref.y), y2: Y(ref.y), stroke: css("--baseline"), "stroke-width": 1 }));
    const ly = Y(ref.y) - 4;
    if (usedLabelYs.every(u => Math.abs(u - ly) >= 12)) {
      usedLabelYs.push(ly);
      svg.append(svgEl("text", { x: M.left + 4, y: ly, fill: css("--text-muted"), "font-size": 10.5 }, ref.label));
    }
  }

  for (const iso of cfg.markers || []) {
    const i = xs.indexOf(iso.slice(0, 19));
    const j = i >= 0 ? i : xs.findIndex(x => x >= iso);
    if (j < 0) continue;
    svg.append(svgEl("line", { x1: X(j), x2: X(j), y1: M.top, y2: M.top + ih, stroke: css("--status-critical"), "stroke-width": 1, opacity: 0.55 }));
    svg.append(svgEl("text", { x: X(j), y: M.top - 3, "text-anchor": "middle", fill: css("--status-critical"), "font-size": 10 }, "▲"));
  }

  for (const s of cfg.series) {
    const color = css(s.cssVar);
    const coords = s.points.map((p, i) => p[1] === null ? null : `${X(i)},${Y(p[1])}`);
    const path = coords.filter(Boolean).map((c, i) => (i === 0 ? "M" : "L") + c).join(" ");
    if (cfg.area) {
      const first = coords.findIndex(Boolean);
      const last = coords.length - 1;
      const areaPath = path + ` L ${X(last)},${Y(Math.max(lo, Math.min(0, hi)))} L ${X(first)},${Y(Math.max(lo, Math.min(0, hi)))} Z`;
      svg.append(svgEl("path", { d: areaPath, fill: color, opacity: "var(--wash-opacity)", stroke: "none" }));
    }
    svg.append(svgEl("path", { d: path, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const lastIdx = s.points.length - 1;
    if (s.points[lastIdx] && s.points[lastIdx][1] !== null) {
      svg.append(svgEl("circle", { cx: X(lastIdx), cy: Y(s.points[lastIdx][1]), r: 4, fill: color, stroke: css("--surface-1"), "stroke-width": 2 }));
    }
  }

  const cross = svgEl("line", { y1: M.top, y2: M.top + ih, stroke: css("--text-muted"), "stroke-width": 1, opacity: 0, "pointer-events": "none" });
  svg.append(cross);
  const hit = svgEl("rect", { x: M.left, y: M.top, width: iw, height: ih, fill: "transparent" });
  svg.append(hit);

  const tip = el("div", { class: "tooltip" });
  mount.append(svg, tip);

  function showTip(clientX) {
    const rect = svg.getBoundingClientRect();
    const px = (clientX - rect.left) * (W / rect.width);
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - M.left) / iw) * (n - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i));
    cross.setAttribute("opacity", 0.6);
    tip.textContent = "";
    tip.append(el("div", { class: "t-when", text: fmtDay(xs[i]) }));
    for (const s of cfg.series) {
      const v = s.points[i] ? s.points[i][1] : null;
      const row = el("div", { class: "t-row" });
      const key = el("span", { class: "t-key" }); key.style.borderTopColor = css(s.cssVar);
      row.append(key, el("span", { class: "t-val", text: (cfg.tipFormat || fmt)(v, s, i) }), el("span", { class: "t-name", text: s.name }));
      tip.append(row);
    }
    tip.style.display = "block";
    const mrect = mount.getBoundingClientRect();
    const xpx = X(i) * (rect.width / W);
    tip.style.left = Math.min(xpx + 14, mrect.width - tip.offsetWidth - 8) + "px";
    tip.style.top = "10px";
  }
  hit.addEventListener("pointermove", e => showTip(e.clientX));
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("opacity", 0); });
}

function renderKpis(data) {
  const kpis = document.getElementById("kpis");
  kpis.textContent = "";
  const gate = data.gate;
  const st = STATUS[gate.state] || STATUS.asleep;

  const stateTile = el("div", { class: "card tile" },
    el("div", { class: "label", text: "Gate state" }),
    (() => {
      const v = el("div", { class: "value" });
      const dot = el("span", { class: "statusdot" }); dot.style.background = css(st.color);
      v.append(dot, document.createTextNode(st.label));
      return v;
    })(),
    el("div", { class: "note", text: gate.reason || "" }));

  kpis.append(
    stateTile,
    el("div", { class: "card tile" },
      el("div", { class: "label", text: "Incidents (wakes)" }),
      el("div", { class: "value", text: String(gate.wakes) }),
      el("div", { class: "note", text: `persistence ${data.thresholds.persistence} · corroboration ${data.thresholds.corroboration_min}+` })),
    el("div", { class: "card tile" },
      el("div", { class: "label", text: "Anomaly streak" }),
      el("div", { class: "value", text: String(gate.streak) }),
      el("div", { class: "note", text: "consecutive corroborated intervals" })),
    el("div", { class: "card tile" },
      el("div", { class: "label", text: "Max deviation now" }),
      el("div", { class: "value", text: gate.max_z.toFixed(1) + "σ" }),
      el("div", { class: "note", text: `enter ${data.thresholds.enter}σ · severe ${data.thresholds.severe}σ` })),
  );
}

function renderZChart(data) {
  const metrics = Object.keys(data.metrics);
  const series = metrics.map((m, i) => ({
    name: m,
    cssVar: SERIES[i % SERIES.length],
    points: data.metrics[m].slice(-rangeIntervals).map(p => [p[0], p[2]]),
  }));
  const markers = data.incidents.map(inc => inc.at);
  lineChart({
    mount: document.getElementById("zchart"),
    series,
    height: 240,
    refLines: [
      { y: data.thresholds.enter, label: `enter +${data.thresholds.enter}` },
      { y: -data.thresholds.enter, label: `enter −${data.thresholds.enter}` },
      { y: data.thresholds.severe, label: `severe +${data.thresholds.severe}` },
    ],
    markers,
    tipFormat: v => v === null ? "no history" : v.toFixed(2) + "σ",
  });

  const legend = document.getElementById("zlegend");
  legend.textContent = "";
  for (const s of series) {
    const item = el("div", { class: "item" });
    const key = el("span", { class: "key" }); key.style.borderTopColor = css(s.cssVar);
    item.append(key, el("span", { text: s.name }));
    legend.append(item);
  }

  const readout = document.getElementById("zreadout");
  readout.textContent = "";
  for (const s of series) {
    const last = s.points[s.points.length - 1];
    const item = el("div", { class: "item" });
    const key = el("span", { class: "key" }); key.style.borderTopColor = css(s.cssVar);
    item.append(key, el("span", { text: s.name + " " }),
      el("span", { class: "val", text: last && last[1] !== null ? last[1].toFixed(2) + "σ" : "—" }));
    readout.append(item);
  }
}

function renderSmalls(data) {
  const smalls = document.getElementById("smalls");
  const metrics = Object.keys(data.metrics);
  const existing = new Map([...smalls.children].map(c => [c.dataset.metric, c]));
  for (const m of metrics) {
    let card = existing.get(m);
    if (!card) {
      card = el("section", { class: "card", "data-metric": m },
        el("h2", { text: m }),
        el("p", { class: "desc", text: "raw value" }),
        el("div", { class: "chartwrap" }));
      smalls.append(card);
    }
    lineChart({
      mount: card.querySelector(".chartwrap"),
      series: [{ name: m, cssVar: "--series-1", points: data.metrics[m].slice(-rangeIntervals).map(p => [p[0], p[1]]) }],
      width: 460,
      height: 170,
      area: true,
    });
  }
}

const SERVICE_STATUS = { good: "--status-good", warning: "--status-warning", critical: "--status-critical" };

function renderServices(data) {
  const mount = document.getElementById("services");
  mount.textContent = "";
  if (!data || !data.services) return;
  const table = el("table");
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "Service" }), el("th", { text: "Status" }),
    el("th", { text: "Detail" }), el("th", { text: "" }))));
  const tbody = el("tbody");
  for (const svc of data.services) {
    const badge = el("span", { class: "badge" });
    const dot = el("span", { class: "statusdot" });
    dot.style.background = css(SERVICE_STATUS[svc.status] || "--status-warning");
    badge.append(dot, document.createTextNode(svc.state_label));
    tbody.append(el("tr", {},
      el("td", { text: svc.name }),
      el("td", {}, badge),
      el("td", { text: svc.detail }),
      el("td", { text: svc.meta || "" })));
  }
  table.append(tbody);
  mount.append(table);
}

function outcomeBadge(inc) {
  const badge = el("span", { class: "badge" });
  const dot = el("span", { class: "statusdot" });
  if (inc.outcome === "diagnosed") {
    dot.style.background = css("--status-good");
    badge.append(dot, document.createTextNode("Diagnosed"));
  } else {
    dot.style.background = css("--status-warning");
    badge.append(dot, document.createTextNode("Escalated"));
  }
  return badge;
}

function renderIncidents(data) {
  const mount = document.getElementById("incidents");
  mount.textContent = "";
  if (!data.incidents.length) {
    mount.append(el("div", { class: "empty", text: "No incidents yet — the gate is watching." }));
    return;
  }
  const table = el("table");
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "When" }), el("th", { text: "Incident" }), el("th", { text: "Outcome" }),
    el("th", { text: "Root cause" }), el("th", { text: "Turns" }), el("th", { text: "Confidence" }))));
  const tbody = el("tbody");
  for (const inc of data.incidents) {
    const row = el("tr", { class: "inc", "aria-selected": String(selectedIncident === inc.id), onclick: () => openHandoff(inc.id) },
      el("td", { text: fmtDay(inc.at) }),
      el("td", { text: inc.id }),
      el("td", {}, outcomeBadge(inc)),
      el("td", { text: inc.entities.length ? inc.entities.join(", ") : "—" }),
      el("td", { text: String(inc.turns) }),
      el("td", { text: inc.final_confidence !== null ? Math.round(inc.final_confidence * 100) + "%" : "—" }));
    tbody.append(row);
  }
  table.append(tbody);
  mount.append(table);
}

async function openHandoff(id) {
  selectedIncident = id;
  const resp = await fetch("/api/incident?id=" + encodeURIComponent(id));
  if (!resp.ok) return;
  const detail = await resp.json();
  document.getElementById("handoff-body").textContent = detail.handoff;
  document.getElementById("handoff").style.display = "block";
  document.querySelectorAll("tr.inc").forEach(r => r.setAttribute("aria-selected", "false"));
  refresh();
}
document.getElementById("handoff-close").addEventListener("click", () => {
  selectedIncident = null;
  document.getElementById("handoff").style.display = "none";
});

for (const chip of document.querySelectorAll(".chip[data-range]")) {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip[data-range]").forEach(c => c.setAttribute("aria-pressed", "false"));
    chip.setAttribute("aria-pressed", "true");
    rangeIntervals = parseInt(chip.dataset.range, 10);
    refresh();
  });
}

let latest = null;
let latestServices = null;
function renderAll() {
  if (!latest) return;
  document.getElementById("clock").textContent = latest.now ? latest.now.replace("T", " ") : "—";
  renderKpis(latest);
  renderZChart(latest);
  renderSmalls(latest);
  renderServices(latestServices);
  renderIncidents(latest);
}
async function refresh() {
  try {
    const resp = await fetch("/api/state");
    if (!resp.ok) return;
    latest = await resp.json();
    renderAll();
  } catch (e) { /* keep the previous render; retry on next tick */ }
}
async function refreshServices() {
  try {
    const resp = await fetch("/api/services");
    if (!resp.ok) return;
    latestServices = await resp.json();
    renderServices(latestServices);
  } catch (e) { /* keep the previous render */ }
}
refresh();
refreshServices();
setInterval(refresh, 2000);
setInterval(refreshServices, 10000);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
</script>
</body>
</html>
"""
