// Inline-SVG visualizations for the comparison screen. No external libraries
// (CSP forbids them). Every chart is driven by measured values from the run.
import { el } from "./api.js";

const NS = "http://www.w3.org/2000/svg";
// Vivid, distinct, light/dark-safe colours assigned by model column order.
export const MODEL_COLORS = ["#3b82f6", "#10b981", "#f59e0b"];

function s(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  return n;
}
function hexToRgba(hex, a) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

/**
 * Overlaid line chart. opts:
 *   series: [{label, color, points:[number|null], dashed?}]
 *   height, yMin, yMax, yTicks, yFmt(v), xName, relativeX (align by relative position),
 *   xStartLabel, xEndLabel, area (faint fill under each line)
 */
export function lineChart(opts) {
  const W = 640, H = opts.height || 210, m = { t: 12, r: 16, b: 30, l: 46 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const allY = opts.series.flatMap((ser) => ser.points.filter((v) => typeof v === "number"));
  let yMin = opts.yMin, yMax = opts.yMax;
  if (yMin === undefined) yMin = Math.min(...allY, 0);
  if (yMax === undefined) yMax = Math.max(...allY, 1);
  if (yMin === yMax) yMax = yMin + 1;
  const maxLen = Math.max(...opts.series.map((ser) => ser.points.length), 1);
  const xOf = (i, len) => m.l + (opts.relativeX ? (len > 1 ? i / (len - 1) : 0)
    : (maxLen > 1 ? i / (maxLen - 1) : 0)) * iw;
  const yOf = (v) => m.t + ih - ((v - yMin) / (yMax - yMin)) * ih;
  const yFmt = opts.yFmt || ((v) => (+v).toFixed(2));

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", preserveAspectRatio: "xMidYMid meet",
    role: "img", "aria-label": (opts.title || "chart") + " — " + opts.series.map((x) => x.label).join(", ") });

  const ticks = opts.yTicks || 4;
  for (let t = 0; t <= ticks; t++) {
    const v = yMin + (yMax - yMin) * t / ticks, y = yOf(v);
    svg.append(s("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, class: "grid" }));
    const lbl = s("text", { x: m.l - 7, y: y + 3.5, class: "axis", "text-anchor": "end" });
    lbl.textContent = yFmt(v); svg.append(lbl);
  }
  // x baseline labels
  const xa = s("text", { x: m.l, y: H - 9, class: "axis", "text-anchor": "start" });
  xa.textContent = opts.xStartLabel || ""; svg.append(xa);
  const xb = s("text", { x: m.l + iw, y: H - 9, class: "axis", "text-anchor": "end" });
  xb.textContent = opts.xEndLabel || ""; svg.append(xb);

  for (const ser of opts.series) {
    const pts = ser.points.map((v, i) => (typeof v === "number")
      ? `${xOf(i, ser.points.length).toFixed(1)},${yOf(v).toFixed(1)}` : null).filter(Boolean);
    if (opts.area && pts.length) {
      const first = pts[0].split(",")[0], last = pts[pts.length - 1].split(",")[0];
      const area = s("polygon", { points: `${first},${yOf(yMin)} ${pts.join(" ")} ${last},${yOf(yMin)}`,
        fill: hexToRgba(ser.color, 0.12), stroke: "none" });
      svg.append(area);
    }
    const line = s("polyline", { points: pts.join(" "), fill: "none", stroke: ser.color,
      "stroke-width": 2.2, "stroke-linejoin": "round", "stroke-linecap": "round" });
    if (ser.dashed) line.setAttribute("stroke-dasharray", "5 3");
    svg.append(line);
    ser.points.forEach((v, i) => {
      if (typeof v !== "number") return;
      const c = s("circle", { cx: xOf(i, ser.points.length), cy: yOf(v), r: 2.6, fill: ser.color });
      const ti = s("title"); ti.textContent = `${ser.label} · ${opts.xName || "x"} ${i}: ${yFmt(v)}`;
      c.append(ti); svg.append(c);
    });
  }
  const fig = el("figure", { class: "chartfig" });
  if (opts.title) fig.append(el("figcaption", { class: "chart-title" }, opts.title));
  fig.append(svg);
  if (opts.series.length > 1 || opts.legend) fig.append(legend(opts.series));
  return fig;
}

export function legend(series) {
  const l = el("div", { class: "legend" });
  for (const ser of series) {
    l.append(el("span", { class: "legend-item" },
      el("span", { class: "swatch", style: `background:${ser.color}` + (ser.dashed ? ";opacity:.6" : "") }),
      ser.label));
  }
  return l;
}

// ---- shared floating tooltip (positioned via CSSOM, CSP-safe) ----
let TIP = null;
function tip(ev, text) {
  if (!TIP) { TIP = el("div", { class: "viz-tip" }); document.body.append(TIP); }
  if (!text) { TIP.hidden = true; return; }
  TIP.hidden = false; TIP.textContent = text;
  const r = TIP.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > window.innerWidth - 4) x = ev.clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 4) y = ev.clientY - r.height - 14;
  TIP.style.left = x + "px"; TIP.style.top = y + "px";
}

/**
 * Layer x token heatmap (canvas). tl = token_layer grid from the run; metric = key of
 * tl ("h_norm" | "attn_norm" | "mlp_norm" | "attn_entropy"), values [token][layer].
 * Colour is clipped at the 98th percentile so one outlier (e.g. an attention-sink
 * token) does not wash out the rest; the colour bar states the exact range.
 */
export function heatmap(tl, metric, color, metricName) {
  const rows = tl[metric] || [];
  const T = rows.length, L = tl.layers;
  const vals = [];
  rows.forEach((r) => r.forEach((v) => { if (typeof v === "number") vals.push(v); }));
  const wrap = el("div", { class: "heatmap" });
  if (!T || !vals.length) { wrap.append(el("p", { class: "lede", text: "No measured values for this metric." })); return wrap; }
  vals.sort((a, b) => a - b);
  const vmin = vals[0], vclip = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.98))] || vals[vals.length - 1];
  const vmax = vals[vals.length - 1];
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const W = 960, cellH = Math.max(6, Math.min(12, 300 / L)), H = L * cellH;
  const cv = document.createElement("canvas");
  cv.width = W * dpr; cv.height = H * dpr; cv.className = "heat-canvas";
  cv.setAttribute("role", "img");
  cv.setAttribute("aria-label", `${metricName}: ${L} layers by ${T} tokens; range ${vmin.toFixed(2)} to ${vmax.toFixed(2)}`);
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const cw = W / T;
  for (let i = 0; i < T; i++) for (let l = 0; l < L; l++) {
    const v = rows[i][l];
    if (typeof v !== "number") continue;
    const t = Math.max(0, Math.min(1, (v - vmin) / ((vclip - vmin) || 1)));
    ctx.fillStyle = hexToRgba(color, 0.06 + 0.94 * Math.sqrt(t));
    ctx.fillRect(i * cw, l * cellH, Math.ceil(cw), Math.ceil(cellH));
  }
  // prompt | generated boundary
  const firstGen = tl.tokens.findIndex((x) => x.segment === "generated");
  if (firstGen > 0) {
    ctx.strokeStyle = "rgba(220,60,60,0.9)"; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(firstGen * cw, 0); ctx.lineTo(firstGen * cw, H); ctx.stroke();
  }
  cv.addEventListener("mousemove", (ev) => {
    const b = cv.getBoundingClientRect();
    const i = Math.floor(((ev.clientX - b.left) / b.width) * T), l = Math.floor(((ev.clientY - b.top) / b.height) * L);
    if (i < 0 || i >= T || l < 0 || l >= L) return tip(ev, null);
    const v = rows[i][l], tk = tl.tokens[i];
    tip(ev, `token ${tk.pos} ${JSON.stringify(tk.text)} (${tk.segment}) · layer ${l} · ${metricName} ${typeof v === "number" ? v.toFixed(3) : "—"}`);
  });
  cv.addEventListener("mouseleave", (ev) => tip(ev, null));
  wrap.append(el("div", { class: "heat-axis" }, el("span", { text: "layer 0 (top) → layer " + (L - 1) + " (bottom)" }),
    el("span", { text: firstGen > 0 ? "red dashes: prompt | generated" : "" })), cv,
    el("div", { class: "heat-legend" }, fmtV(vmin), el("span", { class: "heat-grad", style: `background:linear-gradient(90deg, ${hexToRgba(color, 0.06)}, ${color})` }),
      fmtV(vclip) + (vclip < vmax ? ` (clipped; max ${fmtV(vmax)})` : ""), ` · ${T} tokens →`));
  return wrap;
}
function fmtV(v) { return Math.abs(v) >= 100 ? Math.round(v).toString() : (+v).toFixed(2); }

/**
 * Release timeline: x = verified first-publication date, y = parameter count (log scale).
 * models: [{id, label, released:"YYYY-MM-DD", params, runnable, moe}]
 */
export function releaseTimeline(models) {
  const W = 1000, H = 430, m = { t: 20, r: 190, b: 40, l: 56 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const pts = models.filter((x) => x.released && x.params);
  const t = (d) => new Date(d + "T00:00:00Z").getTime();
  const t0 = Date.UTC(2024, 6, 1), t1 = Date.UTC(2027, 0, 1);
  const lo = Math.log10(2e8), hi = Math.log10(5e10);
  const X = (d) => m.l + ((t(d) - t0) / (t1 - t0)) * iw;
  const Y = (p) => m.t + ih - ((Math.log10(p) - lo) / (hi - lo)) * ih;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart timeline", role: "img",
    "aria-label": "Release timeline: date versus parameter count for " + pts.length + " models" });

  // y grid (log params)
  for (const [v, lab] of [[3e8, "300M"], [1e9, "1B"], [3e9, "3B"], [1e10, "10B"], [3e10, "30B"]]) {
    svg.append(s("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), class: "grid" }));
    const tx = s("text", { x: m.l - 8, y: Y(v) + 4, class: "axis", "text-anchor": "end" }); tx.textContent = lab; svg.append(tx);
  }
  // x grid (quarters)
  for (let y = 2024; y <= 2026; y++) for (const q of [0, 3, 6, 9]) {
    const ts = Date.UTC(y, q, 1); if (ts < t0 || ts > t1) continue;
    const x = m.l + ((ts - t0) / (t1 - t0)) * iw;
    svg.append(s("line", { x1: x, x2: x, y1: m.t, y2: m.t + ih, class: "grid", opacity: q === 0 ? 1 : 0.45 }));
    const tx = s("text", { x, y: H - 14, class: "axis", "text-anchor": "middle" });
    tx.textContent = q === 0 ? String(y) : ["", "", "", "Apr", "", "", "Jul", "", "", "Oct"][q]; svg.append(tx);
  }
  const yl = s("text", { x: 14, y: m.t + ih / 2, class: "axis", transform: `rotate(-90 14 ${m.t + ih / 2})`, "text-anchor": "middle" });
  yl.textContent = "parameters (log)"; svg.append(yl);

  // place labels, nudging to avoid overlaps
  const rOf = (p) => 5 + 3 * Math.log10(p.params / 2e8);
  const dots = pts.map((p) => ({ x: X(p.released), y: Y(p.params), r: rOf(p) }));
  const placed = [];  // label boxes {x0,x1,y0,y1}
  const hits = (b) => placed.some((q) => b.x0 < q.x1 && b.x1 > q.x0 && b.y0 < q.y1 && b.y1 > q.y0)
    || dots.some((d) => d.x + d.r > b.x0 && d.x - d.r < b.x1 && d.y + d.r > b.y0 && d.y - d.r < b.y1);
  const sorted = [...pts].sort((a, b) => t(a.released) - t(b.released) || b.params - a.params);
  for (const p of sorted) {
    const cx = X(p.released), cy = Y(p.params), r = rOf(p);
    const w = p.label.length * 6.7;
    // candidate label positions: right of the dot, then left, at increasing vertical offsets
    const boxAt = (side, ly) => side > 0
      ? { x0: cx + r + 9, x1: cx + r + 11 + w, y0: ly - 11, y1: ly + 3, side }
      : { x0: cx - r - 11 - w, x1: cx - r - 9, y0: ly - 11, y1: ly + 3, side };
    let chosen = null;
    outer: for (const d of [0, -15, 15, -30, 30, -45, 45, -60, 60, -75, 75, -90, 90]) {
      for (const side of [1, -1]) {
        const b = boxAt(side, cy + 4 + d);
        if (b.x0 >= m.l && b.x1 <= W - 4 && !hits(b)) { chosen = b; break outer; }
      }
    }
    if (!chosen) chosen = boxAt(1, cy + 4);  // fall back to the plain position
    placed.push(chosen);
    const ly = chosen.y1 - 3;
    const a = s("a", { href: "model.html?id=" + encodeURIComponent(p.id) });
    const c = s("circle", { cx, cy, r, class: "tl-dot " + (p.runnable ? "run" : "ref") });
    const ti = s("title");
    ti.textContent = `${p.label} — released ${p.released} · ${(p.params / 1e9).toFixed(2)}B params${p.moe ? " (MoE total)" : ""} · ${p.runnable ? "runnable with this tool" : "reference only"}`;
    c.append(ti); a.append(c);
    const sd = chosen.side;
    if (Math.abs(ly - (cy + 4)) > 1)
      a.append(s("line", { x1: cx + sd * r, y1: cy, x2: cx + sd * (r + 8), y2: ly - 4, class: "tl-leader" }));
    const lab = s("text", { x: cx + sd * (r + 10), y: ly, "text-anchor": sd > 0 ? "start" : "end",
      class: "tl-label" + (p.runnable ? " run" : "") });
    lab.textContent = p.label; a.append(lab);
    svg.append(a);
  }
  const fig = el("figure", { class: "chartfig" });
  fig.append(svg, el("div", { class: "legend" },
    el("span", { class: "legend-item" }, el("span", { class: "swatch tl-sw-run" }), "runnable with this tool (pinned + verified engine)"),
    el("span", { class: "legend-item" }, el("span", { class: "swatch tl-sw-ref" }), "reference only"),
    "dot size ∝ log(parameters) · click a model for its page"));
  return fig;
}

/** Per-token confidence heatstrip: each generated token shaded by chosen probability. */
export function tokenHeatstrip(decodeSeries, color) {
  const wrap = el("div", { class: "heatstrip", role: "img",
    "aria-label": "per-token confidence; darker is more confident" });
  for (const tok of decodeSeries) {
    const p = typeof tok.p === "number" ? tok.p : 0;
    const label = tok.text.replace(/\n/g, "⏎").replace(/^ +/, (sp) => "·".repeat(sp.length));
    const chip = el("span", { class: "tok", style: `background:${hexToRgba(color, 0.18 + 0.82 * p)}` },
      label || "·");
    chip.title = `${JSON.stringify(tok.text)} → ${(p * 100).toFixed(1)}% confident`;
    if (p > 0.6) chip.style.color = "#fff";
    wrap.append(chip);
  }
  const strip = el("div", {});
  strip.append(wrap, el("div", { class: "heat-legend" },
    "less confident ", el("span", { class: "heat-grad", style: `background:linear-gradient(90deg, ${hexToRgba(color, 0.18)}, ${color})` }), " more"));
  return strip;
}
