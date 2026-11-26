// Three-model comparison screen. Live runs require the local loopback backend.
import { loadJSON, postJSON, health, BACKEND, fmt, el } from "./api.js";
import { initExplain, infoButton, labelWithInfo, metricMeta } from "./explain.js";
import { lineChart, tokenHeatstrip, heatmap, MODEL_COLORS } from "./viz.js";

const HEAT_METRICS = [
  ["attn_norm", "attention write ‖Δattn‖"],
  ["mlp_norm", "MLP write ‖Δmlp‖"],
  ["h_norm", "residual norm ‖h‖ after layer"],
  ["attn_entropy", "attention entropy (mean over heads, nats)"],
];

const EXAMPLES = [
  "In one sentence, explain what a neural network is.",
  "What is 17 multiplied by 24? Show your reasoning.",
  "Give three tips for writing clear code.",
  "Translate 'good morning' into French, Spanish and Japanese.",
];
let CATALOG = null, METRICS = null, HEALTH = null;

const $ = (id) => document.getElementById(id);

async function boot() {
  [CATALOG, METRICS] = await Promise.all([
    loadJSON("/api/catalog", "./catalog.json"),
    loadJSON("/api/metrics", "./explanations.json"),
  ]);
  initExplain(METRICS);
  HEALTH = await health();
  renderStatus();
  buildSelectors();
  buildPrompt();
}

function runnableModels() { return CATALOG.models.filter((m) => m.runnable); }

// A model runs here only if it is a pinned runnable entry AND the local backend has
// just verified its checkpoint (present, hashes match). Otherwise it is reference-only.
function canRun(m) {
  const s = HEALTH && HEALTH.runnable_slots && m.slot ? HEALTH.runnable_slots[m.slot] : null;
  return !!(BACKEND && m.runnable && s && s.ready);
}

function renderStatus() {
  const bar = $("statusbar");
  bar.replaceChildren();
  if (!BACKEND) {
    bar.append(
      el("span", {}, el("span", { class: "dot wait" }), "Static preview — catalog & model pages only"),
      el("span", { class: "pill metadata", text: "no backend" }),
      el("span", { text: "The live prompt comparison runs only against a local backend on your own machine. Your prompt is never sent anywhere here." }),
    );
    return;
  }
  const slots = (HEALTH && HEALTH.runnable_slots) || {};
  const total = Object.keys(slots).length;
  const ready = Object.values(slots).filter((s) => s.ready).length;
  bar.append(
    el("span", {}, el("span", { class: "dot " + (ready === total ? "ok" : ready ? "wait" : "bad") }),
      `Local backend connected — ${ready}/${total} runnable models verified on this machine`),
    el("span", { class: "pill ok", text: "loopback only" }),
    el("span", { text: "Prompts stay on your machine: the backend binds to 127.0.0.1, is never logged, and the page can reach no other origin." }),
  );
}

function optionLabel(m) {
  if (canRun(m)) return m.label + "  ✔ runs here";
  if (m.runnable) return m.label + "  · not verified on this machine";
  return m.label + "  · reference (not run)";
}

function buildSelectors() {
  const wrap = $("selectors");
  wrap.replaceChildren();
  const runnable = runnableModels();
  const defaults = ["a", "b", "c"];
  for (let i = 0; i < 3; i++) {
    const sel = el("select", { id: "slot" + i, "aria-label": `Model ${i + 1}` });
    // ALL ten models are selectable. Runnable ones produce measured internals; reference
    // ones show verified specs only (nothing is run for them).
    for (const m of CATALOG.models) {
      sel.append(el("option", { value: m.id, text: optionLabel(m) }));
    }
    const def = runnable.find((m) => m.slot === defaults[i]) || runnable[i % runnable.length];
    if (def) sel.value = def.id;
    sel.addEventListener("change", () => updateHint(i));
    const slot = el("div", { class: "slot" },
      el("label", { for: "slot" + i, text: `Model ${i + 1}` }), sel,
      el("div", { class: "hint", id: "hint" + i }));
    wrap.append(slot);
    updateHint(i);
  }
}

function modelById(id) { return CATALOG.models.find((m) => m.id === id); }

function updateHint(i) {
  const m = modelById($("slot" + i).value);
  const hint = $("hint" + i);
  hint.replaceChildren();
  if (!m) return;
  const isBase = m.tuning.startsWith("base");
  const runs = canRun(m);
  const layers = m.metadata.layers == null ? "MoE" : m.metadata.layers + "L";
  const state = runs ? "runs locally"
    : m.runnable ? "pinned, but checkpoint not verified here — specs only"
    : "specs only — not run";
  hint.append(
    el("span", { class: "tuning-tag " + (runs ? (isBase ? "base" : "instruct") : "ref"),
      text: runs ? (isBase ? "BASE" : "INSTRUCT") : "REFERENCE" }),
    " ", `${fmt.params(m.metadata.parameters)} params · ${layers} · ${m.family}`,
    el("div", { class: "mono", style: "margin-top:3px", text: `${state} · rev ${(m.revision || "").slice(0, 10)}` }),
  );
}

function buildPrompt() {
  const ex = $("examples");
  EXAMPLES.forEach((t) => ex.append(el("button", { type: "button", text: t.length > 40 ? t.slice(0, 38) + "…" : t,
    title: t, onclick: () => { $("prompt").value = t; } })));
  $("prompt").value = EXAMPLES[0];
  const btn = $("runbtn");
  if (!BACKEND) { btn.disabled = true; btn.title = "Requires the local backend"; }
  else btn.addEventListener("click", runCompare);
}

function selectedPicks() {
  return [0, 1, 2].map((i) => modelById($("slot" + i).value)).filter(Boolean);
}

async function runCompare() {
  const prompt = $("prompt").value.trim();
  const out = $("results"); const prog = $("progress");
  const picks = selectedPicks();
  if (!picks.length) { prog.textContent = "Pick at least one model."; return; }
  const colorByPos = picks.map((_, i) => MODEL_COLORS[i % MODEL_COLORS.length]);
  // de-duplicated: picking the same runnable model twice runs it once
  const runSlots = [...new Set(picks.filter(canRun).map((m) => m.slot))];

  out.replaceChildren();
  let res = null;
  if (runSlots.length) {
    if (!prompt) { prog.textContent = "Enter a prompt first."; return; }
    const maxTok = Math.max(1, Math.min(96, parseInt($("maxtok").value || "40", 10)));
    $("runbtn").disabled = true; prog.textContent = "Starting…";
    try {
      const { job_id } = await postJSON("/api/compare3", { prompt, slots: runSlots, max_new_tokens: maxTok });
      let st;
      do {
        await new Promise((r) => setTimeout(r, 700));
        st = await loadJSON("/api/compare3/status?job_id=" + job_id, "");
        const last = st.stages && st.stages[st.stages.length - 1];
        if (last) prog.textContent = `[${st.elapsed_s}s] ${last.msg}`;
      } while (st.state === "running");
      if (st.state === "error") {
        prog.innerHTML = ""; prog.append(el("div", { class: "notice bad", text: "Run failed: " + st.error }));
        $("runbtn").disabled = false; return;
      }
      prog.textContent = `Done in ${st.elapsed_s}s.`;
      res = st.result;
    } catch (e) {
      prog.innerHTML = ""; prog.append(el("div", { class: "notice bad", text: String(e.message || e) }));
      $("runbtn").disabled = false; return;
    } finally { $("runbtn").disabled = false; }
  } else {
    prog.textContent = "None of the selected models run locally — showing verified specs only.";
  }

  // one column per pick, in the chosen order; runnable -> measured, reference -> specs only
  const columns = picks.map((m, i) => (canRun(m) && res && res.models[m.slot])
    ? { kind: "run", data: res.models[m.slot], color: colorByPos[i], label: m.label }
    : { kind: "ref", data: m, color: colorByPos[i], label: m.label });
  renderComparison(columns, res, out);
}

// ---- comparison table -----------------------------------------------------
const TABLE_ROWS = ["model", "provenance", "license", "params", "dims", "answer",
  "stop_reason", "tokenization", "first_token_confidence", "mean_entropy",
  "residual_growth", "attention_entropy", "timing"];

function cellEdge(rowId) {
  const m = metricMeta(rowId); return m ? "edge " + m.kind : "edge";
}

// Backward-compatible entry: a result with all-runnable slots.
export function renderTable(res, out) {
  const columns = res.slots.map((s, i) => ({ kind: "run", data: res.models[s],
    color: MODEL_COLORS[i % MODEL_COLORS.length], label: res.models[s].label }));
  renderComparison(columns, res, out);
}

export function renderComparison(columns, res, out) {
  out = out || $("results"); out.replaceChildren();
  const refCount = columns.filter((c) => c.kind === "ref").length;
  const runCount = columns.length - refCount;
  if (refCount) out.append(el("div", { class: "notice", html:
    `<strong>${refCount}</strong> of the selected model(s) are <strong>reference-only</strong> — not in the local cache, ` +
    "or not a Qwen2 architecture this engine runs. Their columns show <strong>verified specs</strong> from the official model " +
    "card (params, architecture, licence), but nothing is run for them and they have no measured internals. Only cached Qwen2 " +
    "checkpoints produce the measured rows and the charts." }));

  const table = el("table", { class: "cmp" });
  const thead = el("thead"); const htr = el("tr");
  htr.append(el("th", { text: "" }));
  for (const col of columns) htr.append(columnHeader(col));
  thead.append(htr); table.append(thead);

  const tb = el("tbody");
  for (const rowId of TABLE_ROWS) {
    const tr = el("tr");
    tr.append(el("th", { scope: "row" }, labelWithInfo(rowId, (metricMeta(rowId) || {}).title || rowId)));
    for (const col of columns) tr.append(renderCell(rowId, col));
    tb.append(tr);
  }
  table.append(tb);
  out.append(el("div", { class: "table-wrap" }, table));

  const runCols = columns.filter((c) => c.kind === "run");
  if (runCols.length) out.append(renderViz(runCols));
  if (res) out.append(summaryStrip(res, refCount));
  out.append(el("p", { class: "footnote", html:
    "<strong>Measured</strong> values were computed locally from each pinned checkpoint this run. " +
    "<strong>Derived</strong> values are summaries of measured values. <strong>Metadata</strong> is from the model card. " +
    (res ? "Greedy decoding; " + res.parity.settings.max_new_tokens + " tokens max. " + res.provenance
         : "No model was run; every column shows verified specs from the official model cards.") }));
}

function colorDot(c) { return el("span", { class: "cdot", style: `background:${c}` }); }

function columnHeader(col) {
  const m = col.data;
  if (col.kind === "run") {
    const isBase = m.tuning.startsWith("base");
    return el("th", { class: "modelhdr" },
      el("div", { class: "mname" }, colorDot(col.color), m.label,
        el("span", { class: "tuning-tag " + (isBase ? "base" : "instruct"), text: isBase ? "BASE" : "INSTRUCT" })),
      el("div", { class: "mrepo", text: m.repo }));
  }
  return el("th", { class: "modelhdr ref" },
    el("div", { class: "mname" }, m.label,
      el("span", { class: "tuning-tag ref", text: "REFERENCE" })),
    el("div", { class: "mrepo", text: m.repo }));
}

function renderCell(rowId, col) {
  const td = el("td", { class: cellEdge(rowId) + (col.kind === "ref" ? " refcell" : "") });
  return col.kind === "run" ? runCell(td, rowId, col.data) : refCell(td, rowId, col.data);
}

function runCell(td, rowId, m) {
  const me = m.measured;
  switch (rowId) {
    case "model":
      return cls(td, el("div", {}, el("span", { class: "val-strong", text: m.label }),
        el("div", { class: "mono", text: m.tuning })));
    case "provenance":
      return cls(td, el("span", { class: "pill ok", text: "verified" }),
        el("div", { class: "mono", style: "margin-top:4px", text: "sha256 " + m.provenance.weights_sha256.slice(0, 12) + "…" }));
    case "license":
      return cls(td, el("span", { class: "pill " + (m.license.osi_open_source ? "ok" : "bad"), text: m.license.id }),
        el("div", { class: "mono", style: "margin-top:4px", text: m.license.osi_open_source ? "OSI open source" : "not OSI — review" }));
    case "params":
      return cls(td, el("span", { class: "val-strong", text: fmt.params(m.metadata.parameters) }));
    case "dims":
      return cls(td, `${m.metadata.layers} layers × ${m.metadata.hidden_size} wide`,
        el("div", { class: "mono", text: `${m.metadata.attention_heads} q-heads / ${m.metadata.kv_heads} kv · head ${m.metadata.head_dim}` }));
    case "answer":
      td.className += " answer"; return cls(td, el("span", { text: m.answer || "(empty)" }));
    case "stop_reason":
      return cls(td, m.stop_reason === "eos" ? "chose to stop (eos)" : "hit token cap");
    case "tokenization":
      return cls(td, el("span", { class: "val-strong", text: m.prompt_tokens }), " prompt tokens",
        el("div", { class: "mono", text: m.generated_tokens + " generated" }));
    case "first_token_confidence":
      return cls(td, el("span", { class: "val-strong", text: pct(me.first_token_confidence) }));
    case "mean_entropy":
      return cls(td, el("span", { class: "val-strong", text: fmt.num(me.mean_entropy, 2) }), " nats");
    case "residual_growth":
      return cls(td, el("span", { class: "val-strong", text: me.residual_growth_ratio ? "×" + me.residual_growth_ratio : "—" }),
        sparkline(m.layer_norm_profile));
    case "attention_entropy":
      return cls(td, el("span", { class: "val-strong", text: fmt.num(me.attention_entropy_layer0, 2) }), " nats");
    case "timing":
      return cls(td, `prefill ${fmt.num(me.prefill_s, 2)}s`,
        el("div", { class: "mono", text: `decode ${fmt.num(me.decode_s, 2)}s · ${fmt.num(me.tokens_per_s, 1)} tok/s` }));
  }
  return cls(td, "—");
}

// A reference (not-run) column: verified specs from the catalog; measured rows say so.
function refCell(td, rowId, m) {
  const md = m.metadata;
  switch (rowId) {
    case "model":
      return cls(td, el("div", {}, el("span", { class: "val-strong", text: m.label }),
        el("div", { class: "mono", text: m.tuning })));
    case "provenance":
      return cls(td, el("span", { class: "pill metadata", text: "not run" }),
        el("div", { class: "mono", style: "margin-top:4px", text: "rev " + (m.revision || "").slice(0, 12) }));
    case "license":
      return cls(td, el("span", { class: "pill " + (m.license.osi_open_source ? "ok" : "bad"), text: m.license.id }),
        el("div", { class: "mono", style: "margin-top:4px", text: m.license.osi_open_source ? "OSI open source" : "not OSI — review" }));
    case "params":
      return cls(td, el("span", { class: "val-strong", text: fmt.params(md.parameters) + (md.mixture_of_experts ? " (MoE)" : "") }));
    case "dims":
      return (md.layers == null)
        ? cls(td, "custom / MoE", el("div", { class: "mono", text: "dims not exposed" }))
        : cls(td, `${md.layers} layers × ${md.hidden_size} wide`,
          el("div", { class: "mono", text: `${md.attention_heads} q-heads / ${fmt.or(md.kv_heads)} kv · head ${fmt.or(md.head_dim)}` }));
    case "answer":
      td.className += " answer"; return cls(td, el("span", { class: "muted-cell", text: "— not run here" }));
  }
  return cls(td, el("span", { class: "muted-cell", text: "—" }));
}

function cls(td, ...kids) { for (const k of kids) td.append(k.nodeType ? k : document.createTextNode(k)); return td; }
function pct(x) { return (x === null || x === undefined) ? "—" : (100 * x).toFixed(1) + "%"; }

function sparkline(profile) {
  if (!profile || !profile.length) return document.createTextNode("");
  const vals = profile.map((v) => (typeof v === "number" ? v : 0));
  const w = 120, h = 24, n = vals.length, mn = Math.min(...vals), mx = Math.max(...vals) || 1;
  const pts = vals.map((v, i) => {
    const x = (i / (n - 1)) * (w - 2) + 1;
    const y = h - 1 - ((v - mn) / (mx - mn || 1)) * (h - 2);
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "sparkline"); svg.setAttribute("width", w); svg.setAttribute("height", h);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "residual-stream norm by layer");
  const pl = document.createElementNS(svg.namespaceURI, "polyline");
  pl.setAttribute("points", pts); pl.setAttribute("fill", "none");
  pl.setAttribute("stroke", "var(--measured)"); pl.setAttribute("stroke-width", "1.5");
  svg.append(pl);
  return svg;
}

// runCols: [{data: summary, color, label}] for the runnable models only.
function renderViz(runCols) {
  const panel = el("div", { class: "panel viz", style: "margin-top:16px" });
  panel.append(el("h2", {}, "Visualizations ", el("span", { class: "pill measured", text: "measured" }),
    el("span", { class: "viz-scope", text: runCols.length === 1
      ? " — the one model that ran" : ` — the ${runCols.length} models that ran` })));
  const have = runCols.every((c) => c.data.decode_series && c.data.layer_series);
  if (!have) { panel.append(el("p", { class: "lede", text: "Run a fresh comparison to see the charts (this recorded result predates them)." })); return panel; }

  const numFmt = (v) => (Math.abs(v) >= 100 ? Math.round(v) : (+v).toFixed(1));

  // 1 — decoding confidence over the generated tokens
  const maxGen = Math.max(...runCols.map((c) => c.data.decode_series.length));
  panel.append(vizCard("viz_confidence", "Decoding confidence across the answer",
    "Probability the model gave the token it actually chose, at each step. Dips = hesitation.",
    lineChart({
      height: 220, yMin: 0, yMax: 1, yTicks: 4, yFmt: (v) => Math.round(v * 100) + "%", xName: "token",
      xStartLabel: "first generated token", xEndLabel: "token " + maxGen,
      series: runCols.map((c) => ({ label: c.data.label, color: c.color, points: c.data.decode_series.map((t) => t.p) })),
    })));

  // 2 — residual-stream growth by relative depth
  panel.append(vizCard("viz_residual_depth", "Residual-stream growth by depth",
    "Length of the internal vector at each layer (aligned by relative depth so models of different heights overlay).",
    lineChart({
      height: 220, relativeX: true, yTicks: 4, yFmt: numFmt, xName: "layer",
      xStartLabel: "input (layer 0)", xEndLabel: "final layer",
      series: runCols.map((c) => ({ label: c.data.label, color: c.color, points: c.data.layer_series.h_norm })),
    })));

  // 3 — where each layer does its work (per-model small multiples)
  const grid = el("div", { class: "viz-grid" });
  runCols.forEach((c) => {
    grid.append(el("div", { class: "viz-mini" }, el("div", { class: "viz-mini-title" }, c.data.label),
      lineChart({
        height: 180, relativeX: true, yTicks: 3, yFmt: numFmt, area: true, xName: "layer",
        xStartLabel: "layer 0", xEndLabel: "last",
        series: [
          { label: "attention write", color: c.color, points: c.data.layer_series.attn_norm },
          { label: "MLP write", color: c.color, points: c.data.layer_series.mlp_norm, dashed: true },
        ],
      })));
  });
  panel.append(vizCard("viz_layer_dynamics", "Where each layer does its work",
    "Per model: how much attention (solid) and the MLP (dashed) write into the vector, layer by layer.", grid));

  // 4 — per-token confidence heatstrips
  const strips = el("div", {});
  runCols.forEach((c) => {
    strips.append(el("div", { class: "strip-row" },
      el("div", { class: "strip-label" }, el("span", { class: "swatch", style: `background:${c.color}` }), c.data.label),
      tokenHeatstrip(c.data.decode_series, c.color)));
  });
  panel.append(vizCard("token_heatstrip", "Per-token confidence",
    "Each generated token shaded by how sure the model was. Hover any token for its exact probability.", strips));

  // 5 — layer x token heatmaps (ported from the single-model explorer), one per model
  if (runCols.every((c) => c.data.token_layer)) {
    const sel = el("select", { "aria-label": "Heatmap metric" });
    HEAT_METRICS.forEach(([k, name]) => sel.append(el("option", { value: k, text: name })));
    const maps = el("div", {});
    const draw = () => {
      const name = HEAT_METRICS.find(([k]) => k === sel.value)[1];
      maps.replaceChildren(...runCols.map((c) => el("div", { class: "heat-row" },
        el("div", { class: "strip-label" }, el("span", { class: "swatch", style: `background:${c.color}` }),
          `${c.data.label} · ${c.data.token_layer.layers} layers`),
        heatmap(c.data.token_layer, sel.value, c.color, name))));
    };
    sel.addEventListener("change", draw);
    draw();
    panel.append(vizCard("layer_token_heatmap", "Layer × token heatmap",
      "Every measured token (across) at every layer (down). Each model uses its own colour scale (range shown below it). Hover a cell for the exact value.",
      el("div", {}, el("div", { class: "heat-controls" }, el("span", { text: "Show:" }), sel), maps)));
  }

  return panel;
}

function vizCard(metricId, title, sub, body) {
  return el("div", { class: "viz-card" },
    el("h3", { class: "viz-h" }, title, infoButton(metricId, title)),
    el("p", { class: "viz-sub", text: sub }), body);
}

function summaryStrip(res, refCount) {
  const p = res.parity, a = res.alignment;
  const strip = el("div", { class: "panel", style: "margin-top:16px" });
  strip.append(el("h2", {}, "Cross-model checks ",
    refCount ? el("span", { class: "viz-scope", text: "— runnable models only" }) : null));
  const grid = el("div", { class: "kv" });
  grid.append(
    el("dt", {}, "Same input?", infoButton("prompt_parity", "Same input")),
    el("dd", { html: `settings identical: <strong>${p.settings_equal ? "yes" : "no"}</strong> · rendered prompt identical: <strong>${p.rendered_prompt_equal ? "yes" : "no"}</strong> · tokenizers produced identical ids: <strong>${p.tokenizers_match ? "yes" : "no"}</strong>` }),
    el("dt", {}, "Token alignment", infoButton("token_alignment", "Token alignment")),
    el("dd", { text: a.comparable ? `agreed on first ${a.shared_generated} generated token(s); ` + a.note : a.note }),
    el("dt", {}, "Peak memory", infoButton("memory", "Peak memory")),
    el("dd", { text: fmt.bytes(res.memory.peak_rss_bytes) + " (models loaded one at a time, then released)" }),
  );
  strip.append(grid);
  return strip;
}

// Only auto-boot on the real compare page (guarded so the renderer can be imported alone).
if (document.getElementById("selectors")) {
  boot().catch((e) => { $("progress").textContent = "Failed to start: " + (e.message || e); });
}
