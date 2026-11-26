// Catalog grid. Fully static: reads /api/catalog or ./catalog.json.
import { loadJSON, BACKEND, fmt, el } from "./api.js";
import { releaseTimeline } from "./viz.js";

function benchText(b) {
  if (!b || b.value === null || b.value === undefined) return null;
  return `${b.label} ${b.value}`;
}

function renderTimeline(cat) {
  document.getElementById("timeline").append(releaseTimeline(cat.models.map((m) => ({
    id: m.id, label: m.label, released: m.released, params: m.metadata.parameters,
    runnable: m.runnable, moe: m.metadata.mixture_of_experts }))));
}

function renderBench(cat) {
  const t = document.getElementById("bench");
  const hr = el("tr");
  ["Model", "Released", "Params", "Benchmark (publisher)", "Score", "Protocol / source"].forEach((h) => hr.append(el("th", { text: h })));
  t.append(el("thead", {}, hr));
  const tb = el("tbody");
  [...cat.models].sort((a, b) => (a.released || "").localeCompare(b.released || "")).forEach((m) => {
    const b = m.benchmark || {};
    const has = b.value !== null && b.value !== undefined;
    tb.append(el("tr", {},
      el("td", {}, el("a", { href: "model.html?id=" + encodeURIComponent(m.id), text: m.label }),
        m.runnable ? el("span", { class: "pill ok", style: "margin-left:6px", text: "runs" }) : null),
      el("td", { class: "mono", text: m.released || "—" }),
      el("td", { text: fmt.params(m.metadata.parameters) + (m.metadata.mixture_of_experts ? " (MoE)" : "") }),
      el("td", { text: has ? b.label : "—" }),
      el("td", { class: "val-strong", text: has ? String(b.value) : "see card" }),
      el("td", {}, el("span", { class: "mono", text: b.protocol || "" }), " · ",
        el("a", { href: m.source_links.model_card, rel: "noreferrer", text: "card" }))));
  });
  t.append(tb);
}

async function boot() {
  const cat = await loadJSON("/api/catalog", "./catalog.json");
  const bar = document.getElementById("statusbar");
  bar.append(
    el("span", {}, el("span", { class: "dot ok" }), `${cat.count} models · ${cat.runnable_count} runnable with this tool`),
    el("span", { class: "pill metadata", text: "metadata" }),
    el("span", { text: "Architecture, dimensions, format and licence below are metadata read from official model cards — not measured here." }),
  );
  document.getElementById("intro").textContent = cat.verified_from;
  renderTimeline(cat);
  renderBench(cat);
  const grid = document.getElementById("grid");
  for (const m of cat.models) grid.append(card(m));
}

function card(m) {
  const md = m.metadata;
  const c = el("div", { class: "card" + (m.runnable ? " runnable" : "") });
  c.append(
    el("h3", {}, el("a", { href: "model.html?id=" + encodeURIComponent(m.id), text: m.label })),
    el("div", { class: "repo", text: m.repo }),
    badges(m),
    facts(m, md),
    el("div", { class: "reason", text: m.status_reason }),
  );
  return c;
}

function badges(m) {
  const md = m.metadata;
  const b = el("div", { class: "badges" });
  b.append(el("span", { class: "pill " + (m.runnable ? "ok" : "metadata"),
    text: m.runnable ? "runnable · slot " + m.slot.toUpperCase() : "catalog only" }));
  const isBase = m.tuning.startsWith("base");
  b.append(el("span", { class: "pill " + (isBase ? "metadata" : "measured"), text: isBase ? "base" : "instruct" }));
  b.append(el("span", { class: "pill " + (m.license.osi_open_source ? "ok" : "bad"), text: m.license.id }));
  if (!m.license.osi_open_source) b.append(el("span", { class: "pill bad", text: "not OSI" }));
  if (md.mixture_of_experts) b.append(el("span", { class: "pill derived", text: "MoE" }));
  if (md.sharded) b.append(el("span", { class: "pill metadata", text: "sharded" }));
  b.append(el("span", { class: "pill metadata", text: md.weight_format }));
  return b;
}

function facts(m, md) {
  const dl = el("dl", { class: "facts" });
  const add = (k, v) => { dl.append(el("dt", { text: k }), el("dd", { text: v })); };
  const pair = (a, b) => (a == null && b == null) ? "—" : `${fmt.or(a)} × ${fmt.or(b)}`;
  add("Parameters", fmt.params(md.parameters) + (md.mixture_of_experts ? " (MoE total)" : ""));
  add("Architecture", m.architecture + " (" + m.model_type + ")");
  add("Layers × width", pair(md.layers, md.hidden_size));
  add("Heads (q/kv)", (md.attention_heads == null)
    ? "—" : `${md.attention_heads} / ${fmt.or(md.kv_heads)} · head ${fmt.or(md.head_dim)}`);
  add("Context", md.context_length == null ? "—" : fmt.loc(md.context_length) + " tokens");
  add("Weights", `${fmt.bytes(md.weight_bytes)} · ${fmt.or(md.stored_dtype)}`);
  add("Released", m.released || "—");
  add("Benchmark", benchText(m.benchmark) ? benchText(m.benchmark) + " (publisher)" : "see model card");
  add("Revision", (m.revision || "").slice(0, 12));
  return dl;
}

boot().catch((e) => { document.getElementById("grid").textContent = "Failed to load catalog: " + (e.message || e); });
