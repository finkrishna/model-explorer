// Per-model static architecture / weights page.
// Reads ?id=<model-id>; works with no backend (uses ./catalog.json). Every number here is
// METADATA from the official model card/config, labelled as such; nothing is measured on
// this page. The live comparison screen is where measured values appear.
import { loadJSON, fmt, el } from "./api.js";
import { initExplain, infoButton } from "./explain.js";

async function boot() {
  const id = new URLSearchParams(location.search).get("id");
  const root = document.getElementById("detail");
  const [cat, metrics] = await Promise.all([
    loadJSON("/api/catalog", "./catalog.json"),
    loadJSON("/api/metrics", "./explanations.json").catch(() => ({ metrics: {}, kind_labels: {} })),
  ]);
  initExplain(metrics);
  const m = cat.models.find((x) => x.id === id);
  if (!m) { root.append(el("div", { class: "notice bad", text: "Unknown model id. " }),
    el("p", {}, el("a", { href: "catalog.html", text: "← Back to catalog" }))); return; }
  document.title = m.label + " — open-weight explorer";
  render(root, m);
}

function render(root, m) {
  const md = m.metadata;
  root.append(el("p", {}, el("a", { href: "catalog.html", text: "← Catalog" })));
  root.append(el("div", { class: "detail-head" },
    el("h2", { style: "font-size:22px;margin:6px 0", text: m.label }),
    el("span", { class: "pill " + (m.runnable ? "ok" : "metadata"),
      text: m.runnable ? "runnable · slot " + m.slot.toUpperCase() : "catalog only" }),
    el("span", { class: "pill " + (m.tuning.startsWith("base") ? "metadata" : "measured"),
      text: m.tuning.startsWith("base") ? "base model" : "instruction-tuned" }),
    el("span", { class: "pill " + (m.license.osi_open_source ? "ok" : "bad"), text: m.license.id }),
  ));
  root.append(el("div", { class: "mono", style: "color:var(--muted);margin-bottom:6px", text: m.repo }));

  // status / why runnable-or-not
  root.append(el("div", { class: "panel" },
    el("h2", {}, m.runnable ? "Runs locally" : "Reference only — why it is not runnable here",
      infoButton(m.runnable ? "provenance" : "model", "status")),
    el("p", { class: "lede", style: "margin:0", text: m.status_reason })));

  // metadata-vs-measured banner
  root.append(el("div", { class: "notice", html:
    "Everything on this page is <strong>metadata</strong> copied from the official model card and "
    + "<code>config.json</code> at the pinned revision — it is not measured. "
    + "Load this model in the <a href='index.html'>Compare</a> screen to see <strong>measured</strong> internals." }));

  // architecture facts
  const facts = el("div", { class: "panel" }, el("h2", {}, "Architecture & weights ",
    infoButton("dims", "architecture")));
  const kv = el("dl", { class: "kv" });
  const add = (k, v, metric) => {
    const dd = el("dd", {}); if (v && v.nodeType) dd.append(v); else dd.textContent = v;
    const dt = el("dt", {}); dt.append(document.createTextNode(k + " ")); if (metric) dt.append(infoButton(metric, k));
    kv.append(dt, dd);
  };
  add("Parameters", fmt.params(md.parameters) + `  (${fmt.loc(md.parameters)})`
    + (md.mixture_of_experts ? "  — mixture-of-experts total" : ""), "params");
  add("Architecture class", m.architecture + "  · type " + m.model_type);
  add("Transformer layers", fmt.or(md.layers), "dims");
  add("Hidden size (width)", fmt.or(md.hidden_size));
  add("Attention heads", (md.attention_heads == null) ? "—"
    : `${md.attention_heads} query · ${fmt.or(md.kv_heads)} key/value · head dim ${fmt.or(md.head_dim)}`);
  add("MLP intermediate", fmt.or(md.mlp_intermediate));
  add("Vocabulary", fmt.loc(md.vocab_size));
  add("Context length", md.context_length == null ? "—" : fmt.loc(md.context_length) + " tokens");
  add("Tied embeddings", md.tied_embeddings == null ? "—" : (md.tied_embeddings ? "yes (input = output matrix)" : "no"));
  add("RoPE theta", fmt.or(md.rope_theta));
  add("Weight format", md.weight_format + (md.sharded ? " (sharded across files)" : " (single file)"));
  add("Weight size / dtype", `${fmt.bytes(md.weight_bytes)} · stored ${fmt.or(md.stored_dtype)}`);
  add("Released on Hugging Face", fmt.or(m.released));
  const b = m.benchmark || {};
  add("Benchmark (publisher-reported)", (b.value === null || b.value === undefined)
    ? "no clean figure — see model card"
    : `${b.label} ${b.value} · ${b.protocol} — self-reported; not comparable across models`, "benchmark");
  facts.append(kv);
  root.append(facts);

  // diagram (only when the model exposes standard transformer dims)
  if (md.hidden_size && md.mlp_intermediate && md.layers) {
    root.append(el("div", { class: "panel" }, el("h2", {}, "Forward-pass shape"),
      el("p", { class: "lede", style: "margin-top:0",
        text: "One token flows bottom-to-top through every layer. Bar width is proportional to √dimension, so wider = larger vector. Metadata-derived sketch, not a measurement." }),
      diagram(md)));
  } else {
    root.append(el("div", { class: "panel" }, el("h2", {}, "Forward-pass shape"),
      el("p", { class: "lede", style: "margin:0",
        text: "This model uses a custom / mixture-of-experts architecture that does not expose the standard transformer dimensions in its public config, so a faithful shape sketch isn’t drawn here. See the model card for its design." })));
  }

  // provenance + license
  const prov = el("div", { class: "panel" }, el("h2", {}, "Provenance & licence ", infoButton("provenance", "provenance")));
  const pkv = el("dl", { class: "kv" });
  const padd = (k, v) => { const dd = el("dd", {}); if (v && v.nodeType) dd.append(v); else dd.textContent = v; pkv.append(el("dt", { text: k }), dd); };
  padd("Pinned revision", el("span", { class: "mono", text: m.revision || "—" }));
  if (m.provenance && m.provenance.verified) {
    padd("Weights SHA-256", el("span", { class: "mono", text: m.provenance.weights_sha256 }));
    padd("Verified", el("span", { class: "pill ok", text: "local bytes match official metadata" }));
  } else {
    padd("Verification", "Metadata pinned to the revision above; weights are not loaded or hashed on this page.");
  }
  padd("Licence", el("span", {}, el("span", { class: "pill " + (m.license.osi_open_source ? "ok" : "bad"), text: m.license.id }),
    " " + m.license.note));
  padd("Sources", el("span", {}, el("a", { href: m.source_links.model_card, text: "model card", rel: "noreferrer" }),
    " · ", el("a", { href: m.source_links.config, text: "config.json", rel: "noreferrer" }),
    " · ", el("a", { href: m.license.source, text: "LICENSE", rel: "noreferrer" })));
  prov.append(pkv);
  root.append(prov);

  if (!m.runnable) root.append(exampleBlock(m));
  root.append(el("p", { class: "footnote", text: "Links open the official Hugging Face pages in a new context. This page sends nothing anywhere." }));
}

// Simple, honest forward-pass diagram from metadata dims.
function diagram(md) {
  const W = 680, rowH = 30, gap = 8;
  const steps = [
    ["Token embedding", md.hidden_size, "embed"],
    [`× ${md.layers}  {  Attention: ${md.attention_heads}q / ${md.kv_heads}kv heads, head ${md.head_dim}`, md.hidden_size, "attn"],
    [`        MLP: gate+up → ${md.mlp_intermediate} → down  }`, md.mlp_intermediate, "mlp"],
    ["Final RMSNorm", md.hidden_size, "norm"],
    [`Unembedding → ${md.vocab_size.toLocaleString()} logits`, md.hidden_size, "out"],
  ];
  const maxDim = Math.max(md.hidden_size, md.mlp_intermediate);
  const scale = (d) => 60 + (Math.sqrt(d) / Math.sqrt(maxDim)) * (W - 240);
  const H = steps.length * (rowH + gap) + 20;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "diagram"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label",
    `Forward pass: ${md.layers} layers, hidden ${md.hidden_size}, MLP ${md.mlp_intermediate}, vocab ${md.vocab_size}`);
  const colors = { embed: "var(--metadata)", attn: "var(--accent)", mlp: "var(--derived)", norm: "var(--muted)", out: "var(--measured)" };
  steps.forEach(([label, dim, kind], i) => {
    const y = H - 10 - (i + 1) * (rowH + gap);
    const w = scale(dim);
    const rect = document.createElementNS(svg.namespaceURI, "rect");
    rect.setAttribute("x", 12); rect.setAttribute("y", y); rect.setAttribute("width", w);
    rect.setAttribute("height", rowH); rect.setAttribute("rx", 5);
    rect.setAttribute("fill", colors[kind]); rect.setAttribute("opacity", "0.16");
    rect.setAttribute("stroke", colors[kind]); rect.setAttribute("stroke-width", "1.2");
    svg.append(rect);
    const t = document.createElementNS(svg.namespaceURI, "text");
    t.setAttribute("x", 20); t.setAttribute("y", y + rowH / 2 + 4);
    t.setAttribute("font-size", "12"); t.setAttribute("font-family", "var(--sans)");
    t.setAttribute("fill", "var(--ink)"); t.textContent = label;
    svg.append(t);
    const dt = document.createElementNS(svg.namespaceURI, "text");
    dt.setAttribute("x", w + 18); dt.setAttribute("y", y + rowH / 2 + 4);
    dt.setAttribute("font-size", "11"); dt.setAttribute("fill", "var(--muted)");
    dt.setAttribute("font-family", "var(--mono)"); dt.textContent = "dim " + dim;
    svg.append(dt);
  });
  return svg;
}

function exampleBlock(m) {
  const reasons = {
    qwen3: "Qwen3 adds per-head Q/K RMSNorm and an explicit head_dim, so a Qwen2-only forward pass would read the weights incorrectly.",
    llama: "Llama-family models (SmolLM2) have no QKV bias and a different tokenizer/chat template, so the Qwen2 engine cannot instrument them faithfully.",
  };
  const ex = el("div", { class: "panel" }, el("h2", {}, "If this model could run here"));
  const vocabPhrase = m.metadata.vocab_size == null
    ? "its own tokenizer" : `its own ${fmt.loc(m.metadata.vocab_size)}-token tokenizer`;
  const layers = m.metadata.layers == null ? "all its" : m.metadata.layers;
  ex.append(el("p", { class: "lede",
    text: `Example: a prompt like “Explain gravity to a child.” would be tokenized by ${vocabPhrase}, `
      + `flow through ${layers} layers, and produce an answer plus the same measured internals you see for the `
      + `runnable models — but only after a verified adapter for its architecture is added and its checkpoint `
      + `is present and provenance-checked.` }));
  const note = reasons[m.model_type] || (m.metadata.mixture_of_experts
    ? "This is a mixture-of-experts model: only a few expert sub-networks fire per token, which the dense Qwen2 engine does not implement." : null);
  if (note) ex.append(el("p", { class: "footnote", text: note }));
  return ex;
}

boot().catch((e) => { document.getElementById("detail").textContent = "Failed to load: " + (e.message || e); });
