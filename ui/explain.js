// Accessible explanation popover for the public explorer.
//
// Every explained item gets a small "i" trigger button. The explanation appears on
// mouse hover AND on click/touch, and is fully keyboard operable:
//   * the trigger is a real <button> (Tab-focusable);
//   * focus or hover opens the popover; Enter/Space toggles a "pinned" open state;
//   * Escape closes and returns focus to the trigger;
//   * clicking elsewhere closes a pinned popover.
// One shared popover element is reused and positioned next to the active trigger.
// `metrics` is the /api/metrics (or explanations.json) payload.

import { el } from "./api.js";

let META = null;           // { kind_labels, metrics, comparison_rows }
let POP = null;            // the shared popover element
let pinnedTrigger = null;  // currently click-pinned trigger, or null

export function initExplain(metricsPayload) {
  META = metricsPayload;
  if (!POP) {
    POP = el("div", { id: "popover", role: "dialog", "aria-live": "polite", "aria-modal": "false" });
    document.body.append(POP);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(true); });
    document.addEventListener("click", (e) => {
      if (pinnedTrigger && !POP.contains(e.target) && e.target !== pinnedTrigger) close(false);
    });
    window.addEventListener("scroll", () => { if (!pinnedTrigger) hide(); }, true);
  }
}

export function metricMeta(id) { return META && META.metrics[id]; }

function render(m) {
  const kindLabel = (META.kind_labels && META.kind_labels[m.kind]) || m.kind;
  POP.replaceChildren(
    el("span", { class: "k pill " + m.kind, text: m.kind }),
    el("h4", { text: m.title }),
    el("p", { text: m.plain }),
    el("p", { html: "<strong>How to read:</strong> " + escapeHtml(m.read) }),
    el("p", { class: "cannot", html: "<strong>What it can’t tell you:</strong> " + escapeHtml(m.cannot) }),
    el("p", { class: "closehint", text: kindLabel }),
  );
}

function place(trigger) {
  const r = trigger.getBoundingClientRect();
  POP.style.display = "block"; POP.dataset.open = "1";
  const pw = POP.offsetWidth, ph = POP.offsetHeight;
  let left = r.left + window.scrollX;
  let top = r.bottom + window.scrollY + 8;
  if (left + pw > window.scrollX + document.documentElement.clientWidth - 10)
    left = window.scrollX + document.documentElement.clientWidth - pw - 10;
  if (r.bottom + ph + 8 > document.documentElement.clientHeight) top = r.top + window.scrollY - ph - 8;
  POP.style.left = Math.max(8, left) + "px";
  POP.style.top = Math.max(8, top) + "px";
}

function open(trigger, metricId, pin) {
  const m = metricMeta(metricId);
  if (!m) return;
  render(m); place(trigger);
  trigger.setAttribute("aria-expanded", "true");
  if (pin) { pinnedTrigger = trigger; POP.setAttribute("tabindex", "-1"); }
}

function hide() { POP.dataset.open = "0"; POP.style.display = "none"; }

function close(returnFocus) {
  hide();
  if (pinnedTrigger) {
    pinnedTrigger.setAttribute("aria-expanded", "false");
    if (returnFocus) pinnedTrigger.focus();
    pinnedTrigger = null;
  }
}

// Build an "i" trigger button for a metric id. Attach next to any label.
export function infoButton(metricId, labelText) {
  const m = metricMeta(metricId);
  const title = m ? m.title : metricId;
  const btn = el("button", {
    type: "button", class: "info", "aria-expanded": "false",
    "aria-label": `Explain: ${labelText || title}`, text: "i",
  });
  btn.addEventListener("mouseenter", () => { if (!pinnedTrigger) open(btn, metricId, false); });
  btn.addEventListener("mouseleave", () => { if (!pinnedTrigger) hide(); });
  btn.addEventListener("focus", () => { if (!pinnedTrigger) open(btn, metricId, false); });
  btn.addEventListener("blur", () => { if (!pinnedTrigger) hide(); });
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (pinnedTrigger === btn) close(true);
    else { if (pinnedTrigger) close(false); open(btn, metricId, true); }
  });
  return btn;
}

// A row label with an attached info trigger.
export function labelWithInfo(metricId, text) {
  return el("span", { class: "rowlabel" }, text, infoButton(metricId, text));
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
