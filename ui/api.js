// Shared backend access + privacy gate for the public three-model explorer.
//
// PRIVACY: the live prompt comparison is enabled ONLY when this page is served from the
// local loopback backend (127.0.0.1 / localhost). When the same files are served from a
// public host (e.g. GitHub Pages), BACKEND is null: no prompt is ever sent anywhere, the
// run button is disabled, and the catalog/detail pages fall back to the static JSON that
// ships with the site. A prompt can only ever reach your own machine's backend.

export const LOOPBACK = ["127.0.0.1", "localhost"].includes(location.hostname);
export const BACKEND = LOOPBACK ? location.origin : null;   // null => static (no backend)

// Load JSON from the backend API when present, else from a static file beside the page.
export async function loadJSON(apiPath, staticPath) {
  if (BACKEND) {
    try {
      const r = await fetch(BACKEND + apiPath, { cache: "no-store" });
      if (r.ok) return await r.json();
    } catch (e) { /* fall through to static */ }
  }
  const r = await fetch(staticPath, { cache: "no-store" });
  if (!r.ok) throw new Error("could not load " + staticPath);
  return await r.json();
}

export async function postJSON(apiPath, body) {
  if (!BACKEND) throw new Error("the local backend is required for this action");
  const r = await fetch(BACKEND + apiPath, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || ("request failed: " + r.status));
  return data;
}

export async function health() {
  if (!BACKEND) return null;
  try {
    const r = await fetch(BACKEND + "/api/health", { cache: "no-store" });
    if (r.ok) return await r.json();
  } catch (e) {}
  return null;
}

export const fmt = {
  bytes(n) {
    if (!n && n !== 0) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0, v = n;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (i >= 2 ? v.toFixed(2) : Math.round(v)) + " " + u[i];
  },
  params(n) {
    if (!n && n !== 0) return "—";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(0) + "M";
    return String(n);
  },
  num(x, d = 3) { return (x === null || x === undefined) ? "—" : (+x).toFixed(d); },
  loc(x) { return (x === null || x === undefined) ? "—" : (+x).toLocaleString(); },
  or(x, dash = "—") { return (x === null || x === undefined || x === "") ? dash : x; },
};

export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    // CSP forbids style attributes from markup/setAttribute; CSSOM assignment is allowed.
    else if (k === "style") n.style.cssText = v;
    else if (k === "text") n.textContent = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}
