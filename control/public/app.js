// fv-control dashboard: a dependency-free SPA over the JSON API (/api).
// Every value from the API goes into the DOM through textContent.
"use strict";

// ---------------------------------------------------------------- basics
const $ = (s, r = document) => r.querySelector(s);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") el.style.cssText = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}
const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}
const fmt$ = (v, d = 2) => (v === null || v === undefined || isNaN(v) ? "–" : `$${Number(v).toFixed(d)}`);
const fmtN = (v, d = 0) => (v === null || v === undefined || isNaN(v) ? "–" : Number(v).toFixed(d));
const fmtPct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v)}%`);
const ago = (ms) => {
  if (!ms) return "–";
  const s = Math.round((Date.now() - ms) / 1000);
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
};
const until = (ms) => {
  if (!ms) return "–";
  const m = Math.round((ms - Date.now()) / 60000);
  return m < 0 ? `${-m} min ago` : m < 120 ? `in ${m} min` : `in ${(m / 60).toFixed(1)} h`;
};
const dt = (ms) => (ms ? new Date(ms).toISOString().replace("T", " ").slice(0, 16) + "Z" : "–");
const dur = (s) => (s === null || s === undefined || isNaN(s) ? "–" : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("on");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("on"), 3500);
}
const SERIES = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => `var(--series-${i})`);

// ---------------------------------------------------------------- API
const state = { csrf: null, me: null, timers: [] };
async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {}, credentials: "same-origin" };
  if (opts.body !== undefined) {
    init.headers["content-type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  if (init.method !== "GET" && state.csrf) init.headers["x-csrf-token"] = state.csrf;
  const r = await fetch(path, init);
  const j = await r.json().catch(() => ({}));
  if (r.status === 401 && !path.startsWith("/api/auth")) {
    renderLogin();
    throw new Error("login required");
  }
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
async function act(label, fn) {
  try {
    const r = await fn();
    toast(`${label}: ok`);
    return r;
  } catch (e) {
    toast(`${label}: ${e.message}`);
    throw e;
  }
}

// ---------------------------------------------------------------- theme
function applyTheme(t) {
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
}
try { applyTheme(localStorage.getItem("fvc-theme")); } catch { /* storage off */ }

// ---------------------------------------------------------------- charts
// Line chart with a crosshair and one tooltip listing every series.
function lineChart({ series, height = 200, yFmt = (v) => fmtN(v), yMax, yMin = 0, title }) {
  const box = h("div", { class: "chart", role: "img", "aria-label": title || "chart" });
  const W = 640, H = height, L = 44, R = 8, T = 8, B = 22;
  const all = series.flatMap((s) => s.points);
  if (!all.length) {
    box.append(h("div", { class: "muted small" }, "No data yet: the collector samples every minute."));
    return box;
  }
  const xs = all.map((p) => p[0]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 60000);
  const ys = all.map((p) => p[1]).filter((v) => v !== null);
  const top = yMax ?? Math.max(1e-9, ...ys) * 1.1;
  const X = (t) => L + ((t - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v) => T + (1 - (v - yMin) / (top - yMin)) * (H - T - B);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });
  const ax = s("g", { class: "axis" });
  for (let i = 0; i <= 4; i++) {
    const v = yMin + ((top - yMin) * i) / 4;
    ax.append(s("line", { class: "gridline", x1: L, x2: W - R, y1: Y(v), y2: Y(v) }));
    const tx = s("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" });
    tx.textContent = yFmt(v);
    ax.append(tx);
  }
  for (let i = 0; i <= 4; i++) {
    const t = x0 + ((x1 - x0) * i) / 4;
    const tx = s("text", { x: X(t), y: H - 6, "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle" });
    tx.textContent = new Date(t).toISOString().slice(11, 16);
    ax.append(tx);
  }
  svg.append(ax);
  series.forEach((se, i) => {
    let d = "";
    let pen = false;
    for (const [t, v] of se.points) {
      if (v === null) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${X(t).toFixed(1)},${Y(v).toFixed(1)}`;
      pen = true;
    }
    svg.append(s("path", { d, fill: "none", stroke: se.color || SERIES[i % 8], "stroke-width": 2, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }));
    const real = se.points.filter((p) => p[1] !== null);
    if (real.length <= 3) for (const [t, v] of real) svg.append(s("circle", { cx: X(t), cy: Y(v), r: 4, fill: se.color || SERIES[i % 8], stroke: "var(--surface-1)", "stroke-width": 2 }));
  });
  const hair = s("line", { class: "hair", y1: T, y2: H - B, visibility: "hidden", "vector-effect": "non-scaling-stroke" });
  svg.append(hair);
  const times = [...new Set(xs)].sort((a, b) => a - b);
  const tip = $("#tip");
  svg.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const tx = x0 + ((((ev.clientX - r.left) / r.width) * W - L) / (W - L - R)) * (x1 - x0);
    let best = times[0];
    for (const t of times) if (Math.abs(t - tx) < Math.abs(best - tx)) best = t;
    hair.setAttribute("x1", X(best));
    hair.setAttribute("x2", X(best));
    hair.setAttribute("visibility", "visible");
    tip.replaceChildren(h("div", { class: "t" }, new Date(best).toISOString().replace("T", " ").slice(0, 16) + "Z"));
    series.forEach((se, i) => {
      const p = se.points.reduce((a, b) => (Math.abs(b[0] - best) < Math.abs(a[0] - best) ? b : a), se.points[0] || [0, null]);
      if (p && p[1] !== null && Math.abs(p[0] - best) < (x1 - x0) / 50 + 60000)
        tip.append(h("div", { class: "r" }, h("span", { class: "key", style: `background:${se.color || SERIES[i % 8]}` }), h("b", {}, yFmt(p[1])), h("span", { class: "muted" }, se.name)));
    });
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 290) + "px";
    tip.style.top = ev.clientY + 14 + "px";
  });
  svg.addEventListener("pointerleave", () => {
    hair.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  box.append(svg);
  if (series.length > 1) box.append(h("div", { class: "legend" }, series.map((se, i) => h("span", {}, h("span", { class: "key", style: `background:${se.color || SERIES[i % 8]}` }), se.name))));
  return box;
}
// Stacked columns (e.g. cost per day by owner) with per-segment tooltips.
function stackedBars({ cats, stacks, height = 200, yFmt }) {
  const box = h("div", { class: "chart" });
  if (!cats.length) {
    box.append(h("div", { class: "muted small" }, "No data yet."));
    return box;
  }
  const W = 640, H = height, L = 44, R = 8, T = 8, B = 22;
  const totals = cats.map((_, i) => stacks.reduce((sum, st) => sum + (st.values[i] || 0), 0));
  const top = Math.max(1e-9, ...totals) * 1.1;
  yFmt = yFmt || ((v) => fmt$(v, top < 10 ? 2 : 0));
  const bw = Math.min(56, ((W - L - R) / cats.length) * 0.7);
  const Y = (v) => T + (1 - v / top) * (H - T - B);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });
  const ax = s("g", { class: "axis" });
  for (let i = 0; i <= 4; i++) {
    const v = (top * i) / 4;
    ax.append(s("line", { class: "gridline", x1: L, x2: W - R, y1: Y(v), y2: Y(v) }));
    const tx = s("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" });
    tx.textContent = yFmt(v);
    ax.append(tx);
  }
  svg.append(ax);
  const tip = $("#tip");
  cats.forEach((c, i) => {
    const cx = L + ((i + 0.5) * (W - L - R)) / cats.length;
    let acc = 0;
    stacks.forEach((st, j) => {
      const v = st.values[i] || 0;
      if (v <= 0) return;
      const y0 = Y(acc), y1 = Y(acc + v);
      acc += v;
      const r = s("rect", { x: cx - bw / 2, y: y1, width: bw, height: Math.max(0.5, y0 - y1 - 2), fill: st.color || SERIES[j % 8], rx: 2 });
      r.addEventListener("pointermove", (ev) => {
        tip.replaceChildren(h("div", { class: "t" }, c), h("div", { class: "r" }, h("b", {}, yFmt(v)), h("span", { class: "muted" }, st.name)), h("div", { class: "muted" }, `total ${yFmt(totals[i])}`));
        tip.hidden = false;
        tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 290) + "px";
        tip.style.top = ev.clientY + 14 + "px";
        r.setAttribute("opacity", "0.8");
      });
      r.addEventListener("pointerleave", () => {
        tip.hidden = true;
        r.removeAttribute("opacity");
      });
      svg.append(r);
    });
    const tx = s("text", { x: cx, y: H - 6, "text-anchor": "middle", class: "axis" });
    tx.textContent = c.slice(5);
    tx.setAttribute("fill", "var(--text-muted)");
    tx.setAttribute("font-size", "11");
    svg.append(tx);
  });
  box.append(svg);
  if (stacks.length > 1) box.append(h("div", { class: "legend" }, stacks.map((st, j) => h("span", {}, h("span", { class: "key box", style: `background:${st.color || SERIES[j % 8]}` }), st.name))));
  return box;
}
function table(cols, rows, empty = "Nothing here.") {
  if (!rows.length) return h("div", { class: "muted small" }, empty);
  return h(
    "div",
    { class: "tablewrap" },
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, cols.map((c) => h("th", { class: c.num ? "num" : "" }, c.label)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, cols.map((c) => { const v = c.get(r); return h("td", { class: (c.num ? "num " : "") + (c.wrap ? "wrap" : "") }, v instanceof Node ? v : v ?? "–"); })))),
    ),
  );
}
function badge(text, kind) {
  return h("span", { class: `badge ${kind || ""}` }, text);
}
const healthKind = (hh) => (hh === "ready" ? "good" : hh === "loading" || hh === "unknown" ? "warn" : hh === "down" ? "critical" : "");
const statusKind = (st) => (st === "running" || st === "RUNNING" || st === "done" ? "good" : st === "starting" || st === "stopping" ? "warn" : st === "failed" ? "critical" : "");
function card(title, ...kids) {
  return h("section", { class: "card" }, title ? h("h2", {}, title) : null, ...kids);
}

// ---------------------------------------------------------------- routing
const routes = [
  [/^#?\/?$/, pageDashboard],
  [/^#\/clusters$/, pageClusters],
  [/^#\/cluster\/([^/]+)$/, pageCluster],
  [/^#\/pods$/, pagePods],
  [/^#\/pod\/([^/]+)$/, pagePod],
  [/^#\/env$/, pageEnv],
  [/^#\/logs$/, pageLogs],
  [/^#\/costs$/, pageCosts],
  [/^#\/releases$/, pageReleases],
  [/^#\/settings$/, pageSettings],
];
function clearTimers() {
  const tip = document.getElementById("tip");
  if (tip) tip.hidden = true;
  for (const t of state.timers) clearInterval(t);
  state.timers = [];
  if (state.ws) { try { state.ws.close(); } catch {} state.ws = null; }
}
async function route() {
  clearTimers();
  if (!state.me?.authenticated) return renderLogin();
  const hash = location.hash || "#/";
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("on", a.getAttribute("href") === (hash.split("?")[0].replace(/^#\/(cluster|pod)\/.*/, "#/$1s") || "#/"));
  const main = $("#main");
  for (const [re, fn] of routes) {
    const m = re.exec(hash.split("?")[0]);
    if (m) {
      try {
        await fn(main, ...m.slice(1).map(decodeURIComponent));
      } catch (e) {
        if (e.message !== "login required") main.replaceChildren(card("Error", h("p", {}, e.message)));
      }
      return;
    }
  }
  main.replaceChildren(card("Not found", h("a", { href: "#/" }, "Dashboard")));
}
function renderLogin() {
  $("#logoutBtn").hidden = true;
  $("#nav").hidden = true;
  const m = state.me || {};
  const main = $("#main");
  if (m.mode === "access") {
    main.replaceChildren(h("div", { class: "card login" }, h("h2", {}, "Cloudflare Access"), h("p", {}, "This controller is behind Cloudflare Access. Reload the page to sign in.")));
    return;
  }
  const inp = h("input", { type: "password", id: "pass", autocomplete: "current-password", placeholder: "Owner passphrase", "aria-label": "Owner passphrase" });
  const err = h("div", { class: "small", style: "color:var(--critical-text)" });
  const form = h(
    "form",
    {
      class: "card login",
      onsubmit: async (ev) => {
        ev.preventDefault();
        err.textContent = "";
        try {
          const r = await api("/api/auth/login", { method: "POST", body: { passphrase: inp.value } });
          state.csrf = r.csrf;
          await boot();
        } catch (e) {
          err.textContent = e.message;
        }
      },
    },
    h("h2", {}, "fv-control"),
    h("p", { class: "muted small" }, m.configured === false ? "No owner passphrase is configured on this Worker (OWNER_PASSPHRASE_HASH)." : "Sign in with the owner passphrase."),
    inp,
    h("button", { class: "primary", type: "submit" }, "Sign in"),
    err,
  );
  main.replaceChildren(form);
  inp.focus();
}
async function boot() {
  state.me = await api("/api/auth/me");
  state.csrf = state.me.csrf || null;
  if (!state.me.authenticated) return renderLogin();
  $("#nav").hidden = false;
  $("#logoutBtn").hidden = state.me.mode !== "passphrase";
  fetch("/healthz").then((r) => r.json()).then((j) => ($("#envTag").textContent = j.environment)).catch(() => {});
  route();
}
window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  $("#themeBtn").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("fvc-theme", next); } catch {}
    route();
  });
  $("#logoutBtn").addEventListener("click", async () => {
    await api("/api/auth/logout", { method: "POST" }).catch(() => {});
    state.me = { authenticated: false, mode: "passphrase" };
    renderLogin();
  });
  boot().catch((e) => $("#main").replaceChildren(card("Error", e.message)));
});
// The editor bundle (CodeMirror 6 + forms, ~140 KiB gzip) loads only on pages that edit or view JSON.
let editorLoading = null;
function fvEditor() {
  if (window.FVEditor) return Promise.resolve(window.FVEditor);
  editorLoading ||= new Promise((res, rej) => {
    const sc = document.createElement("script");
    sc.src = "/editor.js";
    sc.onload = () => res(window.FVEditor);
    sc.onerror = () => { editorLoading = null; rej(new Error("could not load the editor")); };
    document.head.append(sc);
  });
  return editorLoading;
}
/** A document editor panel (Form / JSON, validate, review, plan, save, history). */
function docPanel(kind, id, opts = {}) {
  const host = h("div", { class: "docpanel" }, h("span", { class: "muted small" }, "loading editor…"));
  fvEditor()
    .then((E) => E.openDocPanel(host, { kind, id, api, toast, ...opts }))
    .catch((e) => host.replaceChildren(h("p", { class: "small" }, e.message)));
  return host;
}
/** A read-only JSON tree (search, copy path / value). */
function jsonTree(value, opts = {}) {
  const host = h("div", {}, h("span", { class: "muted small" }, "loading viewer…"));
  fvEditor().then((E) => host.replaceChildren(E.renderTree(value, opts))).catch((e) => host.replaceChildren(h("pre", { class: "log" }, JSON.stringify(value, null, 2))));
  return host;
}
function every(ms, fn) {
  state.timers.push(setInterval(fn, ms));
}

// ---------------------------------------------------------------- dashboard
async function pageDashboard(main) {
  const draw = async () => {
    const [o, series, bal, costs, bp, bpm] = await Promise.all([api("/api/overview"), api("/api/metrics/series?hours=6"), api("/api/balance?hours=24"), api("/api/costs?days=7"), api("/api/buildpod").catch((e) => ({ error: e.message, pods: [] })), api("/api/build-pods").catch((e) => ({ error: e.message, pods: [] }))]);
    const tiles = h(
      "div",
      { class: "tiles" },
      tile("Balance", fmt$(o.balance), o.balance_change_1h !== null ? `${o.balance_change_1h >= 0 ? "+" : ""}${fmt$(o.balance_change_1h)} last hour` : `floor ${fmt$(o.floor, 0)}`),
      tile("Burn", `${fmt$(o.burn_per_hr)}/hr`, `${o.running_pods} running pods, ${o.running_gpu_pods} GPU`),
      tile("Time to floor", o.hours_to_floor === null ? "–" : o.hours_to_floor > 48 ? `${(o.hours_to_floor / 24).toFixed(1)} d` : `${o.hours_to_floor.toFixed(1)} h`, `at the current burn, floor ${fmt$(o.floor, 0)}`),
      tile("Spend today", fmt$(o.cost_today_total), "UTC day, all pods"),
      tile("Idle burn", `${fmt$(o.idle_burn_per_hr)}/hr`, `${o.idle_pods.length} idle GPU pods (> ${o.policies.idle_min} min)`),
      tile("Open alerts", String(o.alerts.length), o.alerts.filter((a) => a.severity === "critical").length ? `${o.alerts.filter((a) => a.severity === "critical").length} critical` : "none critical"),
    );
    const resolve = async (a) => {
      if (!confirm(`Resolve "${a.message}"? It opens again at the next collector pass (each minute) if its condition still holds. (audited)`)) return;
      await act("resolve", () => api(`/api/alerts/${a.id}/resolve`, { method: "POST" }));
      draw().catch(() => {});
    };
    const alerts = card(
      "Alerts",
      o.alerts.length
        ? o.alerts.map((a) =>
            h(
              "div",
              { class: `alert sev-${a.severity}`, "data-alert": a.id },
              h("span", { class: "icon", "aria-hidden": "true" }, a.severity === "critical" ? "!" : a.severity === "warn" ? "▲" : "i"),
              h("div", { style: "flex:1" }, h("div", {}, h("b", {}, a.severity), " ", a.message), h("div", { class: "muted small" }, `${a.kind} · since ${ago(a.opened_at)}${a.action ? ` · action: ${a.action}` : ""}`)),
              h("button", { class: "ghost", title: "Mark resolved", onclick: () => resolve(a) }, "Resolve"),
            ),
          )
        : h("div", { class: "muted small" }, "No open alerts."),
    );
    // Top 8 running pods by $/hr get a line; the rest are in the table.
    const pods = new Map();
    for (const p of series.points) (pods.get(p.pod) || pods.set(p.pod, []).get(p.pod)).push(p);
    const podNames = await podNameMap();
    const ranked = [...pods.keys()].sort((a, b) => (podNames.get(b)?.cost_per_hr || 0) - (podNames.get(a)?.cost_per_hr || 0));
    const gpuPods = ranked.filter((id) => pods.get(id).some((p) => p.gpu !== null)).slice(0, 8);
    const cpuPods = ranked.slice(0, 8);
    const name = (id) => podNames.get(id)?.name || id;
    const gpuChart = lineChart({ title: "GPU utilisation", series: gpuPods.map((id) => ({ name: name(id), points: pods.get(id).map((p) => [p.t, p.gpu]) })), yMax: 100, yFmt: (v) => `${Math.round(v)}%` });
    const cpuChart = lineChart({ title: "CPU utilisation", series: cpuPods.map((id) => ({ name: name(id), points: pods.get(id).map((p) => [p.t, p.cpu]) })), yFmt: (v) => `${Math.round(v)}%` });
    const balChart = lineChart({ title: "Balance", series: [{ name: "balance", points: bal.points.map((p) => [p.at, p.balance]) }], yMin: 0, yFmt: (v) => fmt$(v, 0) });
    const burnChart = lineChart({ title: "Burn", series: [{ name: "$/hr", points: bal.points.map((p) => [p.at, p.spend_per_hr]), color: "var(--series-2)" }], yFmt: (v) => fmt$(v, 1) });
    const owners = [...new Set(costs.by_owner.map((r) => r.owner))];
    const topOwners = owners.slice(0, 7);
    const days = costs.by_day.map((r) => r.day);
    const stacks = topOwners.map((ow) => ({ name: ow, values: days.map((d) => costs.by_day_owner.filter((r) => r.day === d && r.owner === ow).reduce((a, r) => a + r.usd, 0)) }));
    if (owners.length > 7) stacks.push({ name: "other", color: "var(--text-muted)", values: days.map((d) => costs.by_day_owner.filter((r) => r.day === d && !topOwners.includes(r.owner)).reduce((a, r) => a + r.usd, 0)) });
    const clusters = card(
      "Clusters",
      table(
        [
          { label: "cluster", get: (c) => h("a", { href: `#/cluster/${c.id}` }, c.name) },
          { label: "status", get: (c) => badge(c.status, statusKind(c.status)) },
          { label: "pods", get: (c) => c.pods, num: true },
          { label: "$/hr", get: (c) => fmt$(c.dph), num: true },
          { label: "today", get: (c) => fmt$(c.cost_today), num: true },
          { label: "deadline", get: (c) => until(c.deadline) },
        ],
        o.clusters,
        "No clusters yet: define one under Clusters.",
      ),
    );
    const idle = card(
      "Idle GPU pods",
      table(
        [
          { label: "pod", get: (p) => h("a", { href: `#/pod/${p.pod_id}` }, p.name || p.pod_id) },
          { label: "owner", get: (p) => p.owner },
          { label: "idle", get: (p) => `${p.idle_min} min`, num: true },
          { label: "$/hr", get: (p) => fmt$(p.cost_per_hr), num: true },
        ],
        o.idle_pods,
        `No GPU pod has been idle (GPU < ${o.policies.idle_gpu_pct}% and no running jobs) for ${o.policies.idle_min} min.`,
      ),
    );
    main.replaceChildren(
      h("h1", {}, "Dashboard"),
      h("div", { class: "muted small", style: "margin:-8px 0 12px" }, `Updated ${ago(o.at)} · metrics from ${series.source}`),
      tiles,
      h("div", { class: "grid g2" }, alerts, clusters),
      bp.pods.length ? buildPodCard(bp) : "",
      h("div", { class: "grid g2" }, card("Balance, 24 h", balChart), card("Burn $/hr, 24 h", burnChart)),
      h("div", { class: "grid g2" }, card("GPU utilisation, 6 h (top pods by $/hr)", gpuChart), card("CPU utilisation, 6 h", cpuChart)),
      h("div", { class: "grid g2" }, card("Spend per day by owner, 7 days", stackedBars({ cats: days, stacks })), idle),
      buildPodsCard(bpm, () => draw().catch(() => {})),
    );
  };
  await draw();
  every(60_000, () => draw().catch(() => {}));
}
/** Build pods managed by fv-control (docs/dev/build-pods-fv-control.md): up, stop, start, delete; runner, spend, timers. */
function buildPodsCard(bpm, redraw) {
  const pol = bpm.policy || {};
  const call = async (label, path, opts) => {
    try {
      return await act(label, () => api(path, opts));
    } catch (e) {
      // Busy (jobs or a workflow job on its runner): ask before forcing.
      if (/busy/.test(e.message) && confirm(`${e.message}\n\nForce it? Running jobs are killed. (audited)`)) {
        const force = opts.method === "DELETE" ? { path: `${path}?force=1` } : { body: { ...(opts.body || {}), force: true } };
        return act(label, () => api(force.path || path, { ...opts, ...(force.body ? { body: force.body } : {}) }));
      }
      return null;
    }
  };
  const after = () => setTimeout(redraw, 800);
  const one = (p) => {
    const hz = p.health;
    const running = p.state === "running";
    return h(
      "div",
      { class: "buildpod", "data-bp": p.id },
      h(
        "h3",
        {},
        p.pod_id ? h("a", { href: `#/pod/${p.pod_id}` }, p.name) : p.name,
        " ",
        badge(p.state, statusKind(p.state === "running" ? "RUNNING" : p.state === "stopped" ? "EXITED" : p.state)),
        " ",
        running ? badge(p.phase || "starting", p.phase === "ready" ? "good" : "warn") : "",
        " ",
        p.outdated ? badge("outdated: replaced when stopped", "warn") : "",
        p.purpose === "test" ? badge("test", "warn") : "",
      ),
      h("dl", { class: "kv" }, ...[
        ["where", `${p.dc || "?"} · ${p.flavor || "?"} ${p.vcpu || "?"} vCPU · ${p.disk_gb || "?"} GB${p.volume_id ? ` · volume ${p.volume_id}` : ""}`],
        ["$/hr · today", `${fmt$(p.cost_per_hr)} · ${fmt$(p.spend_today)}`],
        ["runs", `${p.server} · ${String(p.image).split(":").pop()}`],
        ["limits", `idle stop ${p.limits.idle_min} min · cap ${p.limits.max_h} h + ${p.limits.max_grace_min} min`],
        ["timers", hz ? `up ${dur(hz.uptime_s)} · ${hz.jobs_active ? `${hz.jobs_active} job(s) running` : `idle ${dur(hz.idle_s)}`}${hz.idle_stop_in_s != null ? ` · idle stop in ${dur(hz.idle_stop_in_s)}` : ""}` : "–"],
        ["runner", `${p.runner.state || "none"}${p.runner.name ? ` · ${p.runner.name}` : ""}${p.runner.labels ? ` · ${p.runner.labels}` : ""}${p.runner.error ? ` · ${p.runner.error}` : ""}`],
        ["error", p.last_error],
      ].filter(([, v]) => v).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
      h(
        "div",
        { class: "row" },
        running ? h("button", { onclick: async () => { if (confirm(`Stop ${p.name}? The container disk (target dirs) is lost; caches stay. (audited)`)) { await call("stop", `/api/build-pods/${p.id}/stop`, { method: "POST", body: {} }); after(); } } }, "Stop") : "",
        p.state === "stopped" ? h("button", { onclick: async () => { await call("start", `/api/build-pods/${p.id}/start`, { method: "POST", body: {} }); after(); } }, "Start") : "",
        p.state !== "deleted" ? h("button", { class: "danger", onclick: async () => { if (confirm(`Delete ${p.name}? (audited)`)) { await call("delete", `/api/build-pods/${p.id}`, { method: "DELETE" }); after(); } } }, "Delete") : "",
      ),
    );
  };
  const live = (bpm.pods || []).filter((p) => p.state !== "deleted");
  return card(
    "Build pods",
    h(
      "p",
      { class: "muted small" },
      pol.enabled
        ? `Managed by fv-control: up to ${pol.max_pods} running, ≤ ${fmt$(pol.max_dph_per_pod)}/hr each, ${fmt$(bpm.spend_today || 0)} of ${fmt$(pol.daily_usd_max)} today; [${(pol.flavors || []).join(" ")}] × [${(pol.vcpus || []).join(" ")}] vCPU, regions ${(pol.regions || []).join(" ") || "any"}${pol.regions_only ? " only" : " first"}; GitHub runner ${pol.runner ? (bpm.secrets?.runner_pat ? "on" : "on, but no GITHUB_RUNNER_PAT") : "off"}${pol.wake_on_queue ? ", woken by queued jobs" : ""}; R2 cache ${bpm.secrets?.r2_cache ? "on" : "off"}.`
        : "Build pods are off (build_pods.enabled). Agents use scripts/dev/build-pod.sh up, which asks fv-control for a pod.",
    ),
    h("div", { class: "row" }, h("button", { class: "primary", disabled: !pol.enabled, onclick: async () => { const r = await call("up", "/api/build-pods/up", { method: "POST", body: {} }); if (r) toast(`up: ${r.action} ${r.pod.name}`); after(); } }, "Up (reuse / start / create)")),
    bpm.error ? h("p", { class: "small" }, bpm.error) : live.length ? live.map(one) : h("p", { class: "muted small" }, "No managed build pod."),
    h("details", {}, h("summary", { class: "small" }, "Policy (build_pods)"), docJson(pol, async (next) => { await act("policy", () => api("/api/build-pods/policy", { method: "PUT", body: { policy: next } })); after(); })),
  );
}
/** A JSON textarea with a Save button (the build pods policy). */
function docJson(value, onSave) {
  const ta = h("textarea", { rows: 18, style: "width:100%;font-family:var(--mono, monospace)" });
  ta.value = JSON.stringify(value, null, 2);
  return h("div", {}, ta, h("button", { onclick: async () => { let v; try { v = JSON.parse(ta.value); } catch (e) { toast(`policy: ${e.message}`); return; } await onSave(v); } }, "Save"));
}
/** The legacy shared build pod (read only): its own self-stop timers (/healthz) and the controller's backstop. */
function buildPodCard(bp) {
  const pol = bp.policy || {};
  const kv = (rows) => h("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v ?? "–")]));
  const one = (p) => {
    const hz = p.health;
    const ss = hz?.self_stop;
    const jobs = hz?.jobs;
    return h(
      "div",
      { class: "buildpod", "data-pod": p.pod_id },
      h("h3", {}, h("a", { href: `#/pod/${p.pod_id}` }, p.name || p.pod_id), " ", badge(p.status || "?", statusKind(p.status)), " ", hz ? badge(hz.ready ? "ready" : hz.phase || "setting up", hz.ready ? "good" : "warn") : p.status === "RUNNING" ? badge("no /healthz", "warn") : ""),
      kv([
        ["$/hr", fmt$(p.cost_per_hr)],
        ["up", dur(hz?.uptime_s ?? p.uptime_s)],
        ["idle", hz ? (hz.jobs_active ? `no (${hz.jobs_active} job(s) running)` : dur(hz.idle_s)) : "–"],
        ["idle stop in", hz ? (hz.idle_stop_in_s === null || hz.idle_stop_in_s === undefined ? (hz.jobs_active ? "paused: jobs running" : "–") : dur(hz.idle_stop_in_s)) : "–"],
        ["cap stop in", hz?.max_stop_in_s !== undefined ? `${dur(hz.max_stop_in_s)} (cap ${dur(hz.max_s)} + ${dur(hz.max_grace_s)} grace for running jobs)` : "–"],
        [
          "last self-stop",
          ss && ss.attempts
            ? h("span", {}, ss.ok ? badge("accepted", "good") : badge("FAILED", "critical"), ` ${ss.reason || ""} ${ss.at ? ago(ss.at * 1000) : ""} · ${ss.attempts} attempt(s)${ss.next_at ? ` · next ${until(ss.next_at * 1000)}` : ""}`, ss.error ? h("div", { class: "small wrap" }, ss.error) : "")
            : ss
              ? "none yet"
              : "– (older build pod server)",
        ],
        ["controller backstop", !p.backstop.enabled ? "off" : p.backstop.verdict ? badge(`due: ${p.backstop.verdict}`, "critical") : `cap in ${dur(p.backstop.cap_in_s)}${p.backstop.idle_in_s !== null ? `, idle in ${dur(p.backstop.idle_in_s)}` : ""}`],
      ]),
      jobs && jobs.length
        ? table([{ label: "job", get: (j) => h("code", {}, j.id) }, { label: "agent", get: (j) => j.agent }, { label: "state", get: (j) => j.state }, { label: "for", get: (j) => dur(j.seconds), num: true }], jobs)
        : h("p", { class: "muted small" }, jobs ? "No jobs running." : hz ? `${hz.jobs_active ?? 0} job(s) running.` : ""),
    );
  };
  return card(
    "Build pod",
    h("p", { class: "muted small" }, `Read only. The shared CPU build pod (owner external:build-pod) stops itself when idle and at its cap; the controller's backstop stops it ${pol.max_h ? `past ${pol.max_h} h up` : ""}${pol.idle_grace_min !== undefined ? ` or ${pol.idle_grace_min} min past its own idle stop` : ""} if that failed (Settings → policies).`),
    bp.error ? h("p", { class: "small" }, bp.error) : bp.pods.length ? bp.pods.map(one) : h("p", { class: "muted small" }, "No build pod (none attributed to external:build-pod)."),
  );
}
function tile(k, v, sub) {
  return h("div", { class: "tile" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "s" }, sub || ""));
}
async function podNameMap() {
  const r = await api("/api/pods?all=1");
  return new Map(r.pods.map((p) => [p.pod_id, p]));
}

// ---------------------------------------------------------------- pods
async function pagePods(main) {
  const r = await api("/api/pods");
  const t = Date.now();
  main.replaceChildren(
    h("h1", {}, "Pods"),
    card(
      `All pods of the Runpod account (${r.pods.length})`,
      table(
        [
          { label: "pod", get: (p) => h("a", { href: `#/pod/${p.pod_id}` }, p.name || p.pod_id) },
          { label: "owner", get: (p) => p.owner },
          { label: "status", get: (p) => badge(p.desired_status, statusKind(p.desired_status)) },
          { label: "health", get: (p) => (p.health ? badge(p.health, healthKind(p.health)) : "–") },
          { label: "GPU", get: (p) => p.gpu || "cpu" },
          { label: "GPU %", get: (p) => fmtPct(p.gpu_util), num: true },
          { label: "CPU %", get: (p) => fmtPct(p.cpu), num: true },
          { label: "jobs", get: (p) => (p.jobs_running === null ? "–" : `${p.jobs_running}/${p.jobs_queued}`), num: true },
          { label: "idle", get: (p) => (p.idle_since ? `${Math.round((t - p.idle_since) / 60000)} min` : "–"), num: true },
          { label: "$/hr", get: (p) => fmt$(p.cost_per_hr), num: true },
          { label: "today", get: (p) => fmt$(p.cost_today), num: true },
          { label: "uptime", get: (p) => (p.uptime_s ? `${(p.uptime_s / 3600).toFixed(1)} h` : "–"), num: true },
          { label: "DC", get: (p) => p.dc },
        ],
        r.pods,
      ),
    ),
  );
}
async function pagePod(main, id) {
  const [r, series] = await Promise.all([api(`/api/pods/${id}`), api(`/api/metrics/series?hours=24&pod=${encodeURIComponent(id)}`)]);
  const p = r.pod;
  const pts = series.points;
  main.replaceChildren(
    h("h1", {}, p.name || p.pod_id),
    h(
      "div",
      { class: "grid g2" },
      card(
        "Pod",
        h(
          "dl",
          { class: "kv" },
          ...[
            ["id", p.pod_id], ["owner", p.owner], ["status", p.desired_status], ["health", p.health], ["GPU", `${p.gpu || "cpu"} × ${p.gpu_count || 0}`], ["DC", p.dc], ["$/hr", fmt$(p.cost_per_hr)],
            ["image", p.image], ["build", p.build_sha], ["first seen", dt(p.first_seen)], ["last seen", ago(p.last_seen)], ["controller", r.controller ? `${r.controller.role} ${r.controller.pool || ""} (${r.controller.status})` : "external"],
          ].flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v ?? "–")]),
        ),
        h("div", { class: "row", style: "margin-top:10px" }, h("a", { class: "btn", href: `#/logs?pod=${p.pod_id}` }, "Logs")),
      ),
      card("Cost per day", table([{ label: "day", get: (c) => c.day }, { label: "cost", get: (c) => fmt$(c.usd), num: true }, { label: "minutes", get: (c) => c.minutes, num: true }, { label: "idle min", get: (c) => c.idle_minutes, num: true }], r.costs)),
    ),
    h(
      "div",
      { class: "grid g2" },
      card("Utilisation, 24 h", lineChart({ series: [{ name: "GPU", points: pts.map((x) => [x.t, x.gpu]) }, { name: "CPU", points: pts.map((x) => [x.t, x.cpu]) }, { name: "memory", points: pts.map((x) => [x.t, x.mem]) }], yFmt: (v) => `${Math.round(v)}%` })),
      card("Running jobs, 24 h", lineChart({ series: [{ name: "jobs", points: pts.map((x) => [x.t, x.jobs]) }] })),
    ),
    r.controller && typeof fvBootTimeline === "function" ? fvBootTimeline(p.pod_id) : null,
    h("div", { class: "grid g2" }, card("Snapshot (JSON)", jsonTree({ pod: p, controller: r.controller, costs: r.costs }, { open: 1 })), card("Metric samples, 24 h (JSON)", jsonTree(series, { open: 1 }))),
  );
}

// ---------------------------------------------------------------- clusters
async function pageClusters(main) {
  const [r, tpl] = await Promise.all([api("/api/clusters"), api("/api/templates")]);
  const ta = { value: JSON.stringify({ ...tpl["tiny-cpu"], name: "tiny" }, null, 2) };
  const edHost = h("div", {});
  let ed = null;
  Promise.all([fvEditor(), api("/api/schemas"), api("/api/schemas/dynamic")]).then(([E, sc, dyn]) => {
    ed = E.createJsonEditor({ parent: edHost, doc: ta.value, schema: sc.schemas["cluster-spec"], dynamic: dyn, onChange: (t) => (ta.value = t) });
  }).catch(() => edHost.replaceChildren(h("textarea", { oninput: (e) => (ta.value = e.target.value) }, ta.value)));
  const tplList = tpl.templates || [{ id: "tiny-cpu", title: "1 fake-engine CPU worker" }, { id: "standard", title: "h3-turbo, h3-max, ltx, wan GPU pools" }];
  const tplSel = h(
    "select",
    { "aria-label": "Template", onchange: () => { ta.value = JSON.stringify({ ...tpl[tplSel.value], name: tplSel.value === "tiny-cpu" ? "tiny" : tplSel.value === "standard" ? "main" : tplSel.value }, null, 2); ed?.set(ta.value); } },
    [...tplList].sort((a, b) => (a.id === "tiny-cpu" ? -1 : b.id === "tiny-cpu" ? 1 : 0)).map((x) => h("option", { value: x.id }, `${x.id}: ${x.title}`)),
  );
  // Add a pool preset to the spec being defined.
  const presetAdd = presetPicker(tpl.pool_presets || [], (pool) => {
    let spec;
    try { spec = JSON.parse(ta.value); } catch { return toast("spec: not JSON"); }
    spec.pools = [...(spec.pools || []).filter((p) => p.id !== pool.id), pool];
    ta.value = JSON.stringify(spec, null, 2);
    ed?.set(ta.value);
    toast(`pool ${pool.id} added`);
  });
  main.replaceChildren(
    h("h1", {}, "Clusters"),
    card(
      null,
      table(
        [
          { label: "cluster", get: (c) => h("a", { href: `#/cluster/${c.id}` }, c.name) },
          { label: "status", get: (c) => badge(c.status, statusKind(c.status)) },
          { label: "image", get: (c) => c.spec.image.channel || c.spec.image.sha || "ref" },
          { label: "pools", get: (c) => c.spec.pools.map((p) => `${p.id}×${p.count}`).join(", ") },
          { label: "deadline", get: (c) => until(c.deadline) },
          { label: "operation", get: (c) => (c.op ? `${c.op.kind} (${c.op.phase})` : "–") },
          { label: "source", get: (c) => c.source },
        ],
        r.clusters,
        "No clusters defined yet.",
      ),
    ),
    h(
      "div",
      { class: "grid g2" },
      card(
        "Define a cluster",
        h("div", { class: "row", style: "margin-bottom:8px" }, tplSel),
        presetAdd,
        edHost,
        h(
          "div",
          { class: "row", style: "margin-top:8px" },
          h("button", {
            class: "primary",
            onclick: async () => {
              let spec;
              try { spec = JSON.parse(ta.value); } catch (e) { return toast("spec: not JSON"); }
              const j = await act("define", () => api("/api/clusters", { method: "POST", body: { spec } }));
              location.hash = `#/cluster/${j.cluster.id}`;
            },
          }, "Define"),
          h("span", { class: "muted small" }, "Defining starts nothing. Start runs a price check first."),
        ),
      ),
    ),
  );
}

async function pageCluster(main, id) {
  // Kept across the 10 s redraws: the key list and the front output stay up.
  const gwKeysEl = h("div", { id: "gwKeys", style: "margin-top:8px" });
  const gwOutEl = h("pre", { id: "gwOut", class: "log", style: "margin-top:8px;max-height:300px" });
  const draw = async () => {
    const r = await api(`/api/clusters/${id}`);
    const c = r.cluster;
    const running = r.pods.length > 0;
    const op = r.op;
    const btn = (label, fn, cls) => h("button", { class: cls, disabled: !!op && label !== "Cancel operation", onclick: fn }, label);
    const post = (path, body, label) => act(label, () => api(`/api/clusters/${c.id}/${path}`, { method: "POST", body: body || {} })).then(() => setTimeout(draw, 800));
    const actions = h(
      "div",
      { class: "row" },
      !running && btn("Start", () => startDialog(c, draw), "primary"),
      running && btn("Stop (delete pods)", () => confirm(`Delete every pod of ${c.name}?`) && post("stop", {}, "stop"), "danger"),
      running && btn("Extend…", () => { const m = prompt("Extend the deadline by how many minutes?", "30"); if (m) post("extend", { minutes: Number(m) }, "extend"); }),
      running && btn("Roll to…", () => rollDialog(c, post)),
      running && btn("Restart…", () => restartDialog(c, r.pods, post)),
      btn("Add pool…", async () => addPoolDialog(c, (await api("/api/templates")).pool_presets || [], () => route())),
      op && btn("Cancel operation", () => confirm("Cancel the running operation? Pods it created stay (the backstops hold them).") && post("cancel", {}, "cancel")),
      !running && btn("Delete definition", async () => { if (confirm(`Delete the definition of ${c.name}?`)) { await act("delete", () => api(`/api/clusters/${c.id}`, { method: "DELETE" })); location.hash = "#/clusters"; } }, "danger"),
    );
    const live = new Map(r.live.map((p) => [p.pod_id, p]));
    const pools = card(
      "Pools",
      table(
        [
          { label: "pool", get: (p) => p.id },
          { label: "variant", get: (p) => p.variant },
          { label: "compute", get: (p) => p.compute },
          { label: "workers", get: (p) => `${(c.state.workers[p.id] || []).length} / ${p.count}`, num: true },
          {
            label: "",
            get: (p) => running ? h("button", { disabled: !!op, onclick: () => { const n = prompt(`Workers for ${p.id}`, String(p.count)); if (n !== null) post("scale", { pool: p.id, count: Number(n) }, "scale"); } }, "Scale…") : "",
          },
        ],
        c.spec.pools,
      ),
    );
    const pods = card(
      "Pods",
      table(
        [
          { label: "pod", get: (p) => h("a", { href: `#/pod/${p.pod_id}` }, p.pod_id) },
          { label: "role", get: (p) => `${p.role}${p.pool ? ` ${p.pool}` : ""}${p.slot !== "workers" && p.slot !== "gateway" ? ` (${p.slot})` : ""}` },
          { label: "status", get: (p) => badge(p.status, statusKind(p.status === "ready" ? "running" : p.status)) },
          { label: "health", get: (p) => (live.get(p.pod_id)?.health ? badge(live.get(p.pod_id).health, healthKind(live.get(p.pod_id).health)) : "–") },
          { label: "GPU/CPU", get: (p) => p.gpu || "–" },
          { label: "GPU %", get: (p) => fmtPct(live.get(p.pod_id)?.gpu_util), num: true },
          { label: "jobs", get: (p) => { const l = live.get(p.pod_id); return l && l.jobs_running !== null ? `${l.jobs_running}/${l.jobs_queued}` : "–"; }, num: true },
          { label: "$/hr", get: (p) => fmt$(p.cost_per_hr), num: true },
          { label: "image", get: (p) => (p.image || "").split("@").pop().slice(0, 19) },
          { label: "logs", get: (p) => h("a", { href: `#/logs?pod=${p.pod_id}` }, "logs") },
        ],
        r.pods,
        "No pods: the cluster is stopped.",
      ),
    );
    const d = r.drift;
    const drift = card(
      "Version",
      h("p", { class: "small" }, `Follows ${c.spec.image.channel ? `channel ${c.spec.image.channel}` : c.spec.image.sha ? `commit ${c.spec.image.sha}` : "an image reference"}; running ${d.running_sha || "?"}; channel head ${d.head_sha || "?"}. `, d.drift ? badge("drift: roll to update", "warn") : badge("in step", "good")),
      table([{ label: "pod", get: (x) => x.pod }, { label: "release key", get: (x) => x.key }, { label: "running", get: (x) => (x.running || "–").slice(0, 19) }, { label: "head", get: (x) => (x.head || "–").slice(0, 19) }, { label: "", get: (x) => (x.drift ? badge("drift", "warn") : "") }], d.pods, "No pods."),
    );
    const opsCard = card(
      "Operations",
      table(
        [
          { label: "when", get: (o) => ago(o.created_at) },
          { label: "kind", get: (o) => h("a", { href: "#", onclick: (ev) => { ev.preventDefault(); showOp(o.id); } }, o.kind) },
          { label: "status", get: (o) => badge(o.status, statusKind(o.status)) },
          { label: "by", get: (o) => o.actor },
          { label: "error", get: (o) => o.error || "", wrap: true },
        ],
        r.ops,
        "No operations yet.",
      ),
      h("div", { id: "opLog" }),
    );
    // edge (docs/serve/edge-control-plane.md): the edge Worker is the front;
    // direct (docs/control/gateway-less-auth.md): clients call each worker; the admin token and keys work on every one.
    const edge = c.spec.control_plane !== "direct";
    const gw = edge ? r.edge_url : null;
    const direct = !edge;
    const wurls = Object.entries(c.state.workers || {}).flatMap(([pool, l]) => l.filter((r) => r.url).map((r) => ({ pool, pod: r.pod, url: r.url })));
    const reach = !!(gw || (direct && wurls.length));
    const out = (t) => { $("#gwOut").textContent = t; };
    const showKeys = async () => {
      const j = await act("keys", () => api(`/api/clusters/${c.id}/keys`));
      const box = $("#gwKeys");
      box.replaceChildren(table(
        [
          { label: "id", get: (k) => k.id },
          { label: "name", get: (k) => k.name },
          { label: "prefix", get: (k) => k.prefix },
          { label: "created", get: (k) => k.created_at || "" },
          { label: "last used", get: (k) => k.last_used_at || "–" },
          { label: "", get: (k) => (k.revoked ? badge("revoked", "critical") : h("button", { onclick: async () => { if (!confirm(`Revoke ${k.name} (${k.id})? (audited)`)) return; const r = await act("revoke", () => api(`/api/clusters/${c.id}/keys/${k.id}`, { method: "DELETE" })); out(`revoked ${k.id} on ${r.applied.join(", ")}${r.failed.length ? `; not reached: ${r.failed.map((f) => f.pod).join(", ")} (they read it from D1 within 30 s)` : ""}`); showKeys(); } }, "Revoke")) },
        ],
        j.keys,
        "No minted keys.",
      ));
    };
    const tools = card(
      edge ? "Edge (the cluster's only front)" : "Workers (direct)",
      gw
        ? h("p", { class: "small" }, "URL ", h("a", { href: gw, target: "_blank", rel: "noopener" }, gw), " · ", h("a", { href: `${gw}/console`, target: "_blank", rel: "noopener" }, "console"))
        : direct && wurls.length
          ? h("div", { class: "small" }, h("p", { class: "muted small" }, "Clients call each worker directly with an API key; the admin token and minted keys work on every worker."), ...wurls.map((w) => h("p", {}, `${w.pool} ${w.pod}: `, h("a", { href: w.url, target: "_blank", rel: "noopener" }, w.url), " · ", h("a", { href: `${w.url}/console`, target: "_blank", rel: "noopener" }, "console"))))
          : h("p", { class: "muted small" }, edge ? "fv-control has no edge (EDGE_URL)." : "No workers."),
      reach && h(
        "div",
        { class: "row" },
        h("button", { onclick: async () => { const j = await act("status", () => api(`/api/clusters/${c.id}/front`)); out(JSON.stringify(j.edge ? { status: j.status, workers: j.workers, families: j.families } : { workers: j.workers }, null, 2)); } }, edge ? "Families view" : "Workers view"),
        h("button", { onclick: async () => { if (!confirm(`Show the ${edge ? "edge" : "cluster"}'s admin token? (audited)`)) return; const j = await act("admin token", () => api(`/api/clusters/${c.id}/admin-token`, { method: "POST" })); out(`admin token: ${j.admin_token}\nconsole: ${j.console}${j.direct ? `\nworkers:\n${j.workers.map((w) => `  ${w.pool} ${w.url}`).join("\n")}` : ""}`); } }, "Reveal admin token"),
        h("button", { onclick: async () => { const n = prompt("Name of the new user API key", "laptop"); if (!n) return; const j = await act("mint", () => api(`/api/clusters/${c.id}/mint-key`, { method: "POST", body: { name: n } })); out(`API key (shown once): ${j.api_key}${j.propagation_s ? `\nworks on ${j.minted_on} now, on the other workers within ${j.propagation_s} s` : ""}`); } }, "Mint user API key"),
        h("button", { onclick: showKeys }, "Keys"),
      ),
      gwKeysEl,
      gwOutEl,
    );
    main.replaceChildren(
      h("h1", {}, c.name, " ", badge(c.status, statusKind(c.status))),
      h("p", { class: "muted small" }, `deadline ${dt(c.deadline)} (${until(c.deadline)}) · floor $${c.spec.balance_floor} · pod watchdog below $${c.spec.min_balance} · ${c.source}`, op ? ` · running: ${op.kind} (${op.phase})` : ""),
      card(null, actions),
      h("div", { class: "grid g2" }, pools, drift),
      pods,
      h("div", { class: "grid g2" }, opsCard, tools),
      card("Environment", h("p", { class: "small" }, h("a", { href: `#/env?cluster=${c.id}` }, "Cluster and pod env, effective values, restarts →"))),
    );
    if (op) showOp(r.ops.find((o) => o.status === "running")?.id);
  };
  const live = h("div", {});
  const outer = main;
  main = live;
  await draw();
  outer.replaceChildren(live, card("Spec", h("p", { class: "muted small" }, "Edit the definition: the form and the JSON stay in sync; Review shows the diff and the plan (what would be created, stopped or restarted, and the $/hr against the floor) before saving."), docPanel("cluster-spec", id, { title: "cluster spec", cluster: id })));
  every(10_000, () => draw().catch(() => {}));
}
async function showOp(id) {
  if (!id) return;
  const j = await api(`/api/ops/${id}`);
  const el = $("#opLog");
  if (!el) return;
  el.replaceChildren(h("h3", {}, `${j.operation.kind} ${j.operation.id}`), h("div", { class: "log" }, j.operation.log.map((l) => h("div", {}, `${new Date(l.at).toISOString().slice(11, 19)}  ${l.msg}`))));
}
/** A pool preset select + description + "Add pool" (calls onAdd with a copy of the preset's pool). */
function presetPicker(presets, onAdd, label = "Add pool") {
  if (!presets.length) return h("span", {});
  const info = h("p", { class: "muted small", style: "margin:4px 0 0" });
  const sel = h("select", { "aria-label": "Pool preset", onchange: () => show() }, presets.map((p) => h("option", { value: p.id }, `${p.id}: ${p.title}`)));
  const show = () => {
    const p = presets.find((x) => x.id === sel.value);
    info.replaceChildren(p.description, ` Image ${p.pool.variant}; weights ${p.weights.join(", ")}.`, p.licence ? h("span", {}, " ", badge("licence", "critical"), " ", p.licence) : "");
  };
  show();
  return h(
    "div",
    { class: "presets", style: "margin-bottom:8px" },
    h("div", { class: "row" }, sel, h("button", { onclick: () => { const p = presets.find((x) => x.id === sel.value); if (p.licence && !confirm(`${p.title}: ${p.licence}. Add it anyway?`)) return; onAdd(JSON.parse(JSON.stringify(p.pool))); } }, label)),
    info,
  );
}
/** A modal with checkboxes; resolves to the checked values (null: cancelled). */
function pickDialog(title, intro, groups, okLabel, extra) {
  return new Promise((resolve) => {
    const dlg = h("dialog", { class: "pick" });
    const boxes = [];
    const body = groups.map((g) =>
      h(
        "fieldset",
        {},
        h("legend", {}, g.label),
        g.items.map((it) => {
          const cb = h("input", { type: "checkbox", value: it.value, checked: !!it.checked });
          boxes.push(cb);
          return h("label", { class: "pickrow" }, cb, " ", it.label, it.note ? h("span", { class: "muted small" }, ` ${it.note}`) : "");
        }),
      ),
    );
    let done = false;
    const finish = (v) => { if (done) return; done = true; dlg.close(); resolve(v); };
    dlg.append(
      h("h2", {}, title),
      intro ? h("p", { class: "muted small" }, intro) : "",
      extra || "",
      ...body,
      h("div", { class: "row", style: "margin-top:10px" }, h("button", { class: "primary", onclick: () => finish(boxes.filter((b) => b.checked).map((b) => b.value)) }, okLabel), h("button", { onclick: () => finish(null) }, "Cancel")),
    );
    dlg.addEventListener("close", () => { finish(null); dlg.remove(); });
    document.body.append(dlg);
    dlg.showModal();
  });
}
/** Roll: a target and which pools move to it. */
async function rollDialog(c, post) {
  const target = h("input", { value: c.spec.image.channel || "stable", "aria-label": "Roll target", placeholder: "channel, git sha or image" });
  const pools = Object.keys(c.state.workers || {}).filter((p) => (c.state.workers[p] || []).length);
  const groups = [{ label: "Pools (new workers come up beside the old ones, then the old ones drain)", items: pools.map((p) => ({ value: p, label: p, checked: true, note: `${c.state.workers[p].length} worker(s)` })) }];
  const sel = await pickDialog(`Roll ${c.name}`, "Target: a channel (stable, latest), a git sha or an image.", groups, "Roll", h("label", {}, "Target ", target));
  if (!sel) return;
  if (!sel.length) return toast("nothing chosen");
  if (!target.value.trim()) return toast("no target");
  post("roll", { target: target.value.trim(), pools: sel }, "roll");
}
/** Restart: whole pools or single pods, whether or not their env changed. */
async function restartDialog(c, pods, post) {
  const live = pods.filter((p) => p.slot !== "retired");
  const poolIds = [...new Set(live.filter((p) => p.role === "worker").map((p) => p.pool))];
  const groups = [
    { label: "Whole pools", items: poolIds.map((p) => ({ value: `pool:${p}`, label: p, note: `${live.filter((x) => x.pool === p).length} pod(s)` })) },
    { label: "Single pods", items: live.map((p) => ({ value: `pod:${p.pod_id}`, label: p.pod_id, note: `${p.role}${p.pool ? ` ${p.pool}` : ""}` })) },
  ];
  const sel = await pickDialog(`Restart pods of ${c.name}`, "Rolling: one pod at a time; each must come back before the next. The chosen pods restart even when their env did not change (Env → Apply restarts only the ones that need it).", groups, "Restart");
  if (!sel) return;
  const body = { pools: sel.filter((v) => v.startsWith("pool:")).map((v) => v.slice(5)), pods: sel.filter((v) => v.startsWith("pod:")).map((v) => v.slice(4)) };
  if (!body.pools.length && !body.pods.length) return toast("nothing chosen");
  post("restart", body, "restart");
}
/** Add a pool preset to a defined cluster's spec (the document API: versioned, validated, audited). */
async function addPoolDialog(c, presets, redraw) {
  const dlg = h("dialog", {});
  dlg.append(
    h("h2", {}, `Add a pool to ${c.name}`),
    h("p", { class: "muted small" }, "The pool is added to the spec with its preset's count; a running cluster creates its workers on Scale."),
    presetPicker(presets.filter((p) => !c.spec.pools.some((x) => x.id === p.pool.id)), async (pool) => {
      const d = await api(`/api/docs/cluster-spec/${c.id}`);
      const doc = { ...d.doc, pools: [...d.doc.pools, pool] };
      await act("add pool", () => api(`/api/docs/cluster-spec/${c.id}`, { method: "PUT", body: { doc, version: d.version } }));
      dlg.close();
      redraw();
    }),
    h("div", { class: "row" }, h("button", { onclick: () => dlg.close() }, "Close")),
  );
  dlg.addEventListener("close", () => dlg.remove());
  document.body.append(dlg);
  dlg.showModal();
}
async function startDialog(c, redraw) {
  const dlg = h("dialog", {});
  document.body.append(dlg);
  dlg.append(h("h2", {}, `Start ${c.name}`), h("p", { class: "muted small" }, "Price check…"));
  dlg.showModal();
  try {
    const p = await api(`/api/clusters/${c.id}/price`, { method: "POST", body: {} });
    dlg.replaceChildren(
      h("h2", {}, `Start ${c.name}`),
      table([{ label: "pod", get: (x) => `${x.role} ${x.pool || ""}` }, { label: "what", get: (x) => x.what }, { label: "$/hr (est.)", get: (x) => fmt$(x.dph), num: true }], p.pods),
      h("dl", { class: "kv", style: "margin-top:10px" }, h("dt", {}, "cluster"), h("dd", {}, `${fmt$(p.cluster_dph)}/hr`), h("dt", {}, "account now"), h("dd", {}, `${fmt$(p.balance)} at ${fmt$(p.account_spend_per_hr)}/hr`), h("dt", {}, `after ${p.hours.toFixed(2)} h`), h("dd", {}, `${fmt$(p.projected_balance)} (floor ${fmt$(p.floor, 0)})`)),
      p.ok ? h("p", { class: "small" }, badge("within the floor", "good")) : h("div", {}, p.reasons.map((x) => h("p", { class: "small" }, badge("refused", "critical"), " ", x))),
      h(
        "div",
        { class: "row", style: "margin-top:10px" },
        h("button", { class: "primary", disabled: !p.ok, onclick: async () => { await act("start", () => api(`/api/clusters/${c.id}/start`, { method: "POST", body: {} })); dlg.close(); redraw(); } }, "Start"),
        h("button", { onclick: () => dlg.close() }, "Cancel"),
      ),
    );
  } catch (e) {
    dlg.replaceChildren(h("p", {}, e.message), h("button", { onclick: () => dlg.close() }, "Close"));
  }
  dlg.addEventListener("close", () => dlg.remove());
}

// ---------------------------------------------------------------- env
async function pageEnv(main) {
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const clusters = (await api("/api/clusters")).clusters;
  const cid = q.get("cluster") || clusters[0]?.id || "";
  const sel = h("select", { "aria-label": "Cluster", onchange: () => { location.hash = `#/env?cluster=${sel.value}`; } }, clusters.map((c) => h("option", { value: c.id, selected: c.id === cid }, c.name)));
  const parts = [
    h("h1", {}, "Environment"),
    h("p", { class: "muted small" }, "Resolution: pod > pool > cluster > account > the controller's own keys (which cannot be overridden). A pool's variables reach every worker of that pool (restarts, rolls and scale-ups included). Secrets are write-only: they show as ••••, and a new value goes in the Form tab's password field. Runpod applies env only on a restart (PATCH): the effective view marks the pods that need one. Key suggestions include the engine's own switches (FASTVIDEO_*, FV_LONGLIVE_*) with what they do."),
    card("Account (every controller cluster)", docPanel("env", "account", { title: "env: account", onSaved: () => refreshEff() })),
  ];
  let refreshEff = async () => {};
  if (cid) {
    const eff = await api(`/api/clusters/${cid}/env`);
    parts.push(h("div", { class: "row", style: "margin-bottom:12px" }, h("label", {}, "Cluster", sel)));
    parts.push(card("Cluster", docPanel("env", `cluster:${cid}`, { title: "env: cluster", cluster: cid, onSaved: () => refreshEff() })));
    const cl = clusters.find((x) => x.id === cid);
    const poolIds = (cl?.spec.pools || []).map((p) => p.id);
    if (poolIds.length) {
      const pool = poolIds.includes(q.get("pool")) ? q.get("pool") : poolIds[0];
      const psel = h("select", { "aria-label": "Pool", onchange: () => { location.hash = `#/env?cluster=${cid}&pool=${psel.value}`; } }, poolIds.map((p) => h("option", { value: p, selected: p === pool }, p)));
      parts.push(card("Pool (every worker of one pool)", h("div", { class: "row", style: "margin-bottom:8px" }, h("label", {}, "Pool", psel)), docPanel("env", `pool:${cid}:${pool}`, { title: `env: pool ${pool}`, cluster: cid, onSaved: () => refreshEff() })));
    }
    const effBox = h("div", {});
    refreshEff = async () => effBox.replaceChildren(effView(await api(`/api/clusters/${cid}/env`), cid));
    effBox.replaceChildren(effView(eff, cid));
    parts.push(card("Effective env per pod", effBox));
  } else parts.push(card("Clusters", h("p", { class: "muted small" }, "Define a cluster to set cluster and pod env.")));
  main.replaceChildren(...parts);
}
function effView(eff, cid) {
  const needs = eff.needs_restart;
  const head = needs.length
    ? h("div", { class: "row", style: "margin-bottom:8px" }, badge(`${needs.length} pod(s) need a restart`, "warn"), h("button", { class: "primary", onclick: async () => { if (confirm(`Rolling restart of ${needs.join(", ")}? One pod at a time; each must come back before the next.`)) { await act("restart", () => api(`/api/clusters/${cid}/restart`, { method: "POST", body: {} })); location.hash = `#/cluster/${cid}`; } } }, "Apply with a rolling restart"))
    : h("p", { class: "small" }, badge("every pod has its env", "good"));
  if (!eff.pods.length) return h("div", {}, head, h("p", { class: "muted small" }, "No pods running; what a new pod would get:"), Object.entries(eff.preview).map(([k, env]) => h("details", {}, h("summary", {}, k), envTable(env))));
  return h(
    "div",
    {},
    head,
    h("p", { class: "muted small" }, "Highlighted: a value that overrides a lower level. Marked at the left: a conflict (two of your levels set the key to different values)."),
    eff.pods.map((p) => {
      const conflicts = p.env.filter((v) => (v.overrides || []).some((x) => x !== "system")).length;
      const podHost = h("div", {});
      const det = h(
        "details",
        {},
        h("summary", {}, `${p.role} ${p.pool || ""} ${p.pod_id} `, p.needs_restart ? badge("needs restart", "warn") : badge("applied", "good"), conflicts ? " " : "", conflicts ? badge(`${conflicts} conflict(s)`, "serious") : ""),
        envTable(p.env),
        h("h3", {}, "Pod-level variables"),
        podHost,
      );
      det.addEventListener("toggle", () => { if (det.open && !podHost.childElementCount) podHost.append(docPanel("env", `pod:${p.pod_id}`, { title: `env: pod ${p.pod_id}`, cluster: cid })); });
      return det;
    }),
  );
}
function envTable(env) {
  const tb = table(
    [
      { label: "name", get: (x) => h("code", {}, x.key) },
      { label: "value", get: (x) => (x.secret ? h("span", { class: "muted" }, "•••••••• secret") : x.runpod_secret_ref ? h("span", {}, h("code", {}, x.value), " ", badge("Runpod secret")) : h("code", {}, x.value)), wrap: true },
      { label: "source", get: (x) => h("span", {}, badge(x.source, x.source === "pod" ? "serious" : x.source === "pool" ? "warn" : x.source === "cluster" ? "warn" : x.source === "account" ? "good" : ""), x.overrides ? h("span", { class: "muted small" }, ` over ${x.overrides.join(", ")}`) : "") },
    ],
    env,
  );
  // Rows: highlight overrides, mark conflicts between user levels.
  const rows = tb.querySelectorAll("tbody tr");
  env.forEach((x, i) => {
    if (x.overrides) rows[i]?.classList.add("env-override");
    if ((x.overrides || []).some((y) => y !== "system")) rows[i]?.classList.add("env-conflict");
  });
  return tb;
}

// ---------------------------------------------------------------- logs
async function pageLogs(main) {
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const pods = (await api("/api/pods?all=1")).pods;
  let pod = q.get("pod") || "";
  const podSel = h("select", { "aria-label": "Pod", onchange: () => { location.hash = `#/logs?pod=${podSel.value}`; } }, h("option", { value: "" }, "choose a pod"), pods.map((p) => h("option", { value: p.pod_id, selected: p.pod_id === pod }, `${p.name || p.pod_id} (${p.owner})`)));
  const level = h("select", { "aria-label": "Level" }, ["trace", "debug", "info", "warn", "error"].map((l) => h("option", { value: l, selected: l === "info" }, l)));
  const text = h("input", { placeholder: "search (job_id, text…)", "aria-label": "Search" });
  const out = h("div", { class: "log", id: "logOut" });
  const status = h("span", { class: "muted small" });
  const tabs = h("div", { class: "tabs" });
  let mode = "shipped";
  let lastId = 0;
  const line = (l) => h("div", { class: `lv-${l.level}` }, `${new Date(l.ts).toISOString().slice(0, 23)} ${l.level.toUpperCase().padEnd(5)} ${l.target ? l.target + ": " : ""}${l.msg}${l.fields ? " " + JSON.stringify(l.fields) : ""}`);
  const load = async () => {
    if (!pod) { out.replaceChildren(h("div", { class: "muted" }, "Pick a pod.")); return; }
    if (mode === "runpod") {
      const r = await api(`/api/pods/${pod}/runpod-logs`);
      out.replaceChildren(...r.container.map((x) => h("div", {}, x)), h("div", { class: "muted" }, "— system —"), ...r.system.map((x) => h("div", { class: "muted" }, x)));
      status.textContent = `Runpod's own log tail (${r.container.length} lines; not searchable, no history).`;
      return;
    }
    const p = new URLSearchParams({ pod, level: level.value, limit: "500" });
    if (text.value) p.set("q", text.value);
    const r = await api(`/api/logs?${p}`);
    lastId = r.lines.length ? r.lines[r.lines.length - 1].id : 0;
    out.replaceChildren(...r.lines.map(line));
    out.scrollTop = out.scrollHeight;
    status.textContent = `${r.lines.length} lines (D1 tail, last 24 h)`;
  };
  const tail = () => {
    if (state.ws) { state.ws.close(); state.ws = null; tailBtn.textContent = "Live tail"; return; }
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/logs/tail?pod=${encodeURIComponent(pod)}`);
    state.ws = ws;
    tailBtn.textContent = "Stop tail";
    ws.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        const order = ["trace", "debug", "info", "warn", "error"];
        for (const l of m.lines) if (order.indexOf(l.level) >= order.indexOf(level.value) && (!text.value || JSON.stringify(l).includes(text.value))) out.append(line(l));
        out.scrollTop = out.scrollHeight;
      } catch {}
    };
    ws.onclose = () => { tailBtn.textContent = "Live tail"; status.textContent = "tail closed"; };
    ws.onopen = () => { status.textContent = "live"; };
    state.timers.push(setInterval(() => ws.readyState === 1 && ws.send("ping"), 25000));
  };
  const tailBtn = h("button", { onclick: tail, disabled: !pod }, "Live tail");
  for (const [k, label] of [["shipped", "Shipped (fv-serve)"], ["runpod", "Runpod container log"]]) tabs.append(h("button", { class: mode === k ? "on" : "", onclick: (ev) => { mode = k; for (const b of tabs.children) b.classList.remove("on"); ev.target.classList.add("on"); load(); } }, label));
  main.replaceChildren(
    h("h1", {}, "Logs"),
    card(
      null,
      h("div", { class: "row", style: "margin-bottom:10px" }, podSel, level, text, h("button", { onclick: load }, "Search"), tailBtn, pod && h("a", { class: "btn", href: `/api/logs/download?pod=${encodeURIComponent(pod)}&day=${new Date().toISOString().slice(0, 10)}` }, "Download today"), status),
      tabs,
      out,
    ),
  );
  text.addEventListener("keydown", (e) => e.key === "Enter" && load());
  level.addEventListener("change", load);
  await load().catch((e) => (status.textContent = e.message));
}

// ---------------------------------------------------------------- costs
async function pageCosts(main) {
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const days = Number(q.get("days") || 7);
  const c = await api(`/api/costs?days=${days}`);
  const owners = c.by_owner.map((r) => r.owner);
  const top = owners.slice(0, 7);
  const cats = c.by_day.map((r) => r.day);
  const stacks = top.map((ow) => ({ name: ow, values: cats.map((d) => c.by_day_owner.filter((r) => r.day === d && r.owner === ow).reduce((a, r) => a + r.usd, 0)) }));
  if (owners.length > 7) stacks.push({ name: "other", color: "var(--text-muted)", values: cats.map((d) => c.by_day_owner.filter((r) => r.day === d && !top.includes(r.owner)).reduce((a, r) => a + r.usd, 0)) });
  main.replaceChildren(
    h("h1", {}, "Costs"),
    h("div", { class: "row", style: "margin-bottom:12px" }, [1, 7, 30, 90].map((d) => h("a", { class: "btn", href: `#/costs?days=${d}`, style: d === days ? "font-weight:700" : "" }, d === 1 ? "today" : `${d} days`))),
    card(`Spend per day by owner (since ${c.since})`, stackedBars({ cats, stacks, height: 240 })),
    h(
      "div",
      { class: "grid g2" },
      card("By owner", table([{ label: "owner", get: (r) => r.owner }, { label: "cost", get: (r) => fmt$(r.usd), num: true }, { label: "pod-hours", get: (r) => fmtN(r.minutes / 60, 1), num: true }, { label: "idle %", get: (r) => (r.minutes ? fmtPct((100 * r.idle_minutes) / r.minutes) : "–"), num: true }], c.by_owner)),
      card("By cluster", table([{ label: "cluster", get: (r) => r.name || r.cluster_id }, { label: "cost", get: (r) => fmt$(r.usd), num: true }, { label: "pod-hours", get: (r) => fmtN(r.minutes / 60, 1), num: true }], c.by_cluster, "No controller cluster has run in this range.")),
    ),
    card("By pod", table([{ label: "pod", get: (r) => h("a", { href: `#/pod/${r.pod_id}` }, r.name || r.pod_id) }, { label: "owner", get: (r) => r.owner }, { label: "cost", get: (r) => fmt$(r.usd), num: true }, { label: "hours", get: (r) => fmtN(r.minutes / 60, 1), num: true }, { label: "idle h", get: (r) => fmtN(r.idle_minutes / 60, 1), num: true }], c.by_pod)),
    h("p", { class: "muted small" }, "Cost accrues each minute for running pods at their Runpod $/hr. Stopped pods' volume storage is not counted."),
  );
}

// ---------------------------------------------------------------- releases
async function pageReleases(main) {
  const [rel, ci] = await Promise.all([api("/api/releases"), api("/api/github/ci").catch((e) => ({ error: e.message }))]);
  const target = h("input", { placeholder: "git sha, digest or tag", "aria-label": "Target" });
  const channel = h("input", { value: "stable", "aria-label": "Channel", size: 8 });
  const dry = h("input", { type: "checkbox", checked: true, "aria-label": "Dry run" });
  const dispatch = async (action) => {
    const body = { action, channel: channel.value, target: target.value, dry_run: dry.checked };
    if (!confirm(`${action} ${action === "promote" ? target.value + " to " : ""}${channel.value}${dry.checked ? " (dry run)" : ""}? This dispatches release.yml.`)) return;
    const r = await act(action, () => api("/api/github/release", { method: "POST", body }));
    toast(`dispatched: see ${r.runs_url}`);
  };
  main.replaceChildren(
    h("h1", {}, "Releases"),
    h(
      "div",
      { class: "grid g2" },
      card(
        "Channels",
        rel.available ? table([{ label: "channel", get: (r) => r.channel }, { label: "sha", get: (r) => h("code", {}, r.git_sha.slice(0, 7)) }, { label: "action", get: (r) => r.action }, { label: "when", get: (r) => ago(r.promoted_at) }, { label: "by", get: (r) => r.promoted_by }], rel.heads) : h("p", { class: "muted small" }, "fv-jobs is not bound (JOBS_DB)."),
        h("h3", {}, "Promote / roll back (release.yml)"),
        h("div", { class: "row" }, target, channel, h("label", { style: "flex-direction:row;align-items:center;gap:5px" }, dry, "dry run"), h("button", { class: "primary", onclick: () => dispatch("promote") }, "Promote"), h("button", { onclick: () => dispatch("rollback") }, "Roll back")),
      ),
      card(
        "CI on main",
        ci.error ? h("p", { class: "muted small" }, ci.error) : table([{ label: "workflow", get: (w) => h("a", { href: w.html_url, target: "_blank", rel: "noopener" }, w.name) }, { label: "result", get: (w) => badge(w.conclusion || w.status, w.conclusion === "success" ? "good" : w.conclusion === "failure" ? "critical" : "warn") }, { label: "sha", get: (w) => h("code", {}, w.head_sha) }, { label: "when", get: (w) => ago(Date.parse(w.created_at)) }], ci.main),
      ),
    ),
    card("Cluster drift", table([{ label: "cluster", get: (d) => d.cluster }, { label: "channel", get: (d) => d.channel || "–" }, { label: "running", get: (d) => d.running_sha || "?" }, { label: "head", get: (d) => d.head_sha || "?" }, { label: "", get: (d) => (d.drift ? badge("drift", "warn") : badge("in step", "good")) }], rel.drift, "No clusters.")),
    card("History", table([{ label: "id", get: (r) => r.id, num: true }, { label: "channel", get: (r) => r.channel }, { label: "sha", get: (r) => h("code", {}, r.git_sha.slice(0, 7)) }, { label: "action", get: (r) => r.action }, { label: "when", get: (r) => dt(r.promoted_at) }, { label: "by", get: (r) => r.promoted_by }, { label: "rolled back", get: (r) => (r.rolled_back_at ? "yes" : "") }], rel.history)),
    card("Release records (JSON)", jsonTree({ heads: rel.heads, history: rel.history }, { open: 1 })),
    card("Deployment registry (fv-jobs)", table([{ label: "resource", get: (r) => r.id }, { label: "pool", get: (r) => r.pool || "" }, { label: "variant", get: (r) => r.variant || "" }, { label: "sha", get: (r) => (r.git_sha || "").slice(0, 7) }, { label: "status", get: (r) => r.status }, { label: "by", get: (r) => r.created_by }, { label: "created", get: (r) => ago(r.created_at) }], rel.registry.slice(0, 50))),
  );
}

// ---------------------------------------------------------------- settings
async function pageSettings(main) {
  const [tok, aud] = await Promise.all([api("/api/tokens"), api("/api/audit?limit=100")]);
  const tName = h("input", { placeholder: "token name", "aria-label": "Token name" });
  const tScope = h("select", { "aria-label": "Scope" }, h("option", { value: "read" }, "read"), h("option", { value: "admin" }, "admin"));
  const tTtl = h("input", { type: "number", min: 1, max: 365, value: 90, "aria-label": "Days", style: "width:80px" });
  const tOut = h("pre", { class: "log", hidden: true });
  const auditView = h("div", {});
  const parse = (x) => { try { return JSON.parse(x); } catch { return x; } };
  main.replaceChildren(
    h("h1", {}, "Settings"),
    card("Alert policies", h("p", { class: "muted small" }, "Auto-actions touch controller clusters only, with one exception: the build pod backstop (build_pod_*) stops the shared build pod (owner external:build-pod) when its own self-stop did not happen. No other external pod is touched. The deadline backstop always applies."), docPanel("policies", "default", { title: "policies" })),
    card("External pod attribution (name prefix → owner; first match wins)", docPanel("attribution", "default", { title: "attribution rules" })),
    card(
      "API tokens (scripts/serve/fv-control.sh, agents)",
      table([{ label: "name", get: (t) => t.name }, { label: "scope", get: (t) => t.scope }, { label: "created", get: (t) => ago(t.created_at) }, { label: "last used", get: (t) => ago(t.last_used_at) }, { label: "expires", get: (t) => dt(t.expires_at) }, { label: "", get: (t) => (t.revoked_at ? "revoked" : h("button", { class: "ghost danger", onclick: async () => { await act("revoke", () => api(`/api/tokens/${t.id}`, { method: "DELETE" })); route(); } }, "Revoke")) }], tok.tokens, "No tokens."),
      h("div", { class: "row", style: "margin-top:8px" }, tName, tScope, h("label", { style: "flex-direction:row;align-items:center;gap:4px" }, tTtl, "days"), h("button", { onclick: async () => { const j = await act("mint", () => api("/api/tokens", { method: "POST", body: { name: tName.value, scope: tScope.value, ttl_days: Number(tTtl.value) } })); tOut.hidden = false; tOut.textContent = `${j.token}\n(shown once; expires ${dt(j.expires_at)})`; } }, "Mint token")),
      h("p", { class: "muted small" }, "read: GET only · admin: everything but minting tokens (validated against the token-create schema)."),
      tOut,
    ),
    card(
      "Audit log",
      h("p", { class: "muted small" }, "Click an entry to inspect its before / after."),
      table([{ label: "when", get: (a) => h("a", { href: "#", onclick: (e) => { e.preventDefault(); auditView.replaceChildren(h("h3", {}, `#${a.id} ${a.action} ${a.target || ""}`), jsonTree({ ...a, before: parse(a.before), after: parse(a.after) }, { open: 2 })); auditView.scrollIntoView({ block: "nearest" }); } }, dt(a.at)) }, { label: "who", get: (a) => a.actor }, { label: "action", get: (a) => a.action }, { label: "target", get: (a) => a.target || "" }, { label: "ok", get: (a) => (a.ok ? "" : badge("failed", "critical")) }, { label: "detail", get: (a) => (a.detail || "").slice(0, 120), wrap: true }], aud.audit),
      auditView,
    ),
  );
}
