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
function every(ms, fn) {
  state.timers.push(setInterval(fn, ms));
}

// ---------------------------------------------------------------- dashboard
async function pageDashboard(main) {
  const draw = async () => {
    const [o, series, bal, costs] = await Promise.all([api("/api/overview"), api("/api/metrics/series?hours=6"), api("/api/balance?hours=24"), api("/api/costs?days=7")]);
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
    const alerts = card(
      "Alerts",
      o.alerts.length
        ? o.alerts.map((a) => h("div", { class: `alert sev-${a.severity}` }, h("span", { class: "icon", "aria-hidden": "true" }, a.severity === "critical" ? "!" : a.severity === "warn" ? "▲" : "i"), h("div", {}, h("div", {}, h("b", {}, a.severity), " ", a.message), h("div", { class: "muted small" }, `${a.kind} · since ${ago(a.opened_at)}${a.action ? ` · action: ${a.action}` : ""}`))))
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
      h("div", { class: "grid g2" }, card("Balance, 24 h", balChart), card("Burn $/hr, 24 h", burnChart)),
      h("div", { class: "grid g2" }, card("GPU utilisation, 6 h (top pods by $/hr)", gpuChart), card("CPU utilisation, 6 h", cpuChart)),
      h("div", { class: "grid g2" }, card("Spend per day by owner, 7 days", stackedBars({ cats: days, stacks })), idle),
    );
  };
  await draw();
  every(60_000, () => draw().catch(() => {}));
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
  );
}

// ---------------------------------------------------------------- clusters
async function pageClusters(main) {
  const [r, tpl] = await Promise.all([api("/api/clusters"), api("/api/templates")]);
  const ta = h("textarea", { "aria-label": "Cluster spec (JSON)" }, JSON.stringify({ ...tpl["tiny-cpu"], name: "tiny" }, null, 2));
  const tplSel = h(
    "select",
    { onchange: () => { ta.value = JSON.stringify({ ...tpl[tplSel.value], name: tplSel.value === "tiny-cpu" ? "tiny" : "main" }, null, 2); } },
    h("option", { value: "tiny-cpu" }, "tiny-cpu: CPU gateway + 1 fake-engine CPU worker"),
    h("option", { value: "standard" }, "standard: CPU gateway + h3-turbo, h3-max, ltx, wan GPU pools"),
  );
  const imp = h("textarea", { placeholder: "Paste artifacts/runpod/serve/cluster.json here", style: "min-height:120px", "aria-label": "Script state" });
  const pem = h("textarea", { placeholder: "Optional: cluster.json.admin-key.pem", style: "min-height:60px", "aria-label": "Admin key" });
  const impName = h("input", { placeholder: "name", "aria-label": "Imported cluster name" });
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
        ta,
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
      card(
        "Import a runpod-cluster.sh cluster",
        h("p", { class: "muted small" }, "The state file holds the cluster's internal token; the controller seals it in D1. Without the key pair it can still use a stored admin token."),
        impName, imp, pem,
        h("div", { class: "row", style: "margin-top:8px" }, h("button", {
          onclick: async () => {
            let st;
            try { st = JSON.parse(imp.value); } catch { return toast("state: not JSON"); }
            const j = await act("import", () => api("/api/clusters/import", { method: "POST", body: { name: impName.value || undefined, state: st, admin_key_pem: pem.value || undefined } }));
            location.hash = `#/cluster/${j.cluster.id}`;
          },
        }, "Import")),
      ),
    ),
  );
}

async function pageCluster(main, id) {
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
      running && c.state.gateway && btn(c.state.gateway_stopped ? "Start gateway" : "Stop gateway", () => post(c.state.gateway_stopped ? "gateway/start" : "gateway/stop", {}, "gateway")),
      running && !c.state.gateway && c.spec.gateway.enabled && btn("Start gateway", () => post("gateway/start", {}, "gateway")),
      running && btn("Roll to…", () => { const t = prompt("Target: a channel (stable, latest), a git sha or an image", c.spec.image.channel || "stable"); if (t) post("roll", { target: t, gateway: confirm("Move the gateway to the same build too?") }, "roll"); }),
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
    const gw = c.state.gateway_url;
    const tools = card(
      "Gateway",
      gw ? h("p", { class: "small" }, "URL ", h("a", { href: gw, target: "_blank", rel: "noopener" }, gw), " · ", h("a", { href: `${gw}/console`, target: "_blank", rel: "noopener" }, "console")) : h("p", { class: "muted small" }, "No gateway pod."),
      gw && h(
        "div",
        { class: "row" },
        h("button", { onclick: async () => { const j = await act("status", () => api(`/api/clusters/${c.id}/gateway`)); $("#gwOut").textContent = JSON.stringify({ status: j.status, pools: j.pools }, null, 2); } }, "Pools view"),
        h("button", { onclick: async () => { if (!confirm("Show the gateway's admin token? (audited)")) return; const j = await act("admin token", () => api(`/api/clusters/${c.id}/admin-token`, { method: "POST" })); $("#gwOut").textContent = `admin token: ${j.admin_token}\nconsole: ${j.console}`; } }, "Reveal admin token"),
        h("button", { onclick: async () => { const n = prompt("Name of the new user API key", "laptop"); if (!n) return; const j = await act("mint", () => api(`/api/clusters/${c.id}/mint-key`, { method: "POST", body: { name: n } })); $("#gwOut").textContent = `API key (shown once): ${j.api_key}`; } }, "Mint user API key"),
      ),
      h("pre", { id: "gwOut", class: "log", style: "margin-top:8px;max-height:300px" }),
    );
    main.replaceChildren(
      h("h1", {}, c.name, " ", badge(c.status, statusKind(c.status))),
      h("p", { class: "muted small" }, `deadline ${dt(c.deadline)} (${until(c.deadline)}) · floor $${c.spec.balance_floor} · pod watchdog below $${c.spec.min_balance} · ${c.source}`, op ? ` · running: ${op.kind} (${op.phase})` : ""),
      card(null, actions),
      h("div", { class: "grid g2" }, pools, drift),
      pods,
      h("div", { class: "grid g2" }, opsCard, tools),
      card("Environment", h("p", { class: "small" }, h("a", { href: `#/env?cluster=${c.id}` }, "Cluster and pod env, effective values, restarts →"))),
      card("Spec", h("pre", { class: "log" }, JSON.stringify(c.spec, null, 2))),
    );
    if (op) showOp(r.ops.find((o) => o.status === "running")?.id);
  };
  await draw();
  every(10_000, () => draw().catch(() => {}));
}
async function showOp(id) {
  if (!id) return;
  const j = await api(`/api/ops/${id}`);
  const el = $("#opLog");
  if (!el) return;
  el.replaceChildren(h("h3", {}, `${j.operation.kind} ${j.operation.id}`), h("div", { class: "log" }, j.operation.log.map((l) => h("div", {}, `${new Date(l.at).toISOString().slice(11, 19)}  ${l.msg}`))));
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
  const acc = await api("/api/env/account");
  const editor = (scope, sid, vars, title, hint) => {
    const k = h("input", { placeholder: "NAME", "aria-label": "Variable name" });
    const v = h("input", { placeholder: "value", "aria-label": "Value" });
    const sec = h("input", { type: "checkbox", "aria-label": "Secret" });
    const base = scope === "account" ? "/api/env/account" : `/api/env/${scope}/${encodeURIComponent(sid)}`;
    return card(
      title,
      hint && h("p", { class: "muted small" }, hint),
      table(
        [
          { label: "name", get: (x) => h("code", {}, x.key) },
          { label: "value", get: (x) => (x.secret ? h("span", { class: "muted" }, "•••••••• (secret)") : h("code", {}, x.value)), wrap: true },
          { label: "by", get: (x) => `${x.updated_by}, ${ago(x.updated_at)}` },
          { label: "", get: (x) => h("button", { class: "ghost danger", onclick: async () => { if (confirm(`Delete ${x.key}?`)) { await act("delete", () => api(`${base}/${encodeURIComponent(x.key)}`, { method: "DELETE" })); route(); } } }, "Delete") },
        ],
        vars,
        "No variables at this level.",
      ),
      h(
        "div",
        { class: "row", style: "margin-top:8px" },
        k, v, h("label", { style: "flex-direction:row;align-items:center;gap:5px" }, sec, "secret"),
        h("button", { class: "primary", onclick: async () => { await act("set", () => api(`${base}/${encodeURIComponent(k.value.trim())}`, { method: "PUT", body: { value: v.value, secret: sec.checked } })); route(); } }, "Set"),
      ),
    );
  };
  const parts = [h("h1", {}, "Environment"), h("p", { class: "muted small" }, "Resolution: pod > cluster > account > the controller's own keys (which cannot be overridden). Runpod applies an env change only on a restart (PATCH): the table below marks the pods that need one."), editor("account", "", acc.vars, "Account (every controller cluster)")];
  const sel = h("select", { onchange: () => { location.hash = `#/env?cluster=${sel.value}`; } }, clusters.map((c) => h("option", { value: c.id, selected: c.id === cid }, c.name)));
  if (cid) {
    const [cv, eff] = await Promise.all([api(`/api/env/cluster/${cid}`), api(`/api/clusters/${cid}/env`)]);
    parts.push(h("div", { class: "row", style: "margin-bottom:12px" }, h("label", {}, "Cluster", sel)));
    parts.push(editor("cluster", cid, cv.vars, "Cluster"));
    const needs = eff.needs_restart;
    parts.push(
      card(
        "Effective env per pod",
        needs.length
          ? h("div", { class: "row", style: "margin-bottom:8px" }, badge(`${needs.length} pod(s) need a restart`, "warn"), h("button", { class: "primary", onclick: async () => { if (confirm(`Rolling restart of ${needs.join(", ")}? Workers first, the gateway last; each must come back before the next.`)) { await act("restart", () => api(`/api/clusters/${cid}/restart`, { method: "POST", body: {} })); location.hash = `#/cluster/${cid}`; } } }, "Apply with a rolling restart"))
          : h("p", { class: "small" }, badge("every pod has its env", "good")),
        eff.pods.length ? eff.pods.map((p) => h("details", {}, h("summary", {}, `${p.role} ${p.pool || ""} ${p.pod_id} `, p.needs_restart ? badge("needs restart", "warn") : ""), envTable(p.env), h("div", { class: "muted small" }, "Pod-level variables:"), podEnvEditor(p.pod_id))) : h("p", { class: "muted small" }, "No pods running; what a new pod would get:"),
        !eff.pods.length && Object.entries(eff.preview).map(([k, env]) => h("details", {}, h("summary", {}, k), envTable(env))),
      ),
    );
  } else parts.push(card("Clusters", h("p", { class: "muted small" }, "Define a cluster to set cluster and pod env.")));
  main.replaceChildren(...parts);
  async function fillPod(el, pod) {
    const pv = await api(`/api/env/pod/${pod}`);
    el.replaceChildren(editor("pod", pod, pv.vars, null));
  }
  function podEnvEditor(pod) {
    const el = h("div", {});
    fillPod(el, pod).catch(() => {});
    return el;
  }
}
function envTable(env) {
  return table(
    [
      { label: "name", get: (x) => h("code", {}, x.key) },
      { label: "value", get: (x) => (x.secret ? h("span", { class: "muted" }, "•••••••• secret") : x.runpod_secret_ref ? h("span", {}, h("code", {}, x.value), " ", badge("Runpod secret")) : h("code", {}, x.value)), wrap: true },
      { label: "source", get: (x) => `${x.source}${x.overrides ? ` (over ${x.overrides.join(", ")})` : ""}` },
    ],
    env,
  );
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
    card("Deployment registry (fv-jobs)", table([{ label: "resource", get: (r) => r.id }, { label: "pool", get: (r) => r.pool || "" }, { label: "variant", get: (r) => r.variant || "" }, { label: "sha", get: (r) => (r.git_sha || "").slice(0, 7) }, { label: "status", get: (r) => r.status }, { label: "by", get: (r) => r.created_by }, { label: "created", get: (r) => ago(r.created_at) }], rel.registry.slice(0, 50))),
  );
}

// ---------------------------------------------------------------- settings
async function pageSettings(main) {
  const [pol, tok, aud] = await Promise.all([api("/api/policies"), api("/api/tokens"), api("/api/audit?limit=100")]);
  const p = pol.policies;
  const fields = {};
  const num = (k, label) => h("label", {}, label, (fields[k] = h("input", { type: "number", step: "any", value: p[k], style: "width:110px" })));
  const bool = (k, label) => h("label", { style: "flex-direction:row;align-items:center;gap:6px" }, (fields[k] = h("input", { type: "checkbox", checked: !!p[k] })), label);
  const tName = h("input", { placeholder: "token name", "aria-label": "Token name" });
  const tScope = h("select", { "aria-label": "Scope" }, h("option", { value: "read" }, "read"), h("option", { value: "admin" }, "admin"));
  const tOut = h("pre", { class: "log", hidden: true });
  main.replaceChildren(
    h("h1", {}, "Settings"),
    card(
      "Alert policies",
      h("div", { class: "row" }, num("idle_gpu_pct", "idle: GPU below %"), num("idle_min", "idle alert after min"), num("cluster_dph_max", "cluster $/hr above"), num("daily_spend_max", "daily spend above $"), num("balance_margin", "balance margin over floor $"), num("pod_down_min", "pod down after min")),
      h("div", { class: "row", style: "margin-top:10px" }, bool("auto_stop_idle", "Auto-remove idle workers of controller clusters after"), num("auto_stop_idle_min", "min"), bool("stop_on_floor", "Stop controller clusters below the balance floor")),
      h("p", { class: "muted small" }, "Auto-actions touch controller clusters only, never external pods. The deadline backstop always applies."),
      h("button", { class: "primary", onclick: async () => { const b = {}; for (const [k, el] of Object.entries(fields)) b[k] = el.type === "checkbox" ? el.checked : Number(el.value); await act("policies", () => api("/api/policies", { method: "PUT", body: b })); } }, "Save"),
      h("h3", {}, "External pod attribution (name prefix → owner)"),
      table([{ label: "prefix", get: (r) => h("code", {}, r.prefix) }, { label: "owner", get: (r) => r.owner }], p.attribution),
    ),
    card(
      "API tokens (scripts/serve/fv-control.sh, agents)",
      table([{ label: "name", get: (t) => t.name }, { label: "scope", get: (t) => t.scope }, { label: "created", get: (t) => ago(t.created_at) }, { label: "last used", get: (t) => ago(t.last_used_at) }, { label: "expires", get: (t) => dt(t.expires_at) }, { label: "", get: (t) => (t.revoked_at ? "revoked" : h("button", { class: "ghost danger", onclick: async () => { await act("revoke", () => api(`/api/tokens/${t.id}`, { method: "DELETE" })); route(); } }, "Revoke")) }], tok.tokens, "No tokens."),
      h("div", { class: "row", style: "margin-top:8px" }, tName, tScope, h("button", { onclick: async () => { const j = await act("mint", () => api("/api/tokens", { method: "POST", body: { name: tName.value, scope: tScope.value, ttl_days: 90 } })); tOut.hidden = false; tOut.textContent = `${j.token}\n(shown once; expires ${dt(j.expires_at)})`; } }, "Mint token")),
      tOut,
    ),
    card("Audit log", table([{ label: "when", get: (a) => dt(a.at) }, { label: "who", get: (a) => a.actor }, { label: "action", get: (a) => a.action }, { label: "target", get: (a) => a.target || "" }, { label: "ok", get: (a) => (a.ok ? "" : badge("failed", "critical")) }, { label: "detail", get: (a) => [a.detail, a.after].filter(Boolean).join(" ").slice(0, 160), wrap: true }], aud.audit)),
  );
}
