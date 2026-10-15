// fv-console: the operator console (Phase A2). Fleet is the first surface:
// every worker with its engine state, joined by GET /api/v1/fleet.
"use strict";

const fleetKind = (st) =>
  st === "Ready" ? "good" : st === "Loading" || st === "Busy" ? "warn" : st === "Draining" ? "warn" : st === "Down" || st === "Unhealthy" || st === "Failed" ? "critical" : "";

function vramBar(usedPct, totalGb) {
  if (usedPct === null || usedPct === undefined || !totalGb) return "–";
  const pct = Math.max(0, Math.min(100, usedPct));
  const usedGb = ((pct / 100) * totalGb).toFixed(1);
  const color = pct > 90 ? "var(--red)" : pct > 75 ? "var(--amber)" : "var(--green)";
  return h(
    "span",
    { title: `${usedGb} / ${totalGb} GB` },
    h("span", {
      style: `display:inline-block;width:72px;height:8px;border-radius:4px;background:var(--border);vertical-align:middle;margin-right:8px;overflow:hidden`,
    }, h("span", { style: `display:block;height:100%;width:${pct}%;background:${color}` })),
    `${usedGb}/${totalGb}G`
  );
}

function poolCards(pools) {
  if (!pools.length) return null;
  return h(
    "div",
    { class: "grid g2", style: "margin-bottom:16px" },
    ...pools.map((pl) =>
      card(
        `${pl.id} ${pl.kind ? `(${pl.kind})` : ""}`,
        h("dl", { class: "kv" },
          ["state", badge(pl.state, fleetKind(pl.state))],
          ["workers", pl.workers.length],
          ["models", (pl.models || []).join(", ") || "–"],
          ["queued / running", `${pl.queued} / ${pl.running}`],
        )
      )
    )
  );
}

async function pageFleet(main) {
  const render = async () => {
    const f = await api("/api/v1/fleet");
    const s = f.summary;
    main.replaceChildren(
      h("h1", {}, "Fleet"),
      h("p", { class: "muted" }, `Joined engine + infra view · updated ${new Date(f.at).toLocaleTimeString()}`),
    h(
      "div",
      { class: "tiles" },
      h("div", { class: "tile" }, h("div", { class: "v" }, `${s.workers_with_engine}/${s.workers}`), h("div", { class: "muted small" }, "engines reachable")),
      h("div", { class: "tile" }, h("div", { class: "v" }, s.gpus), h("div", { class: "muted small" }, `${s.vram_total_gb} GB VRAM`)),
      h("div", { class: "tile" }, h("div", { class: "v" }, `${s.queued}/${s.running}`), h("div", { class: "muted small" }, "queued / running")),
      h("div", { class: "tile" }, h("div", { class: "v" }, fmt$(s.cost_per_hr)), h("div", { class: "muted small" }, "burn / hour")),
    ),
    poolCards(f.pools),
    card(
      `Workers (${f.workers.length})`,
      table(
        [
          { label: "worker", get: (w) => h("a", { href: `#/pod/${w.pod_id}` }, w.name || w.pod_id) },
          { label: "cluster", get: (w) => w.cluster_id || "–" },
          { label: "GPU", get: (w) => (w.gpu ? `${w.gpu.replace("NVIDIA ", "")} × ${w.gpu_count}` : "cpu") },
          { label: "VRAM", get: (w) => vramBar(w.vram_used_pct, w.vram_total_gb) },
          { label: "engine", get: (w) => (w.engine ? badge("reachable", "good") : badge("unreachable", "warn")) },
          {
            label: "pools",
            get: (w) => (w.engine ? w.engine.pools.map((pl) => `${pl.id} `).join("") + badge(w.engine.pools.map((pl) => pl.state).join(","), fleetKind(w.engine.pools[0]?.state)) : "–"),
          },
          { label: "models", get: (w) => (w.engine ? [...new Set(w.engine.pools.flatMap((pl) => pl.models))].join(", ") : "–") },
          { label: "q/r", get: (w) => (w.engine ? `${w.engine.queued}/${w.engine.running}` : `${w.jobs_queued ?? "–"}/${w.jobs_running ?? "–"}`), num: true },
          { label: "$/hr", get: (w) => fmt$(w.cost_per_hr), num: true },
          { label: "build", get: (w) => (w.build_sha ? w.build_sha.slice(0, 7) : "–") },
        ],
        f.workers
      )
    )
  );
  };
  // Auto-refresh every 15s; route() clears timers on navigation.
  state.timers.push(setInterval(() => render().catch(() => {}), 15000));
  render();
}

async function pageModels(main) {
  const r = await api("/api/v1/models");
  const tierKind = (t) => (t === "Max" ? "good" : t === "Turbo" ? "warn" : "");
  main.replaceChildren(
    h("h1", {}, "Models"),
    h("p", { class: "muted" }, `${r.models.length} models served · ${r.engines_reached}/${r.engines_total} engines reachable · live from /fv/v1/capabilities`),
    h(
      "div",
      { class: "grid g2" },
      ...r.models.map((m) =>
        card(
          h("span", {}, h("code", {}, m.id), " ", m.tier ? badge(m.tier, tierKind(m.tier)) : "", " ", m.resident ? badge("resident", "good") : badge("on-demand", "")),
          h("dl", { class: "kv" },
            ["family", m.family || "–"],
            ["recipe", m.recipe ? h("code", {}, m.recipe) : "–"],
            ["served as", (m.served_names || []).join(", ") || "–"],
            ["tasks", (m.tasks || []).join(", ") || "–"],
            ["served by", (m.available_on || []).join(", ") || "–"],
          )
        )
      )
    )
  );
}

const jobKind = (s) => (s === "running" || s === "Running" ? "good" : s === "queued" || s === "Queued" ? "warn" : s === "failed" || s === "Failed" ? "critical" : "");

async function pageEngineJobs(main) {
  const render = async () => {
    const r = await api("/api/v1/jobs?limit=100");
    const active = r.jobs.filter((j) => ["queued", "running", "Queued", "Running"].includes(j.state));
    main.replaceChildren(
      h("h1", {}, "Jobs"),
      h("p", { class: "muted" }, `${r.jobs.length} jobs · ${active.length} active · ${r.engines_reached}/${r.engines_total} engines reachable`),
      card(
        `Engine jobs (${r.jobs.length})`,
        table(
          [
            { label: "job", get: (j) => h("code", { title: j.id }, String(j.id).slice(0, 12)) },
            { label: "model", get: (j) => j.resolved?.model || j.request_echo?.model || "–" },
            { label: "state", get: (j) => badge(j.state, jobKind(j.state)) },
            { label: "progress", get: (j) => (j.progress != null ? `${Math.round(j.progress * 100)}%` : "–"), num: true },
            { label: "pod", get: (j) => h("a", { href: `#/pod/${j.pod_id}` }, j.pod_name || j.pod_id) },
            { label: "created", get: (j) => (j.created_at ? new Date(j.created_at).toLocaleTimeString() : "–") },
            { label: "inference", get: (j) => (j.metrics?.inference_time_s != null ? `${Number(j.metrics.inference_time_s).toFixed(1)}s` : "–"), num: true },
          ],
          r.jobs
        )
      )
    );
  };
  state.timers.push(setInterval(() => render().catch(() => {}), 10000));
  render();
}
