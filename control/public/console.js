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
  main.replaceChildren(
    h("h1", {}, "Models"),
    h("p", { class: "muted" }, `${r.models.length} deployable presets · VRAM floor is the least GPU memory a worker needs`),
    h(
      "div",
      { class: "grid g2" },
      ...r.models.map((m) =>
        card(
          h("span", {}, m.title, " ", badge(`${m.min_vram_gb} GB`, m.min_vram_gb >= 80 ? "warn" : "good")),
          h("dl", { class: "kv" },
            ["preset", h("code", {}, m.id)],
            ["pool", h("code", {}, m.pool.id)],
            ["serves", m.pool.models.map((mo) => `${mo.id} (${mo.family}/${mo.recipe})`).join(", ") || "–"],
            ["weights", (m.weights || []).join(", ")],
            ...(m.prequant?.length ? [["prequant", m.prequant.join(", ")]] : []),
            ...(m.licence ? [["licence", badge(m.licence, "warn")]] : []),
          ),
          h("p", { class: "muted small" }, m.description),
          h("div", { class: "row" }, h("a", { class: "btn", href: `#/fleet` }, "View fleet"))
        )
      )
    )
  );
}
