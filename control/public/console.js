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

async function pageEngineJobs(main) {  const render = async () => {
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

async function pageDeploy(main) {
  const { recipes } = await api("/api/v1/deploy/recipes");
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const st = {
    step: 1,
    recipe: q.get("recipe") || null,
    name: "",
    count: 1,
    gpu_types: "",
    data_centers: "",
    validation: null,
    result: null,
    error: null,
  };

  const steps = ["Recipe", "Configure", "Review", "Launch"];
  function stepBar() {
    return h("div", { class: "row", style: "gap:8px;margin-bottom:16px" },
      ...steps.map((s, i) =>
        h("span", { class: `badge ${st.step === i + 1 ? "primary" : st.step > i + 1 ? "good" : ""}` }, `${i + 1}. ${s}`)
      )
    );
  }

  function render() {
    const kids = [h("h1", {}, "Deploy"), stepBar()];
    if (st.step === 1) {
      kids.push(
        h("p", { class: "muted" }, "Pick a recipe — a tested pool preset with its weights and worker config."),
        h("div", { class: "grid g2" },
          ...recipes.map((r) =>
            card(
              h("span", {}, r.title, " ", badge(`${r.min_vram_gb} GB`, "warn")),
              h("p", { class: "muted small" }, r.description),
              h("dl", { class: "kv" },
                ["serves", r.models.map((m) => m.id).join(", ")],
                ["weights", r.weights.join(", ")],
              ),
              h("div", { class: "row" },
                h("button", {
                  class: st.recipe === r.id ? "primary" : "",
                  onclick: () => { st.recipe = r.id; if (!st.name) st.name = `fv-${r.id}-1`; render(); },
                }, st.recipe === r.id ? "Selected" : "Select")
              )
            )
          )
        ),
        h("div", { class: "row", style: "margin-top:12px" },
          h("button", { class: "primary", disabled: !st.recipe, onclick: () => { st.step = 2; render(); } }, "Continue")
        )
      );
    } else if (st.step === 2) {
      const recipe = recipes.find((r) => r.id === st.recipe);
      const field = (label, input) => h("label", {}, h("span", { class: "k" }, label), input);
      kids.push(
        h("p", { class: "muted" }, h("code", {}, recipe.id), ` — ${recipe.title}`),
        card("Configuration",
          h("div", { class: "grid g2" },
            field("Cluster name", h("input", { value: st.name, oninput: (e) => (st.name = e.target.value), placeholder: "fv-h3-turbo-1" })),
            field("Pod count", h("input", { type: "number", min: 1, max: 8, value: st.count, oninput: (e) => (st.count = Math.max(1, parseInt(e.target.value) || 1)) })),
            field("GPU types (comma-separated, blank = default)", h("input", { value: st.gpu_types, oninput: (e) => (st.gpu_types = e.target.value), placeholder: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" })),
            field("Data centers (comma-separated, blank = default)", h("input", { value: st.data_centers, oninput: (e) => (st.data_centers = e.target.value), placeholder: "EU-RO-1" })),
          )
        ),
        h("div", { class: "row", style: "margin-top:12px" },
          h("button", { onclick: () => { st.step = 1; render(); } }, "Back"),
          h("button", {
            class: "primary",
            onclick: async () => {
              st.error = null;
              try {
                st.validation = await api("/api/v1/deploy/validate", {
                  method: "POST",
                  body: deployBody(),
                });
              } catch (e) { st.error = e.message; }
              st.step = 3; render();
            },
          }, "Validate & review")
        )
      );
    } else if (st.step === 3) {
      const v = st.validation;
      const issues = v?.issues || v?.errors || [];
      kids.push(
        st.error ? card("Validation failed", h("p", {}, st.error)) : null,
        card("Validation",
          issues.length
            ? h("ul", {}, ...issues.map((i) => h("li", {}, `${(i.path || []).join(".")}: ${i.message}`)))
            : h("p", {}, badge("valid", "good"), " No issues found.")
        ),
        h("div", { class: "row", style: "margin-top:12px" },
          h("button", { onclick: () => { st.step = 2; render(); } }, "Back"),
          h("button", {
            class: "primary",
            disabled: !!issues.length,
            onclick: async () => {
              st.error = null;
              try {
                st.result = await api("/api/v1/deploy", { method: "POST", body: { ...deployBody(), start: true } });
              } catch (e) { st.error = e.message; }
              st.step = 4; render();
            },
          }, "Launch deploy")
        )
      );
    } else {
      kids.push(
        st.error
          ? card("Deploy failed", h("p", {}, st.error))
          : card("Deploy launched",
              h("dl", { class: "kv" },
                ["cluster", h("a", { href: `#/cluster/${st.result.cluster.id}` }, st.result.cluster.name)],
                ["operation", st.result.operation ? h("code", {}, st.result.operation) : "not started"],
              ),
              h("p", { class: "muted small" }, "Follow provisioning on the cluster page."),
              h("div", { class: "row" },
                h("a", { class: "btn primary", href: `#/cluster/${st.result.cluster.id}` }, "View cluster"),
                h("a", { class: "btn", href: `#/fleet` }, "View fleet")
              )
            ),
        h("div", { class: "row", style: "margin-top:12px" },
          h("button", { onclick: () => { st.step = 1; st.result = null; render(); } }, "Deploy another")
        )
      );
    }
    main.replaceChildren(...kids);
  }

  function deployBody() {
    const split = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);
    const b = { recipe: st.recipe, name: st.name.trim(), count: st.count };
    if (st.gpu_types.trim()) b.gpu_types = split(st.gpu_types);
    if (st.data_centers.trim()) b.data_centers = split(st.data_centers);
    return b;
  }

  render();
}
