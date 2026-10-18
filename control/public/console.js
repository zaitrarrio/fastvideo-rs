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
        kv([
          ["state", badge(pl.state, fleetKind(pl.state))],
          ["workers", pl.workers.length],
          ["models", (pl.models || []).join(", ") || "–"],
          ["queued / running", `${pl.queued} / ${pl.running}`],
        ])
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
  const kids = [
    h("h1", {}, "Models"),
    h("p", { class: "muted" }, `${r.models.length} models served · ${r.engines_reached}/${r.engines_total} engines reachable · live from /fv/v1/capabilities`),
  ];
  if (!r.models.length) {
    kids.push(card("No models found",
      h("p", { class: "muted" }, r.engines_reached === 0
        ? "No engines are reachable. Start a cluster from Deploy, then check Fleet."
        : "Engines are reachable but report no models. Check an engine's /fv/v1/capabilities directly."),
      h("div", { class: "row" }, h("a", { class: "btn", href: "#/deploy" }, "Deploy"), h("a", { class: "btn ghost", href: "#/fleet" }, "Fleet"))
    ));
  } else {
    kids.push(
      h(
        "div",
        { class: "grid g2" },
        ...r.models.map((m) =>
          card(
            h("span", {}, h("code", {}, m.id), " ", m.tier ? badge(m.tier, tierKind(m.tier)) : "", " ", m.resident ? badge("resident", "good") : badge("on-demand", "")),
            kv([
              ["family", m.family || "–"],
              ["recipe", m.recipe ? h("code", {}, m.recipe) : "–"],
              ["served as", (m.served_names || []).join(", ") || "–"],
              ["tasks", (m.tasks || []).join(", ") || "–"],
              ["served by", (m.available_on || []).join(", ") || "–"],
            ])
          )
        )
      )
    );
  }
  main.replaceChildren(...kids);
}

const jobKind = (s) => (s === "running" || s === "Running" ? "good" : s === "queued" || s === "Queued" ? "warn" : s === "failed" || s === "Failed" ? "critical" : "");

async function pageJobsUnified(main) {
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const tab = q.get("tab") === "clusters" || q.has("cluster") ? "clusters" : "engines"; // #/jobs?cluster=<id> (older links) opens that cluster
  const tabBtn = (id, label) => h("a", { class: `btn ${tab === id ? "primary" : ""}`, href: `#/jobs?tab=${id}` }, label);
  const tabs = h("div", { class: "row", style: "margin-bottom:14px" }, tabBtn("engines", "Live engines"), tabBtn("clusters", "Clusters & pods"));
  main.replaceChildren(h("h1", {}, "Jobs"), tabs, h("div", { id: "jobsTab" }));
  const slot = document.getElementById("jobsTab");
  if (tab === "engines") {
    await renderEngineJobsTab(slot);
  } else {
    await renderClusterJobsTab(slot);
  }
}

async function renderEngineJobsTab(main) {
  const render = async () => {
    const r = await api("/api/v1/jobs?limit=100");
    const active = r.jobs.filter((j) => ["queued", "running", "Queued", "Running"].includes(j.state));
    const kids = [
      h("p", { class: "muted" }, `${r.jobs.length} jobs · ${active.length} active · ${r.engines_reached}/${r.engines_total} engines reachable`),
    ];
    if (!r.jobs.length) {
      kids.push(card("No engine jobs", h("p", { class: "muted" }, "No jobs on any reachable engine. Submit one from a model's playground or via the API.")));
    } else {
      kids.push(card(
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
      ));
    }
    main.replaceChildren(...kids);
  };
  state.timers.push(setInterval(() => render().catch(() => {}), 10000));
  render();
}

async function renderClusterJobsTab(main) {
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const list = (await api("/api/v1/clusters?all=1")).clusters || [];
  const pick = q.get("cluster") || list.find((c) => c.status === "running")?.id || list[0]?.id;
  const sel = list.find((c) => c.id === pick || c.name === pick);
  main.replaceChildren(
    card(
      null,
      h("p", { class: "small muted" }, "fv-serve jobs of each cluster and standalone pod, from the jobs D1 the workers write."),
      h("div", { class: "row", style: "flex-wrap:wrap;gap:6px" }, list.map((c) => h("a", { class: `btn ${sel && c.id === sel.id ? "primary" : ""}`, href: `#/jobs?tab=clusters&cluster=${c.id}` }, `${c.name}${c.source === "standalone" ? " (pod)" : ""}${c.status === "running" ? " ●" : ""}`))),
    ),
    sel ? window.fvJobsCard(sel.id, { title: `Jobs: ${sel.name}` }) : card("Jobs", h("p", { class: "muted small" }, "No clusters or standalone pods.")),
  );
}

// One-click deploy (ui/forms/deploy.ts in the editor bundle; docs/control/README.md §4b): model families (or all of
// them), each with its pools (a preset or a template: h3, fasth3, streaming, …) and its instance types (the launch
// form's table), into a new cluster or an existing one; the plan is shown before anything is saved.
async function pageDeploy(main) {
  const host = h("div", { class: "muted small" }, "loading the deploy form…");
  main.replaceChildren(
    h("h1", {}, "Deploy"),
    h("p", { class: "muted" }, "Pick model families (or all), what each deploys and on which instance types; review what will be created, then deploy into a new cluster or add the pools to an existing one."),
    host,
  );
  fvEditor()
    .then((E) => E.mountDeployForm(host, { api, toast }))
    .catch((e) => host.replaceChildren(h("p", { class: "small" }, e.message)));
}

async function pageBenchmarks(main) {
  const r = await api("/api/v1/benchmarks?limit=200");
  const fmtS = (v) => (v == null ? "–" : `${Number(v).toFixed(1)}s`);
  const fmtM = (v) => (v == null ? "–" : `${(Number(v) / 1024).toFixed(1)} GiB`);
  const kids = [
    h("h1", {}, "Benchmarks"),
    h("p", { class: "muted" }, `Observed performance from completed engine jobs · ${r.models.reduce((s, m) => s + m.jobs, 0)} jobs aggregated`),
  ];
  if (!r.models.length) {
    kids.push(card("No benchmarks yet",
      h("p", { class: "muted" }, "Benchmarks are aggregated from completed engine jobs. Run some generations first — timings appear here automatically."),
      h("div", { class: "row" }, h("a", { class: "btn", href: "#/jobs" }, "Jobs"), h("a", { class: "btn ghost", href: "#/models" }, "Models"))
    ));
  } else {
    kids.push(card(
      `Models (${r.models.length})`,
      table(
        [
          { label: "model", get: (m) => h("a", { href: `#/compare?models=${encodeURIComponent(m.model)}` }, h("code", {}, m.model)) },
          { label: "jobs", get: (m) => m.jobs, num: true },
          { label: "p50 infer", get: (m) => fmtS(m.inference_s.p50), num: true },
          { label: "p90 infer", get: (m) => fmtS(m.inference_s.p90), num: true },
          { label: "p50 peak mem", get: (m) => fmtM(m.peak_memory_mb.p50), num: true },
          { label: "max peak mem", get: (m) => fmtM(m.peak_memory_mb.max), num: true },
        ],
        [...r.models].sort((a, b) => (a.inference_s.p50 ?? 1e18) - (b.inference_s.p50 ?? 1e18))
      )
    ));
  }
  main.replaceChildren(...kids);
}

async function pageCompare(main) {
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const initial = (q.get("models") || "").split(",").map((s) => s.trim()).filter(Boolean).slice(0, 2);
  const st = { a: initial[0] || "", b: initial[1] || "", data: null, error: null };

  const render = async () => {
    const kids = [h("h1", {}, "Compare"), h("p", { class: "muted" }, "Side-by-side capabilities and observed performance for up to two models.")];
    kids.push(
      card("Models",
        h("div", { class: "row" },
          h("label", {}, h("span", { class: "k" }, "Model A"), h("input", { value: st.a, oninput: (e) => (st.a = e.target.value), placeholder: "fasth3", style: "width:200px" })),
          h("label", {}, h("span", { class: "k" }, "Model B"), h("input", { value: st.b, oninput: (e) => (st.b = e.target.value), placeholder: "wan22-ti2v-5b", style: "width:200px" })),
          h("button", {
            class: "primary",
            onclick: async () => {
              st.error = null;
              try {
                const ids = [st.a, st.b].filter(Boolean).join(",");
                st.data = await api(`/api/v1/compare?models=${encodeURIComponent(ids)}`);
                history.replaceState(null, "", `#/compare?models=${encodeURIComponent(ids)}`);
              } catch (e) { st.error = e.message; }
              render();
            },
          }, "Compare")
        )
      )
    );
    if (st.error) kids.push(card("Error", h("p", {}, st.error)));
    if (st.data) {
      const [ma, mb] = st.data.models;
      const rows = [
        ["family", (m) => m.capabilities?.family || "–"],
        ["tier", (m) => m.capabilities?.tier || "–"],
        ["recipe", (m) => m.capabilities?.recipe || "–"],
        ["served names", (m) => (m.capabilities?.served_names || []).join(", ") || "–"],
        ["served by", (m) => `${(m.capabilities?.available_on || []).length} workers`],
        ["jobs observed", (m) => m.benchmarks?.jobs ?? "–"],
        ["p50 inference", (m) => (m.benchmarks?.inference_s.p50 != null ? `${Number(m.benchmarks.inference_s.p50).toFixed(1)}s` : "–")],
        ["p90 inference", (m) => (m.benchmarks?.inference_s.p90 != null ? `${Number(m.benchmarks.inference_s.p90).toFixed(1)}s` : "–")],
        ["p50 peak mem", (m) => (m.benchmarks?.peak_memory_mb.p50 != null ? `${(Number(m.benchmarks.peak_memory_mb.p50) / 1024).toFixed(1)} GiB` : "–")],
      ];
      kids.push(card("Comparison",
        table(
          [{ label: "", get: (r) => r[0] },
           { label: ma.id, get: (r) => r[1](ma) },
           ...(mb ? [{ label: mb.id, get: (r) => r[1](mb) }] : [])],
          rows
        )
      ));
    }
    main.replaceChildren(...kids);
  };
  render();
}

async function pageKeyRequests(main) {
  const r = await api("/api/v1/key-requests");
  const reqs = r.requests || [];
  const pending = reqs.filter((x) => x.status === "pending");
  const render = () => {
    main.replaceChildren(
      h("h1", {}, "Key requests"),
      h("p", { class: "muted" }, `${pending.length} pending · from the fv-app portal. Mint the key in the engine admin, deliver it out of band, then mark fulfilled.`),
      card(
        `Requests (${reqs.length})`,
        table(
          [
            { label: "id", get: (x) => h("code", {}, x.id) },
            { label: "name", get: (x) => x.name },
            { label: "email", get: (x) => x.email },
            { label: "use case", get: (x) => (x.use_case || "").slice(0, 60) },
            { label: "status", get: (x) => badge(x.status === "pending" ? "pending" : x.status === "fulfilled" ? "ok" : "down", x.status) },
            {
              label: "", get: (x) => x.status === "pending" ? h("span", {},
                h("button", {
                  class: "sm", onclick: async () => {
                    const hint = prompt("Last 4 chars of the minted key (for the user's reference):") || "";
                    await api(`/api/v1/key-requests/${encodeURIComponent(x.id)}/fulfill`, { method: "POST", body: JSON.stringify({ status: "fulfilled", key_hint: hint }) });
                    location.reload();
                  }
                }, "Fulfill"),
                " ",
                h("button", {
                  class: "sm ghost", onclick: async () => {
                    if (!confirm("Reject this request?")) return;
                    await api(`/api/v1/key-requests/${encodeURIComponent(x.id)}/fulfill`, { method: "POST", body: JSON.stringify({ status: "rejected" }) });
                    location.reload();
                  }
                }, "Reject")
              ) : (x.key_hint ? h("code", {}, `…${x.key_hint}`) : "")
            },
          ],
          reqs
        )
      )
    );
  };
  render();
}

async function pageQuality(main) {
  const r = await api("/api/v1/quality");
  const q = r.quality;
  const kids = [
    h("h1", {}, "Quality gates"),
    h("p", { class: "muted" }, `Reference-free video sanity checks · ${r.engines_reached}/${r.engines_total} engines reached`),
  ];
  if (!q) {
    kids.push(card("Gates", h("p", { class: "muted" }, "No engine reachable. The gates are defined in fastvideo-protocol/src/quality.rs.")));
  } else {
    kids.push(card(
      "Gates",
      table(
        [
          { label: "check", get: (g) => h("strong", {}, g.title) },
          { label: "threshold", get: (g) => h("code", {}, g.threshold) },
          { label: "catches", get: (g) => g.failure_mode },
        ],
        q.gates || []
      )
    ));
    const tiers = q.tiers || {};
    kids.push(card(
      "Tier policy",
      table(
        [
          { label: "tier", get: ([t]) => h("code", {}, t) },
          { label: "policy", get: ([, p]) => p },
        ],
        Object.entries(tiers)
      )
    ));
  }
  main.replaceChildren(...kids);
}
