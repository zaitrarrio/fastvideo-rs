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
  const tab = q.get("tab") === "clusters" ? "clusters" : "engines";
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

async function pageDeploy(main) {
  const [{ families }, { presets }] = await Promise.all([
    api("/api/v1/families"),
    api("/api/v1/deploy/presets"),
  ]);
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const tab = q.get("tab") || "oneclick";

  const tabBtn = (id, label) => h("a", { class: `btn ${tab === id ? "primary" : ""}`, href: `#/deploy?tab=${id}` }, label);
  main.replaceChildren(
    h("h1", {}, "Deploy"),
    h("div", { class: "row", style: "margin-bottom:14px" },
      tabBtn("oneclick", "One-click"), tabBtn("presets", "Presets"), tabBtn("requests", "Requests")),
    h("div", { id: "deployTab" }),
  );
  const slot = document.getElementById("deployTab");
  if (tab === "presets") return renderPresetsTab(slot, presets);
  if (tab === "requests") return renderRequestsTab(slot);
  return renderOneClick(slot, families, presets);
}

// ---------------------------------------------------------------- one-click wizard
async function renderOneClick(slot, families, presets) {
  // Start clean: a suggested name, H3 preselected, no errors.
  let suggested = "";
  try { suggested = (await api("/api/v1/names/suggest?kind=cluster")).name || ""; } catch {}
  const st = {
    name: suggested,
    fams: new Set(["h3"]),
    specs: [{ vram_gb: 80, arch: "", model: "" }],
    gpu_count: 1,
    storage_gb: 0, // 0 = auto (1.1x)
    kind: "standalone",
    count: 1,
    wait_h: 4,
    spend: "",
    error: null,
    done: null,
  };
  const famById = Object.fromEntries(families.map((f) => [f.id, f]));

  const selWeights = () => [...new Set([...st.fams].flatMap((id) => famById[id]?.weights || []))];
  const selBytes = () => [...st.fams].reduce((s, id) => s + (famById[id]?.bytes || 0), 0);
  const minGb = () => {
    const b = [...st.fams].reduce((s, id) => { const x = famById[id]?.bytes; return x == null ? null : (s == null ? null : s + x); }, 0);
    return b == null ? null : Math.ceil((b * 1.1) / 1e9);
  };
  const effStorage = () => st.storage_gb > 0 ? st.storage_gb : (minGb() || 0);

  function render() {
    const need = minGb();
    const kids = [
      st.error ? card("Could not save", h("p", {}, st.error)) : null,
      st.done ? card("Request queued",
        h("p", {}, `Polling providers for ${st.done.preset_name || "your preset"}. Watch it under the Requests tab.`),
        h("div", { class: "row" }, h("a", { class: "btn primary", href: "#/deploy?tab=requests" }, "View requests"))
      ) : null,
      card("1 · Model families",
        h("p", { class: "muted small" }, "Deploying a family deploys ALL of its weights. STREAMING covers every streaming model; ALL covers everything."),
        h("div", { class: "row", style: "gap:8px;flex-wrap:wrap" },
          ...families.map((f) => h("button", {
            class: st.fams.has(f.id) ? "primary" : "",
            title: `${f.description}\n${f.weights.length} weight trees · ${f.size}`,
            onclick: () => { st.fams.has(f.id) ? st.fams.delete(f.id) : st.fams.add(f.id); render(); },
          }, f.title))
        ),
        h("p", { class: "muted small" }, `${selWeights().length} weight trees · ${need != null ? `needs ≥ ${need} GB` : "size unknown"}`),
      ),
      card("2 · GPUs",
        ...st.specs.map((s, i) => h("div", { class: "row", style: "margin-bottom:6px" },
          h("label", {}, h("span", { class: "k" }, "Min VRAM (GB)"),
            h("input", { type: "number", min: 8, value: s.vram_gb, style: "width:90px",
              oninput: (e) => (s.vram_gb = Number(e.target.value) || 0) })),
          h("label", {}, h("span", { class: "k" }, "Architecture"),
            h("input", { value: s.arch, placeholder: "blackwell (optional)", style: "width:150px",
              oninput: (e) => (s.arch = e.target.value) })),
          h("label", {}, h("span", { class: "k" }, "Model"),
            h("input", { value: s.model, placeholder: "RTX PRO 6000 (optional)", style: "width:170px",
              oninput: (e) => (s.model = e.target.value) })),
          st.specs.length > 1 ? h("button", { class: "ghost", onclick: () => { st.specs.splice(i, 1); render(); } }, "✕") : null,
        )),
        h("div", { class: "row" },
          h("button", { class: "ghost", onclick: () => { st.specs.push({ vram_gb: 80, arch: "", model: "" }); render(); } }, "+ another GPU spec"),
          h("label", {}, h("span", { class: "k" }, "GPUs each"),
            h("input", { type: "number", min: 1, max: 8, value: st.gpu_count, style: "width:70px",
              oninput: (e) => (st.gpu_count = Math.max(1, Number(e.target.value) || 1)) })),
        ),
      ),
      card("3 · Storage & instances",
        h("div", { class: "row" },
          h("label", {}, h("span", { class: "k" }, `Storage GB (min ${need ?? "?"})`),
            h("input", { type: "number", min: 0, value: st.storage_gb || "", placeholder: need ? `auto: ${need}` : "auto",
              style: "width:120px", oninput: (e) => (st.storage_gb = Number(e.target.value) || 0) })),
          h("span", { class: "muted small" }, "Blank = auto (10% over the weights). Never below the minimum."),
        ),
        h("div", { class: "row", style: "margin-top:8px" },
          h("label", {}, h("span", { class: "k" }, "Deploy as"),
            h("select", { onchange: (e) => { st.kind = e.target.value; render(); } },
              ["standalone", "cluster", "serverless"].map((k) =>
                h("option", { value: k, selected: st.kind === k }, k)))),
          h("label", {}, h("span", { class: "k" },
            st.kind === "standalone" ? "Pods" : st.kind === "cluster" ? "Pods in cluster" : "Workers"),
            h("input", { type: "number", min: 1, max: 32, value: st.count, style: "width:70px",
              oninput: (e) => (st.count = Math.max(1, Number(e.target.value) || 1)) })),
        ),
      ),
      card("4 · Limits & launch",
        h("div", { class: "row" },
          h("label", {}, h("span", { class: "k" }, "Name"),
            h("input", { value: st.name, style: "width:170px", oninput: (e) => (st.name = e.target.value) }),
            h("button", { class: "ghost small", title: "New memorable name",
              onclick: async () => { try { st.name = (await api("/api/v1/names/suggest?kind=cluster")).name || st.name; } catch {} render(); } }, "🎲")),
          h("label", {}, h("span", { class: "k" }, "Max wait (hours)"),
            h("input", { type: "number", min: 0.1, max: 168, step: 0.5, value: st.wait_h, style: "width:80px",
              oninput: (e) => (st.wait_h = Number(e.target.value) || 4) })),
          h("label", {}, h("span", { class: "k" }, "Max spend USD (optional)"),
            h("input", { type: "number", min: 0, value: st.spend, placeholder: "no cap", style: "width:110px",
              oninput: (e) => (st.spend = e.target.value) })),
        ),
        h("p", { class: "muted small" },
          `We poll Runpod, Brev and GMI for ${st.specs.map((s) => `${s.vram_gb}GB+`).join(" or ")} GPUs until a match under your cap appears or ${st.wait_h}h passes.`),
        h("div", { class: "row", style: "margin-top:8px" },
          h("button", { class: "primary", onclick: launch }, "Deploy now"),
          h("button", { onclick: savePreset }, "Save as preset"),
        ),
      ),
    ];
    slot.replaceChildren(...kids.filter(Boolean));
  }

  function body() {
    return {
      name: st.name.trim() || undefined,
      families: [...st.fams],
      gpu_specs: st.specs.map((s) => ({ vram_gb: s.vram_gb, arch: s.arch.trim() || null, model: s.model.trim() || null })),
      gpu_count: st.gpu_count,
      storage_gb: effStorage(),
      instance_kind: st.kind,
      instance_count: st.count,
      max_wait_s: Math.round(st.wait_h * 3600),
      max_spend_usd: st.spend === "" ? null : Number(st.spend),
    };
  }

  async function savePreset() {
    st.error = null;
    try {
      const p = await api("/api/v1/deploy/presets", { method: "POST", body: body() });
      toast(`preset saved: ${p.name}`);
      render();
    } catch (e) { st.error = e.message; render(); }
  }

  async function launch() {
    st.error = null; st.done = null;
    try {
      const p = await api("/api/v1/deploy/presets", { method: "POST", body: body() });
      const r = await api("/api/v1/deploy/requests", { method: "POST", body: { preset_id: p.id } });
      st.done = { preset_name: p.name, request: r };
      try { st.name = (await api("/api/v1/names/suggest?kind=cluster")).name || st.name; } catch {}
      render();
    } catch (e) { st.error = e.message; render(); }
  }

  render();
}

async function renderPresetsTab(slot, presets) {
  const render = async () => {
    const { presets: ps } = await api("/api/v1/deploy/presets");
    slot.replaceChildren(
      h("p", { class: "muted" }, `${ps.length} saved one-click presets.`),
      ...ps.map((p) => card(
        h("span", {}, p.name, " ",
          h("span", { class: "muted small" },
            `${p.families.join("+")} · ${p.gpu_specs.map((s) => `${s.vram_gb}GB`).join("/")} ×${p.gpu_count} · ${p.storage_gb}GB · ${p.instance_kind} ×${p.instance_count}`)),
        h("div", { class: "row" },
          h("button", { class: "primary small", onclick: async () => {
            await api("/api/v1/deploy/requests", { method: "POST", body: { preset_id: p.id } });
            toast(`launched ${p.name}`); location.hash = "#/deploy?tab=requests"; route();
          } }, "Launch"),
          h("button", { class: "small", onclick: async () => {
            const n = prompt("Preset name:", `${p.name} (copy)`);
            if (!n) return;
            const d = await api(`/api/v1/deploy/presets/${p.id}/duplicate`, { method: "POST" });
            await api(`/api/v1/deploy/presets/${d.id}`, { method: "PUT", body: { name: n } });
            render();
          } }, "Duplicate"),
          h("button", { class: "small ghost", onclick: () => { location.hash = `#/deploy?tab=oneclick&preset=${p.id}`; } }, "Edit"),
          h("button", { class: "small ghost danger", onclick: async () => {
            if (!confirm(`Delete preset ${p.name}?`)) return;
            await api(`/api/v1/deploy/presets/${p.id}`, { method: "DELETE" });
            render();
          } }, "Delete"),
        ),
      )),
      ps.length ? null : h("p", { class: "muted" }, "No presets yet — build one on the One-click tab."),
    );
  };
  render();
}

async function renderRequestsTab(slot) {
  const render = async () => {
    const { requests } = await api("/api/v1/deploy/requests");
    const agoS = (t) => { const s = Math.max(0, Math.round((Date.now() - t) / 1000)); return s < 60 ? `${s}s ago` : `${Math.round(s / 60)}m ago`; };
    const leftS = (t) => { const s = Math.max(0, Math.round((t - Date.now()) / 1000)); return s < 3600 ? `${Math.round(s / 60)}m left` : `${(s / 3600).toFixed(1)}h left`; };
    slot.replaceChildren(
      h("div", { class: "row", style: "margin-bottom:10px" },
        h("span", { class: "muted" }, `${requests.filter((r) => r.status === "pending").length} polling`),
        h("button", { class: "ghost small", onclick: async () => { await api("/api/v1/deploy/requests/tick", { method: "POST" }); render(); } }, "Check now"),
      ),
      table(
        [
          { label: "preset", get: (r) => r.preset_name },
          { label: "status", get: (r) => badge(r.status, r.status === "done" ? "good" : r.status === "pending" ? "warn" : r.status === "failed" || r.status === "expired" ? "critical" : "") },
          { label: "match", get: (r) => r.match ? `${r.match.provider} ${r.match.display}${r.match.usd_per_hr != null ? ` $${r.match.usd_per_hr.toFixed(2)}/hr` : ""}` : "—" },
          { label: "checks", get: (r) => r.checks, num: true },
          { label: "waiting", get: (r) => r.status === "pending" ? `${agoS(r.created_at)} · ${leftS(r.expires_at)}` : "—" },
          { label: "", get: (r) => r.status === "pending" ? h("button", { class: "ghost small danger", onclick: async () => {
            if (!confirm("Cancel this request?")) return;
            await api(`/api/v1/deploy/requests/${r.id}/cancel`, { method: "POST" }); render();
          } }, "Cancel") : "" },
        ],
        requests,
        "No requests. Launch one from the One-click or Presets tab.",
      ),
    );
    if (requests.some((r) => r.status === "pending")) setTimeout(() => { if (document.body.contains(slot)) render(); }, 30000);
  };
  render();
}

async function pageVolumes(main) {
  const [{ volumes, tracked_only }, { families }] = await Promise.all([
    api("/api/v1/volumes"),
    api("/api/v1/families"),
  ]);
  const st = { name: "", size: 500, region: "", fams: new Set(), error: null };

  function render() {
    const kids = [
      h("h1", {}, "Volumes"),
      h("p", { class: "muted" }, "Runpod network volumes for model weights. Create in any region, track costs, delete when done."),
      st.error ? card("Could not create", h("p", {}, st.error)) : null,
      card("Create volume",
        h("div", { class: "row" },
          h("label", {}, h("span", { class: "k" }, "Name"),
            h("input", { value: st.name, placeholder: "weights-eu", style: "width:150px", oninput: (e) => (st.name = e.target.value) })),
          h("label", {}, h("span", { class: "k" }, "Size GB"),
            h("input", { type: "number", min: 10, value: st.size, style: "width:100px", oninput: (e) => (st.size = Number(e.target.value) || 0) })),
          h("label", {}, h("span", { class: "k" }, "Region"),
            h("input", { value: st.region, placeholder: "EU-RO-1", style: "width:120px", oninput: (e) => (st.region = e.target.value) })),
        ),
        h("p", { class: "muted small" }, "Pre-size for families:"),
        h("div", { class: "row", style: "gap:6px" },
          ...families.map((f) => h("button", {
            class: `small ${st.fams.has(f.id) ? "primary" : ""}`,
            title: `${f.size} minimum`,
            onclick: () => { st.fams.has(f.id) ? st.fams.delete(f.id) : st.fams.add(f.id); render(); },
          }, f.title))),
        h("p", { class: "muted small" }, (() => {
          let bytes = 0, unknown = false;
          for (const id of st.fams) { const f = families.find((x) => x.id === id); if (!f || f.bytes == null) unknown = true; else bytes += f.bytes; }
          return unknown ? "size unknown for selection" : st.fams.size ? `suggested: ${Math.ceil((bytes * 1.1) / 1e9)} GB` : "";
        })()),
        h("div", { class: "row", style: "margin-top:8px" },
          h("button", { class: "primary", onclick: async () => {
            st.error = null;
            try {
              await api("/api/v1/volumes", { method: "POST", body: { name: st.name.trim(), size_gb: st.size, data_center_id: st.region.trim() } });
              toast("volume created"); pageVolumes(main);
            } catch (e) { st.error = e.message; render(); }
          } }, "Create volume")),
      ),
      card(`Volumes (${volumes.length})`,
        table(
          [
            { label: "name", get: (v) => v.name },
            { label: "size", get: (v) => `${v.size} GB`, num: true },
            { label: "region", get: (v) => v.dataCenterId },
            { label: "cost", get: (v) => v.tracked?.price ? `$${v.tracked.price.usd_per_gb_mo.toFixed(4)}/GB-mo ($${(v.tracked.price.usd_per_gb_mo * v.size).toFixed(2)}/mo)` : "—" },
            { label: "", get: (v) => h("span", {},
              h("button", { class: "small", title: "Serverless endpoint with this volume mounted + Flash Boot", onclick: async () => {
                const name = prompt("Endpoint name:", v.name);
                if (!name) return;
                try {
                  await api(`/api/v1/volumes/${encodeURIComponent(v.id)}/serverless`, { method: "POST", body: { name } });
                  toast("serverless endpoint creating"); location.hash = "#/serverless"; route();
                } catch (e) { toast(e.message); }
              } }, "Serverless"),
              " ",
              h("button", { class: "ghost small danger", onclick: async () => {
                if (!confirm(`Delete volume ${v.name} (${v.size} GB)? The weights on it go too.`)) return;
                await api(`/api/v1/volumes/${encodeURIComponent(v.id)}`, { method: "DELETE" });
                toast("volume deleted"); pageVolumes(main);
              } }, "Delete")) },
          ],
          volumes,
          "No volumes.",
        ),
      ),
    ];
    main.replaceChildren(...kids.filter(Boolean));
  }
  render();
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
