// fv-control dashboard: fv-serve jobs of clusters and standalone pods
// (docs/control/README.md "Jobs"). fvJobsCard(cluster) is the Jobs card the
// cluster and standalone pages show; #/jobs lists every cluster's with
// "Cancel a job by id". Jobs come from the jobs D1 the workers write (the
// edge's for edge clusters, fv-jobs for direct ones); a cancel goes to the
// worker that holds the job. Self-contained like serverless.js: it uses
// app.js's helpers (h, card, table, badge, api, act, every, …); every API
// value goes into the DOM through textContent.
"use strict";

(function () {
  const KIND = { queued: "warn", running: "good", succeeded: "", failed: "critical", cancelled: "" };
  const FILTERS = [["active", "queued,running", "Queued and running"], ["queued", "queued", "Queued"], ["running", "running", "Running"], ["all", "", "All recent"]];

  /** The Jobs card of one cluster or standalone pod; it refreshes itself every 10 s and keeps its filter across the page's redraws. */
  const cards = new Map();
  function fvJobsCard(clusterId, opts = {}) {
    if (cards.has(clusterId)) return cards.get(clusterId).el;
    const st = { filter: "active", data: null, error: null, busy: new Set(), last: null };
    const head = h("div", { class: "row", style: "flex-wrap:wrap;gap:8px;margin-bottom:8px" });
    const body = h("div", {});
    const msg = h("div", { class: "small", style: "margin-top:6px" });
    const el = card(opts.title || "Jobs", head, body, msg);
    el.dataset.jobsCard = clusterId;
    const load = async () => {
      const f = FILTERS.find((x) => x[0] === st.filter);
      try {
        st.data = await api(`/api/clusters/${encodeURIComponent(clusterId)}/jobs?limit=100${f[1] ? `&status=${f[1]}` : ""}`);
        st.error = null;
      } catch (e) {
        st.error = e.message;
      }
      draw();
    };
    const cancelOne = async (j) => {
      if (!confirm(`Cancel ${j.external_id} (${j.api}, ${j.model}, ${j.status})?`)) return;
      st.busy.add(j.id);
      draw();
      try {
        const r = await api(`/api/jobs/${encodeURIComponent(j.id)}/cancel`, { method: "POST", body: { cluster: clusterId } });
        showResult(r);
      } catch (e) {
        msg.replaceChildren(h("span", { style: "color:var(--critical-text)" }, e.message));
      } finally {
        st.busy.delete(j.id);
        await load();
      }
    };
    const showResult = (r) => {
      toast(`${r.external_id}: ${r.status}${r.cancel_requested ? " (cancel requested)" : ""}`);
      msg.replaceChildren(
        h("span", {}, badge(r.status, KIND[r.status] || "warn"), " ", `${r.external_id} (${r.api}) via ${r.via || "–"}${r.pod ? ` on ${r.pod}` : ""}: ${r.note}`),
      );
    };
    const cancelQueued = async () => {
      const queued = st.data?.counts?.queued || 0;
      const pools = [...new Set((st.data?.pods || []).map((p) => p.pool).filter(Boolean))];
      const x = await (await fvEditor()).askCancelQueued(api, st.data?.cluster?.name || clusterId, queued, pools);
      if (!x) return;
      try {
        const r = await act("cancel queued", () => api(`/api/clusters/${encodeURIComponent(clusterId)}/jobs/cancel-queued`, { method: "POST", body: x }));
        msg.replaceChildren(h("div", {}, `cancelled ${r.cancelled} of ${r.queued} queued`), ...r.failed.map((f) => h("div", { style: "color:var(--critical-text)" }, `${f.job} (${f.api}): ${f.note}`)));
      } catch (e) {
        msg.replaceChildren(h("span", { style: "color:var(--critical-text)" }, e.message));
      }
      await load();
    };
    const cancelById = async () => {
      const x = await (await fvEditor()).askJobCancel(api, st.data?.cluster?.id);
      if (!x) return;
      try {
        showResult(await api(`/api/jobs/${encodeURIComponent(x.job)}/cancel`, { method: "POST", body: x.cluster ? { cluster: x.cluster } : {} }));
      } catch (e) {
        msg.replaceChildren(h("span", { style: "color:var(--critical-text)" }, e.message));
      }
      await load();
    };
    function draw() {
      const d = st.data;
      const counts = d?.counts || {};
      head.replaceChildren(
        ...FILTERS.map(([id, , label]) => h("button", { class: id === st.filter ? "primary" : "ghost", "aria-pressed": String(id === st.filter), onclick: () => { st.filter = id; load(); } }, label)),
        h("span", { class: "small muted", style: "margin-left:6px" }, ["queued", "running", "succeeded", "failed", "cancelled"].filter((k) => counts[k]).map((k) => `${counts[k]} ${k}`).join(" · ") || (d ? "no jobs yet" : "")),
        h("span", { style: "flex:1" }),
        h("button", { class: "danger", disabled: !counts.queued, onclick: cancelQueued }, `Cancel all queued (${counts.queued || 0})…`),
        h("button", { onclick: cancelById }, "Cancel by id…"),
      );
      if (st.error) {
        body.replaceChildren(h("p", { class: "small muted" }, st.error));
        return;
      }
      if (!d) {
        body.replaceChildren(h("p", { class: "small muted" }, "loading…"));
        return;
      }
      body.replaceChildren(
        table(
          [
            { label: "job", get: (j) => h("code", { title: `internal id ${j.id}` }, j.external_id) },
            { label: "API", get: (j) => j.api },
            { label: "status", get: (j) => h("span", {}, badge(j.status, KIND[j.status] || ""), j.cancel_requested && !["cancelled", "succeeded", "failed"].includes(j.status) ? h("span", { class: "small muted" }, " cancel requested") : null) },
            { label: "model", get: (j) => j.model },
            { label: "pool", get: (j) => j.pool || "–" },
            { label: "worker", get: (j) => (j.worker ? h("a", { href: `#/pod/${j.worker}` }, j.worker) : h("span", { class: "muted" }, "edge queue")) },
            { label: "submitted", get: (j) => ago(j.created_at) },
            { label: "started", get: (j) => (j.started_at ? ago(j.started_at) : "–") },
            { label: "progress", get: (j) => (j.status === "running" ? `${Math.round((j.progress || 0) * 100)}%` : "–"), num: true },
            { label: "", get: (j) => (["queued", "running"].includes(j.status) ? h("button", { class: "ghost", disabled: st.busy.has(j.id) || (j.cancel_requested && j.status === "running"), title: j.cancel_route ? `the API's own route: ${j.cancel_route}` : "", onclick: () => cancelOne(j) }, st.busy.has(j.id) ? "cancelling…" : "Cancel") : "") },
          ],
          d.jobs,
          st.filter === "all" ? "No jobs in the jobs D1 for these pods." : "No queued or running jobs.",
        ),
        d.note ? h("p", { class: "small muted" }, d.note) : null,
        h("p", { class: "small muted" }, `From ${d.source === "edge" ? "the edge's D1" : "fv-jobs"} (${d.cluster.control_plane}). A cancel goes to the worker that holds the job (its internal route); a queued job stops at once, a running one at its next denoise step.`),
      );
    }
    draw();
    load();
    every(10000, () => (document.body.contains(el) ? load() : null));
    cards.set(clusterId, { el });
    state.cleanups.push(() => cards.delete(clusterId));
    return el;
  }
  window.fvJobsCard = fvJobsCard;

  /** #/jobs: every cluster's and standalone pod's jobs (live ones first). */
  async function pageJobs(main) {
    const q = new URLSearchParams(location.hash.split("?")[1] || "");
    const list = (await api("/api/clusters?all=1")).clusters || [];
    const pick = q.get("cluster") || list.find((c) => c.status === "running")?.id || list[0]?.id;
    const sel = list.find((c) => c.id === pick || c.name === pick);
    main.replaceChildren(
      h("h1", {}, "Jobs"),
      card(
        null,
        h("p", { class: "small muted" }, "fv-serve jobs of each cluster and standalone pod, from the jobs D1 the workers write. Serverless endpoints have their own (Serverless → an endpoint → Jobs)."),
        h("div", { class: "row", style: "flex-wrap:wrap;gap:6px" }, list.map((c) => h("a", { class: `btn ${sel && c.id === sel.id ? "primary" : ""}`, href: `#/jobs?cluster=${c.id}` }, `${c.name}${c.source === "standalone" ? " (pod)" : ""}${c.status === "running" ? " ●" : ""}`))),
      ),
      sel ? fvJobsCard(sel.id, { title: `Jobs: ${sel.name}` }) : card("Jobs", h("p", { class: "muted small" }, "No clusters or standalone pods.")),
    );
  }
  routes.push([/^#\/jobs$/, pageJobs]);
})();
