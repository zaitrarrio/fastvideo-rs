// fv-control dashboard: the Serverless page (docs/control/serverless.md).
// Runpod serverless endpoints fv-control manages: create from a form or a
// spec JSON, status, workers, queue, cost, scale, extend, test invoke, logs,
// delete. Self-contained: it adds its route to app.js's router and uses its
// helpers (h, card, table, badge, api, act, every, …). Every API value goes
// into the DOM through textContent.
"use strict";

(function () {
  const slsKind = (st) => (st === "active" ? "good" : st === "creating" || st === "deleting" || st === "scaled-down" ? "warn" : st === "failed" || st === "gone" ? "critical" : "");
  const ms = (v) => (v === null || v === undefined ? "–" : v < 1000 ? `${Math.round(v)} ms` : dur(v / 1000));
  const pre = (v) => h("pre", { class: "log", style: "max-height:320px;overflow:auto;white-space:pre-wrap" }, typeof v === "string" ? v : JSON.stringify(v, null, 2));
  const workersLine = (hh) => {
    const w = hh?.workers;
    if (!w) return "–";
    return `${w.running || 0} running, ${w.idle || 0} idle, ${w.initializing || 0} starting${w.throttled ? `, ${w.throttled} throttled` : ""}${w.unhealthy ? `, ${w.unhealthy} unhealthy` : ""}`;
  };
  const queueLine = (hh) => (hh?.jobs ? `${hh.jobs.inQueue || 0} queued, ${hh.jobs.inProgress || 0} running` : "–");

  async function pageServerless(main) {
    const q = new URLSearchParams((location.hash.split("?")[1] || ""));
    if (q.get("ep")) return pageEndpoint(main, q.get("ep"));
    const [list, defs] = await Promise.all([api(`/api/serverless${q.get("all") ? "?all=1" : ""}`), api("/api/serverless/defaults?name=fake&variant=cpu")]);
    const eps = list.endpoints || [];
    main.replaceChildren(
      card(
        "Serverless endpoints",
        h("p", { class: "small muted" }, "Runpod serverless endpoints fv-control created (named fvc-…; nothing else is touched). Cost is Runpod's billing per endpoint (lags a few minutes) and goes into the cost ledger as owner serverless:<name>. Limits: ", `${list.policy.max_endpoints} endpoints, ${list.policy.max_workers} workers in all, balance ≥ floor + $${list.policy.balance_margin} to create or scale up.`),
        table(
          [
            { label: "endpoint", get: (e) => h("a", { href: `#/serverless?ep=${e.id}` }, e.name) },
            { label: "Runpod id", get: (e) => e.endpoint_id || "–" },
            { label: "status", get: (e) => badge(e.status, slsKind(e.status)) },
            { label: "mode", get: (e) => `${e.mode} · ${e.spec.variant}/${e.spec.compute}` },
            { label: "workers", get: (e) => `${e.workers ?? 0} (${e.spec.workers_min}..${e.spec.workers_max})` },
            { label: "queue", get: (e) => queueLine(e.health) },
            { label: "$/hr now", get: (e) => fmt$(e.live_dph, 3), num: true },
            { label: "today", get: (e) => fmt$(e.cost_today, 3), num: true },
            { label: "billed", get: (e) => fmt$(e.cost_usd, 3), num: true },
            { label: "backstop", get: (e) => (e.deadline && !e.deleted_at ? `${e.deadline_action} ${until(e.deadline)}` : "–") },
            { label: "created", get: (e) => ago(e.created_at) },
          ],
          eps,
          "No serverless endpoints yet.",
        ),
        h("div", { class: "row", style: "margin-top:8px" }, h("a", { class: "btn", href: q.get("all") ? "#/serverless" : "#/serverless?all=1" }, q.get("all") ? "Recent only" : "Show all (deleted too)"), h("button", { onclick: async () => { await act("tick", () => api("/api/serverless/tick", { method: "POST" })); route(); } }, "Refresh health and billing")),
      ),
      createCard(defs.spec),
      card("Policy", h("p", { class: "small muted" }, "Limits over every endpoint fv-control manages."), docPanel("serverless-policy", "default", { title: "serverless policy" })),
    );
    every(15000, async () => {
      if (!location.hash.startsWith("#/serverless") || location.hash.includes("ep=")) return;
      const fresh = await api(`/api/serverless${q.get("all") ? "?all=1" : ""}`).catch(() => null);
      if (fresh && JSON.stringify(fresh.endpoints.map((e) => [e.status, e.workers, e.health])) !== JSON.stringify(eps.map((e) => [e.status, e.workers, e.health]))) route();
    });
  }

  /** The new-endpoint form (ui/forms/serverless.ts): every field bound to the serverless-endpoint schema, a JSON tab, Create off until valid. */
  function createCard(def) {
    const host = h("div", { class: "muted small" }, "loading the form…");
    fvEditor().then((E) => E.mountEndpointForm(host, { api, toast, defaults: def, onCreated: (id) => { location.hash = `#/serverless?ep=${id}`; } })).catch((e) => host.replaceChildren(h("p", { class: "small" }, e.message)));
    return card("New endpoint", host);
  }

  async function pageEndpoint(main, id) {
    const d = await api(`/api/serverless/${encodeURIComponent(id)}`);
    const e = d.endpoint;
    const live = !e.deleted_at;
    const hh = d.health && !d.health.error ? d.health : e.health;
    const st = d.stats || {};
    const kv = (k, v) => h("div", { class: "kv" }, h("span", { class: "muted small" }, k), h("b", {}, v));
    const head = card(
      `${e.name} `,
      h("div", { class: "row", style: "flex-wrap:wrap;gap:18px" },
        kv("status", badge(e.status, slsKind(e.status))),
        kv("Runpod", `${e.runpod_name} · ${e.endpoint_id || "–"}`),
        kv("mode", `${e.mode} · ${e.spec.variant} · ${e.spec.compute}`),
        kv("workers", workersLine(hh)),
        kv("queue", queueLine(hh)),
        kv("jobs (Runpod)", hh?.jobs ? `${hh.jobs.completed || 0} done, ${hh.jobs.failed || 0} failed` : "–"),
        kv("scale", `${e.spec.workers_min}..${e.spec.workers_max}, idle ${e.spec.idle_timeout_s}s${e.spec.flashboot ? ", FlashBoot" : ""}`),
        kv("$/hr now", fmt$(e.live_dph, 3)),
        kv("billed", `${fmt$(e.cost_usd, 4)} (${dur((e.billed_ms || 0) / 1000)})`),
        kv("cold start", ms(st.cold_start_ms)),
        kv("warm queue wait", ms(st.warm_delay_ms)),
        kv("exec (median)", ms(st.exec_ms)),
        kv("backstop", e.deadline && live ? `${e.deadline_action} ${until(e.deadline)}` : "–"),
      ),
      e.last_error ? h("p", { class: "small", style: "color:var(--critical-text)" }, e.last_error) : null,
      h("p", { class: "small muted" }, "Image ", e.image || "–", e.urls ? h("span", {}, " · ", Object.entries(e.urls).map(([k, u]) => h("span", {}, `${k} `, h("code", {}, u), " "))) : null),
      h("div", { class: "row", style: "margin-top:6px" },
        h("a", { class: "btn", href: "#/serverless" }, "← all endpoints"),
        h("a", { class: "btn", href: `#/logs?q=&pod=${(d.runpod?.workers || [])[0]?.id || ""}` }, "Log explorer"),
        // fv-serve's console for this endpoint, served by fv-control (docs/control/serverless.md "Console").
        live && e.endpoint_id ? h("a", { class: "btn primary", id: "open-console", href: `/serverless/${encodeURIComponent(e.endpoint_id)}/console`, target: "_blank", rel: "noopener", title: "Models, Playground and API tabs, Native API: each request runs as a queue job" }, "Open console") : null,
        live && e.endpoint_id ? h("button", { title: "Forget the console's cached capabilities and schemas: the next page load asks a worker again", onclick: async () => { await act("console cache", () => api(`/api/serverless/${e.id}/console-cache`, { method: "DELETE" })); } }, "Refresh console cache") : null,
      ),
    );
    if (!live) {
      main.replaceChildren(head, jobsCard(d, null), costCard(d), auditCard(d));
      return;
    }
    // Scale / extend / delete.
    const scaleHost = h("div", { class: "muted small" }, "loading…");
    fvEditor().then((E) => E.mountScaleForm(scaleHost, { api, toast, endpoint: e, onDone: () => route() })).catch((err) => scaleHost.replaceChildren(h("p", { class: "small" }, err.message)));
    const ctl = card(
      "Scale",
      h("div", { class: "row", style: "flex-wrap:wrap" },
        scaleHost,
        h("button", { onclick: async () => { await act("scale to 0", () => api(`/api/serverless/${e.id}/scale`, { method: "POST", body: { workers_min: 0, workers_max: 0 } })); route(); } }, "Scale to 0"),
        h("button", { onclick: async () => { await act("extend", () => api(`/api/serverless/${e.id}/extend`, { method: "POST", body: { minutes: 30 } })); route(); } }, "Backstop +30 min"),
        h("button", { class: "danger", onclick: async () => {
          if (!confirm(`Delete ${e.runpod_name} (${e.endpoint_id}) and its template? Running jobs are lost.`)) return;
          await act("delete", () => api(`/api/serverless/${e.id}`, { method: "DELETE" }));
          route();
        } }, "Delete endpoint"),
      ),
      h("p", { class: "small muted" }, "Scaling up (or creating) needs the balance above the floor plus the serverless margin. The cron scales every endpoint to 0 below the floor, and runs the backstop at its time."),
    );
    main.replaceChildren(head, ctl, invokeCard(e), queueCard(e, hh), workersCard(e, d), specCard(e), jobsCard(d, e), costCard(d), auditCard(d));
    every(10000, async () => {
      if (!location.hash.includes(`ep=${id}`)) return;
      const f = await api(`/api/serverless/${encodeURIComponent(id)}`).catch(() => null);
      if (f && (f.endpoint.status !== e.status || JSON.stringify(f.health) !== JSON.stringify(d.health) || f.jobs.length !== d.jobs.length)) {
        if (!document.activeElement || !main.contains(document.activeElement) || document.activeElement.tagName === "BUTTON") route();
      }
    });
  }

  function invokeCard(e) {
    const out = h("div", { class: "small", style: "margin-top:8px" });
    const ta = h("textarea", { rows: 4, style: "width:100%;font-family:var(--mono, monospace);font-size:12px", "aria-label": "Invoke input" });
    ta.value = e.mode === "lb" ? JSON.stringify({ method: "GET", path: "/ping" }) : JSON.stringify({ kind: "info" });
    const presets = e.mode === "lb"
      ? [["ping", { method: "GET", path: "/ping" }], ["capabilities", { method: "GET", path: "/fv/v1/capabilities" }]]
      : [["info", { kind: "info" }], ["capabilities", { kind: "http", method: "GET", path: "/fv/v1/capabilities" }], ["fake job", { kind: "http", method: "POST", path: "/fv/v1/jobs", body: { model: "fake-wan", prompt: "a red fox trotting through fresh snow", seed: 1 }, wait: true }]];
    const show = (r) => out.replaceChildren(
      h("div", { class: "row", style: "flex-wrap:wrap;gap:14px" },
        badge(r.status, /COMPLETED|HTTP 2/.test(r.status || "") ? "good" : /FAILED|TIMED_OUT|HTTP [45]/.test(r.status || "") ? "critical" : "warn"),
        r.cold ? badge("cold start", "warn") : null,
        h("span", {}, "queue wait ", h("b", {}, ms(r.delay_ms))), h("span", {}, "execution ", h("b", {}, ms(r.exec_ms))), h("span", {}, "end to end ", h("b", {}, ms(r.wall_ms))),
        r.worker_id ? h("span", { class: "muted" }, `worker ${r.worker_id}`) : null),
      r.error ? h("p", { style: "color:var(--critical-text)" }, typeof r.error === "string" ? r.error : JSON.stringify(r.error)) : null,
      r.output !== undefined && r.output !== null ? pre(typeof r.output === "string" ? (() => { try { return JSON.parse(r.output); } catch { return r.output; } })() : r.output) : null,
    );
    const go = h("button", { class: "primary", onclick: async () => {
      let x;
      try { x = JSON.parse(ta.value); } catch (err) { return toast(`input: ${err.message}`); }
      go.disabled = true;
      out.replaceChildren(h("span", { class: "muted" }, "submitted… (a cold start can take minutes; Runpod holds /runsync ~90 s, then this polls)"));
      try {
        let r = await api(`/api/serverless/${e.id}/invoke`, { method: "POST", body: e.mode === "lb" ? x : { input: x } });
        show(r);
        if (r.done === false && r.job) {
          for (let i = 0; i < 400; i++) {
            await new Promise((res) => setTimeout(res, 3000));
            const j = (await api(`/api/serverless/${e.id}/jobs/${r.job}`)).job;
            show({ ...j, output: j.output, error: j.error });
            if (j.finished_at) break;
          }
        }
      } catch (err) { out.replaceChildren(h("span", { style: "color:var(--critical-text)" }, err.message)); } finally { go.disabled = false; }
    } }, "Send");
    return card("Test invoke", h("p", { class: "small muted" }, e.mode === "lb" ? "A request through Runpod's load balancer ({method, path, body})." : "A queue job with the native envelope (/runsync): info, http (one request into fv-serve's router) or stream."),
      h("div", { class: "row", style: "margin-bottom:6px" }, presets.map(([n, v]) => h("button", { class: "ghost", onclick: () => { ta.value = JSON.stringify(v, null, 1); } }, n))),
      ta, h("div", { class: "row", style: "margin-top:6px" }, go), out);
  }

  // ---- cancel and purge (docs/control/serverless.md "Cancel and purge"). The outcome survives the page's redraws.
  const outcomes = new Map();
  const TERMINAL = ["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"];
  function renderOutcome(epId) {
    const box = document.getElementById("slsCancelOut");
    const o = outcomes.get(epId);
    if (!box || !o) return;
    if (o.purge) {
      const p = o.purge;
      box.replaceChildren(h("p", {}, badge("purged", "good"), ` removed ${p.removed ?? "?"} queued job(s); queue ${p.queued_before} → ${p.queued_after ?? "?"}, ${p.in_progress ?? "?"} running. ${p.note}`));
      return;
    }
    const r = o.cancel;
    const kind = (st) => (st === "CANCELLED" || st === "COMPLETED" ? "good" : st === "FAILED" || st === "TIMED_OUT" ? "critical" : "warn");
    const fvLine = r.fv_cancel?.sent
      ? h("div", {}, "fv-serve job ", h("code", {}, r.fv_job.id), ` (${r.fv_job.api}): ${r.fv_cancel.route} sent as queue job ${r.fv_cancel.runpod_job || "?"} `, o.fvStatus ? badge(o.fvStatus, kind(o.fvStatus)) : null, ` · ${r.fv_cancel.reason}`)
      : h("div", { class: "muted" }, r.fv_job ? `fv-serve job ${r.fv_job.id} (${r.fv_job.api}): ` : "fv-serve: ", r.fv_cancel?.reason || "");
    box.replaceChildren(
      h("div", { class: "row", style: "flex-wrap:wrap;gap:10px" }, h("code", {}, r.runpod_job), h("span", {}, `${r.before} →`), badge(o.status || r.status, kind(o.status || r.status)), h("span", { class: "muted" }, r.note)),
      fvLine,
    );
  }
  async function followCancel(e, r) {
    const o = { cancel: r, status: r.status, fvStatus: null };
    outcomes.set(e.id, o);
    renderOutcome(e.id);
    // Status updates: the job's row (and the fv-serve cancel's queue job) until both finish, ~3 min at most.
    for (let i = 0; i < 60; i++) {
      const doneJob = !r.job || TERMINAL.includes(o.status);
      const doneFv = !r.fv_cancel?.job || TERMINAL.includes(o.fvStatus);
      if (doneJob && doneFv) break;
      await new Promise((res) => setTimeout(res, 3000));
      if (outcomes.get(e.id) !== o) return;
      if (!doneJob) o.status = (await api(`/api/serverless/${e.id}/jobs/${r.job}`).catch(() => null))?.job?.status || o.status;
      if (!doneFv) o.fvStatus = (await api(`/api/serverless/${e.id}/jobs/${r.fv_cancel.job}`).catch(() => null))?.job?.status || o.fvStatus;
      renderOutcome(e.id);
    }
  }
  async function cancelJob(e, job) {
    const x = await (await fvEditor()).askSlsCancel(api, e.name, job || "");
    if (!x) return;
    const { job: id, ...rest } = x;
    try {
      const r = await act("cancel", () => api(`/api/serverless/${e.id}/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST", body: rest }));
      followCancel(e, r);
      route(); // the queue, the invokes and the audit as they are now (the outcome stays)
    } catch (err) {
      const box = document.getElementById("slsCancelOut");
      if (box) box.replaceChildren(h("span", { style: "color:var(--critical-text)" }, err.message));
    }
  }
  async function purge(e) {
    let q;
    try { q = await api(`/api/serverless/${e.id}/queue`); } catch (err) { return toast(`queue: ${err.message}`); }
    const x = await (await fvEditor()).askPurge(api, e.name, q.queued, q.in_progress);
    if (!x) return;
    try {
      const r = await act("purge", () => api(`/api/serverless/${e.id}/purge`, { method: "POST", body: x }));
      outcomes.set(e.id, { purge: r });
      renderOutcome(e.id);
      route();
    } catch (err) {
      const box = document.getElementById("slsCancelOut");
      if (box) box.replaceChildren(h("span", { style: "color:var(--critical-text)" }, err.message));
    }
  }
  function queueCard(e, hh) {
    const out = h("div", { id: "slsCancelOut", class: "small", style: "margin-top:8px" });
    const c = card("Queue: cancel and purge",
      h("p", { class: "small muted" }, `Now: ${queueLine(hh)}. Cancel drops a queued job, or has the worker running it stop it; a job that reached a worker also gets fv-serve's cancel (DELETE of the job it created) as a queue job when fv-control knows that job. Purge drops every queued job; running ones stay.`),
      h("div", { class: "row" },
        h("button", { onclick: () => cancelJob(e) }, "Cancel a job…"),
        h("button", { class: "danger", disabled: e.mode !== "queue", onclick: () => purge(e) }, `Purge queue (${hh?.jobs?.inQueue ?? "?"} queued)…`)),
      out);
    setTimeout(() => renderOutcome(e.id));
    return c;
  }

  function workersCard(e, d) {
    const ws = d.runpod?.workers || [];
    const out = h("div", {});
    const load = async (w) => {
      out.replaceChildren(h("span", { class: "muted small" }, "loading…"));
      try {
        const r = await api(`/api/serverless/${e.id}/logs${w ? `?worker=${encodeURIComponent(w)}` : ""}`);
        out.replaceChildren(r.note ? h("p", { class: "small muted" }, r.note) : null, pre([...(r.system || []), ...(r.container || []), ...((r.stored || []).map((x) => `${new Date(x.ts).toISOString()} ${x.msg}`))].join("\n") || "(empty)"),
          h("p", { class: "small muted" }, `Also in the log store: source ${d.log_source}, pod ${r.worker || "-"}.`));
      } catch (err) { out.replaceChildren(h("span", { style: "color:var(--critical-text)" }, err.message)); }
    };
    return card("Workers and logs",
      table([{ label: "worker", get: (w) => w.id }, { label: "state", get: (w) => w.desiredStatus || "–" }, { label: "$/hr", get: (w) => fmt$(w.costPerHr, 3), num: true }, { label: "dc", get: (w) => w.dc || "–" }, { label: "", get: (w) => h("button", { class: "ghost", onclick: () => load(w.id) }, "Logs") }], ws, "No worker is up (the endpoint scales from zero)."),
      out);
  }

  /** The spec as JSON (ui/forms/serverless.ts mountSpecEditor): checked as you type against the schema and the server's validator; Save off while invalid. */
  function specCard(e) {
    const host = h("div", { class: "muted small" }, "loading the editor…");
    fvEditor().then((E) => E.mountSpecEditor(host, { api, toast, endpoint: e, onSaved: () => route() })).catch((err) => host.replaceChildren(h("p", { class: "small" }, err.message)));
    return card("Spec", h("p", { class: "small muted" }, "Image, env and config update the template (Runpod rolls the workers); scaling and placement update the endpoint. Mode, compute, volume and CPU flavors need a new endpoint."), host);
  }

  function jobsCard(d, e) {
    const open = (j) => j.job_id && !j.finished_at && !/^cancel:/.test(j.route);
    return card("Recent invokes",
      table([
        { label: "when", get: (j) => ago(j.submitted_at) },
        { label: "route", get: (j) => j.route },
        { label: "Runpod job", get: (j) => (j.job_id ? h("code", {}, j.job_id) : "–") },
        { label: "status", get: (j) => j.status },
        { label: "cold", get: (j) => (j.cold ? "yes" : "") },
        { label: "queue wait", get: (j) => ms(j.delay_ms), num: true },
        { label: "exec", get: (j) => ms(j.exec_ms), num: true },
        { label: "end to end", get: (j) => ms(j.wall_ms), num: true },
        { label: "by", get: (j) => j.actor },
        { label: "", get: (j) => (e && (open(j) || (j.route === "runsync" || j.route === "run") && /"kind":"http"/.test(j.input || "") && j.status === "COMPLETED" && !/"wait":true/.test(j.input || "")) ? h("button", { class: "ghost", onclick: () => cancelJob(e, j.job_id) }, "Cancel") : "") },
      ], d.jobs || [], "No test invokes yet."));
  }
  function costCard(d) {
    return card("Cost (Runpod billing)", table([{ label: "day", get: (c) => c.day }, { label: "cost", get: (c) => fmt$(c.usd, 4), num: true }, { label: "worker min", get: (c) => c.minutes, num: true }], d.costs || [], "No billed time yet (Runpod's billing lags a few minutes)."));
  }
  function auditCard(d) {
    return card("Audit", table([{ label: "when", get: (a) => dt(a.at) }, { label: "who", get: (a) => a.actor }, { label: "action", get: (a) => a.action }, { label: "ok", get: (a) => (a.ok ? "ok" : "failed") }, { label: "detail", get: (a) => a.detail || "", wrap: true }], d.audit || []));
  }

  routes.push([/^#\/serverless$/, pageServerless]);
})();
