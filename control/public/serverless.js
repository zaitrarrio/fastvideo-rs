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
  const field = (label, input, help) => h("label", { class: "small", style: "display:flex;flex-direction:column;gap:3px;min-width:160px" }, h("span", { class: "muted" }, label), input, help ? h("span", { class: "muted small" }, help) : null);
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
    );
    every(15000, async () => {
      if (!location.hash.startsWith("#/serverless") || location.hash.includes("ep=")) return;
      const fresh = await api(`/api/serverless${q.get("all") ? "?all=1" : ""}`).catch(() => null);
      if (fresh && JSON.stringify(fresh.endpoints.map((e) => [e.status, e.workers, e.health])) !== JSON.stringify(eps.map((e) => [e.status, e.workers, e.health]))) route();
    });
  }

  function createCard(def) {
    const v = (id) => document.getElementById(id);
    const inp = (id, value, attrs = {}) => h("input", { id, value: value ?? "", ...attrs });
    const sel = (id, opts, cur) => h("select", { id }, opts.map((o) => h("option", { value: o, selected: o === cur }, o)));
    const out = h("div", { class: "small", style: "margin-top:8px" });
    const json = h("textarea", { rows: 18, style: "width:100%;font-family:var(--mono, monospace);font-size:12px", "aria-label": "Spec JSON" });
    json.value = JSON.stringify(def, null, 2);
    const lines = (s) => s.split(/[\n,]/).map((x) => x.trim()).filter(Boolean);
    const fromForm = () => {
      const variant = v("sv_variant").value;
      const spec = {
        name: v("sv_name").value.trim(),
        mode: v("sv_mode").value,
        variant,
        image: { [v("sv_imgkind").value]: v("sv_img").value.trim() },
        workers_min: Number(v("sv_wmin").value),
        workers_max: Number(v("sv_wmax").value),
        idle_timeout_s: Number(v("sv_idle").value),
        execution_timeout_s: Number(v("sv_exec").value),
        flashboot: v("sv_fb").checked,
        scaler_type: v("sv_scaler").value,
        scaler_value: Number(v("sv_scalerv").value),
        deadline_min: v("sv_deadline").value ? Number(v("sv_deadline").value) : null,
        deadline_action: v("sv_dact").value,
      };
      if (variant !== "cpu") {
        const g = lines(v("sv_gpus").value);
        if (g.length) spec.gpu_types = g;
        spec.network_volume = v("sv_vol").value || null;
      }
      const dcs = lines(v("sv_dcs").value);
      if (dcs.length) spec.data_centers = dcs;
      if (v("sv_config").value.trim()) spec.config = v("sv_config").value.trim();
      return spec;
    };
    let mode = "form";
    const form = h(
      "div",
      { class: "row", style: "flex-wrap:wrap;gap:10px;align-items:flex-start" },
      field("name", inp("sv_name", "", { placeholder: "e.g. fake-test", pattern: "[a-z][a-z0-9-]{0,30}" }), "the Runpod endpoint is fvc-<name>"),
      field("variant", sel("sv_variant", ["cpu", "h3-turbo", "h3-max", "ltx", "wan", "wan5b", "sfwan"], "cpu"), "cpu: CPU workers, fake engine"),
      field("mode", sel("sv_mode", ["queue", "lb"], "queue"), "lb: GPU only"),
      field("image", h("div", { class: "row" }, sel("sv_imgkind", ["channel", "sha", "ref"], "channel"), inp("sv_img", "stable", { style: "width:120px" }))),
      field("GPU types (priority order)", h("textarea", { id: "sv_gpus", rows: 3, placeholder: "NVIDIA RTX PRO 6000 Blackwell Server Edition\nNVIDIA GeForce RTX 5090" }), "GPU variants; default RTX PRO 6000"),
      field("data centers", inp("sv_dcs", "", { placeholder: "default: the volume's" })),
      field("network volume", sel("sv_vol", ["jg48s6o1w0", ""], "jg48s6o1w0"), "EU weights (GPU variants)"),
      field("config in image", inp("sv_config", "", { placeholder: "default: the variant's" })),
      field("workers min / max", h("div", { class: "row" }, inp("sv_wmin", "0", { type: "number", min: 0, max: 4, style: "width:60px" }), inp("sv_wmax", "1", { type: "number", min: 0, max: 8, style: "width:60px" }))),
      field("idle timeout s", inp("sv_idle", "5", { type: "number", min: 1, style: "width:80px" })),
      field("execution timeout s", inp("sv_exec", "1800", { type: "number", min: 10, style: "width:90px" })),
      field("scaler", h("div", { class: "row" }, sel("sv_scaler", ["QUEUE_DELAY", "REQUEST_COUNT"], "QUEUE_DELAY"), inp("sv_scalerv", "4", { type: "number", min: 0.5, step: 0.5, style: "width:60px" }))),
      field("FlashBoot", h("input", { id: "sv_fb", type: "checkbox" })),
      field("backstop (min) / action", h("div", { class: "row" }, inp("sv_deadline", "120", { type: "number", min: 5, style: "width:70px" }), sel("sv_dact", ["delete", "scale0"], "delete"))),
    );
    const jsonBox = h("div", { hidden: true }, json, h("p", { class: "small muted" }, "The full spec (GET /api/schemas/serverless-endpoint); missing fields take the variant's defaults."));
    const toggle = h("button", { class: "ghost", onclick: () => {
      if (mode === "form") { try { json.value = JSON.stringify({ ...JSON.parse(json.value), ...fromForm() }, null, 2); } catch { json.value = JSON.stringify(fromForm(), null, 2); } }
      mode = mode === "form" ? "json" : "form";
      form.hidden = mode === "json";
      jsonBox.hidden = mode === "form";
      toggle.textContent = mode === "form" ? "Edit as JSON" : "Back to the form";
    } }, "Edit as JSON");
    const spec = () => (mode === "form" ? fromForm() : JSON.parse(json.value));
    const validateBtn = h("button", { onclick: async () => {
      try {
        const r = await api("/api/serverless/validate", { method: "POST", body: { spec: spec() } });
        out.replaceChildren(r.ok ? h("div", {}, badge("valid", "good"), " with the defaults filled:", pre(r.spec)) : h("div", {}, badge("invalid", "critical"), " ", r.error));
      } catch (e) { out.replaceChildren(h("span", { style: "color:var(--critical-text)" }, e.message)); }
    } }, "Validate");
    const createBtn = h("button", { class: "primary", onclick: async () => {
      let s;
      try { s = spec(); } catch (e) { return toast(`spec: ${e.message}`); }
      if (!confirm(`Create the Runpod serverless endpoint fvc-${s.name}? Workers bill while they run.`)) return;
      createBtn.disabled = true;
      try {
        const r = await act("create", () => api("/api/serverless", { method: "POST", body: { spec: s } }));
        location.hash = `#/serverless?ep=${r.endpoint.id}`;
      } catch (e) { out.replaceChildren(h("span", { style: "color:var(--critical-text)" }, e.message)); } finally { createBtn.disabled = false; }
    } }, "Create endpoint");
    return card("New endpoint", form, jsonBox, h("div", { class: "row", style: "margin-top:10px" }, toggle, validateBtn, createBtn), out);
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
      h("div", { class: "row", style: "margin-top:6px" }, h("a", { class: "btn", href: "#/serverless" }, "← all endpoints"), h("a", { class: "btn", href: `#/logs?q=&pod=${(d.runpod?.workers || [])[0]?.id || ""}` }, "Log explorer")),
    );
    if (!live) {
      main.replaceChildren(head, jobsCard(d), costCard(d), auditCard(d));
      return;
    }
    // Scale / extend / delete.
    const wmin = h("input", { type: "number", min: 0, max: 4, value: e.spec.workers_min, style: "width:60px", "aria-label": "workers min" });
    const wmax = h("input", { type: "number", min: 0, max: 8, value: e.spec.workers_max, style: "width:60px", "aria-label": "workers max" });
    const ctl = card(
      "Scale",
      h("div", { class: "row", style: "flex-wrap:wrap" },
        field("workers min", wmin), field("workers max", wmax),
        h("button", { class: "primary", onclick: async () => { await act("scale", () => api(`/api/serverless/${e.id}/scale`, { method: "POST", body: { workers_min: Number(wmin.value), workers_max: Number(wmax.value) } })); route(); } }, "Apply"),
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
    main.replaceChildren(head, ctl, invokeCard(e), workersCard(e, d), specCard(e), jobsCard(d), costCard(d), auditCard(d));
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

  function specCard(e) {
    const ta = h("textarea", { rows: 14, style: "width:100%;font-family:var(--mono, monospace);font-size:12px", "aria-label": "Endpoint spec" });
    ta.value = JSON.stringify(e.spec, null, 2);
    return card("Spec",
      h("p", { class: "small muted" }, "Image, env and config update the template (Runpod rolls the workers); scaling and placement update the endpoint. Mode, compute, volume and CPU flavors need a new endpoint."),
      ta,
      h("div", { class: "row", style: "margin-top:6px" }, h("button", { class: "primary", onclick: async () => {
        let s;
        try { s = JSON.parse(ta.value); } catch (err) { return toast(`spec: ${err.message}`); }
        await act("update", () => api(`/api/serverless/${e.id}`, { method: "PUT", body: { spec: s } }));
        route();
      } }, "Save")));
  }

  function jobsCard(d) {
    return card("Recent invokes",
      table([
        { label: "when", get: (j) => ago(j.submitted_at) },
        { label: "route", get: (j) => j.route },
        { label: "status", get: (j) => j.status },
        { label: "cold", get: (j) => (j.cold ? "yes" : "") },
        { label: "queue wait", get: (j) => ms(j.delay_ms), num: true },
        { label: "exec", get: (j) => ms(j.exec_ms), num: true },
        { label: "end to end", get: (j) => ms(j.wall_ms), num: true },
        { label: "by", get: (j) => j.actor },
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
