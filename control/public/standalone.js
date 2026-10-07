// fv-control: the Standalone pods page (docs/control/standalone-pods.md).
// Self-contained: it adds its route to app.js's router and uses app.js's
// helpers (h, api, act, card, table, badge, …), loaded before it. Every API
// value reaches the DOM through textContent.
//   #/standalone            the list and the launch form
//   #/standalone?id=<name>  one pod: status, boot, cost, operation, log tail
"use strict";

/** A boot timeline card for a controller pod (GET /api/pods/:id/boot): each phase's time since create and duration, the phase it is in. Also used by the pod page. */
function fvBootTimeline(podId) {
  const body = h("div", { class: "muted small" }, "loading…");
  const box = card("Boot timeline", body);
  api(`/api/pods/${encodeURIComponent(podId)}/boot`)
    .then((r) => {
      const rows = r.timeline.filter((x) => x.at !== null);
      body.replaceWith(
        h(
          "div",
          {},
          r.phase ? h("p", { class: "small" }, badge(`now: ${r.phase.phase}`, "warn"), ` for ${dur(r.phase.since_s)}`) : null,
          table(
            [
              { label: "phase", get: (x) => x.phase },
              { label: "after create", get: (x) => (x.t_s === null ? "–" : `${x.t_s.toFixed(1)} s`), num: true },
              { label: "took", get: (x) => (x.took_s === null ? "–" : `${x.took_s.toFixed(1)} s`), num: true },
              { label: "at (UTC)", get: (x) => dt(x.at).slice(11) },
              { label: "", get: (x) => x.detail || "", wrap: true },
            ],
            rows,
            "No phases seen yet.",
          ),
        ),
      );
    })
    .catch((e) => body.replaceWith(h("p", { class: "muted small" }, e.message === "not a controller pod" ? "Not a controller pod: no boot timeline." : e.message)));
  return box;
}

(function registerStandalone() {
  const BOOT_KIND = { running: "good", pulling: "warn", starting: "warn", error: "serious", crashloop: "critical", image_error: "critical", stuck: "critical", gone: "critical" };
  const poolKind = (s) => (s === "ready" ? "good" : s === "starting" ? "warn" : s === "failed" || s === "no_stock" ? "critical" : "");
  const imageText = (img) => (img?.channel ? `channel ${img.channel}` : img?.sha ? `sha ${img.sha}` : img?.image || img?.ref || "–");

  async function pageStandalone(main) {
    const id = new URLSearchParams(location.hash.split("?")[1] || "").get("id");
    if (id) return pageStandaloneOne(main, id);
    const list = await api("/api/standalone");
    const draw = async () => {
      const r = await api("/api/standalone");
      listHost.replaceChildren(listTable(r.pods));
    };
    const listHost = h("div", {}, listTable(list.pods));
    main.replaceChildren(
      h("h1", {}, "Standalone pods"),
      h("p", { class: "muted small" }, "One pod on its own, not part of a cluster: the same price check, image preflight, deadline backstop, balance floor, idle stop, cost ledger (owner pod:<name>) and logs from boot as cluster pods. Stop deletes the pod and keeps its definition; Start makes a new pod."),
      card("Pods", listHost),
      card("Launch", launchForm(() => draw())),
    );
    every(10000, () => draw().catch(() => {}));
  }

  function listTable(pods) {
    return table(
      [
        { label: "name", get: (p) => h("a", { href: `#/standalone?id=${encodeURIComponent(p.name)}` }, p.name) },
        { label: "status", get: (p) => badge(p.op ? `${p.op.kind}…` : p.status, statusKind(p.status)) },
        { label: "boot", get: (p) => (p.pod?.boot ? badge(p.pod.boot.phase, BOOT_KIND[p.pod.boot.phase] || "") : p.pool_status && p.pool_status.status !== "ready" ? badge(p.pool_status.status, poolKind(p.pool_status.status)) : "–") },
        { label: "pod", get: (p) => (p.pod ? h("a", { href: `#/pod/${p.pod.pod_id}` }, p.pod.pod_id) : "–") },
        { label: "variant", get: (p) => p.definition.variant },
        { label: "image", get: (p) => imageText(p.definition.image), wrap: true },
        { label: "GPU", get: (p) => p.pod?.gpu || (p.definition.compute === "CPU" ? "cpu" : (p.definition.gpu_types || ["region default"])[0]) },
        { label: "DC", get: (p) => p.pod?.dc || p.definition.dc },
        { label: "$/hr", get: (p) => (p.pod ? fmt$(p.pod.cost_per_hr) : "–"), num: true },
        { label: "today", get: (p) => fmt$(p.cost.today), num: true },
        { label: "total", get: (p) => fmt$(p.cost.total), num: true },
        { label: "deadline", get: (p) => until(p.deadline) },
        { label: "", get: (p) => actions(p, () => route()) },
      ],
      pods,
      "No standalone pods. Launch one below.",
    );
  }

  function actions(p, after) {
    const busy = !!p.op;
    const post = (path, body, label) => act(label, () => api(`/api/standalone/${encodeURIComponent(p.name)}/${path}`, { method: "POST", body: body || {} })).then(() => setTimeout(after, 800));
    return h(
      "div",
      { class: "row" },
      !p.pod && h("button", { disabled: busy, onclick: () => post("start", {}, "start") }, "Start"),
      (p.pod || busy) && h("button", { onclick: () => confirm(`Stop ${p.name}? Its pod is deleted (logs and costs are kept); Start makes a new one.`) && post("stop", {}, "stop") }, "Stop"),
      p.pod && h("button", { disabled: busy, onclick: async () => { const m = await (await fvEditor()).askExtend(api, `Extend ${p.name}`); if (m) post("extend", { minutes: m }, "extend"); } }, "Extend"),
      h("button", { class: "danger", onclick: async () => { if (confirm(`Delete ${p.name}${p.pod ? " (its pod is stopped first)" : ""}?`)) { await act("delete", () => api(`/api/standalone/${encodeURIComponent(p.name)}`, { method: "DELETE" })); location.hash = "#/standalone"; setTimeout(after, 800); } } }, "Delete"),
    );
  }

  /** The launch form (ui/forms/standalone.ts in the editor bundle): every field bound to the standalone-launch schema. */
  function launchForm(after) {
    const host = h("div", { class: "muted small" }, "loading the form…");
    fvEditor()
      .then((E) => E.mountLaunchForm(host, { api, toast, onLaunched: (name) => { location.hash = `#/standalone?id=${encodeURIComponent(name)}`; after(); } }))
      .catch((e) => host.replaceChildren(h("p", { class: "small" }, e.message)));
    return host;
  }

  async function pageStandaloneOne(main, id) {
    const draw = async () => {
      const r = await api(`/api/standalone/${encodeURIComponent(id)}`);
      const p = r.pod;
      const [ops, logs] = await Promise.all([api(`/api/clusters/${encodeURIComponent(p.id)}/ops`).catch(() => ({ operations: [] })), p.pod ? api(`/api/pods/${p.pod.pod_id}/logs?limit=40`).catch(() => ({ lines: [] })) : { lines: [] }]);
      const last = ops.operations[0];
      const kv = (pairs) => h("dl", { class: "kv" }, ...pairs.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v ?? "–")]));
      main.replaceChildren(
        h("div", { class: "row" }, h("a", { href: "#/standalone" }, "← Standalone pods")),
        h("h1", {}, p.name),
        h("div", { class: "row", style: "margin-bottom:10px" }, badge(p.op ? `${p.op.kind}…` : p.status, statusKind(p.status)), actions(p, draw)),
        h(
          "div",
          { class: "grid g2" },
          card(
            "Pod",
            p.pod
              ? kv([
                  ["pod", h("a", { href: `#/pod/${p.pod.pod_id}` }, p.pod.pod_id)],
                  ["URL", p.pod.url],
                  ["Runpod", p.pod.desired_status],
                  ["health", p.pod.health],
                  ["boot", p.pod.boot ? h("span", {}, badge(p.pod.boot.phase, BOOT_KIND[p.pod.boot.phase] || ""), " ", p.pod.boot.detail || "") : "–"],
                  ["GPU / DC", `${p.pod.gpu || "–"} · ${p.pod.dc || "–"}`],
                  ["uptime", dur(p.pod.uptime_s)],
                  ["created", ago(p.pod.created_at)],
                  ["ready", p.pod.ready_at ? ago(p.pod.ready_at) : "not yet"],
                  ["image", p.pod.image],
                ])
              : h("p", { class: "muted small" }, p.pool_status && p.pool_status.status !== "ready" ? `No pod: ${p.pool_status.status}${p.pool_status.detail ? ` (${p.pool_status.detail})` : ""}` : "No pod: Start makes one."),
          ),
          card(
            "Definition and cost",
            kv([
              ["variant", p.definition.variant],
              ["image", imageText(p.definition.image)],
              ["compute", `${p.definition.compute}${p.definition.gpu_types ? ` · ${p.definition.gpu_types.join(", ")}` : ""}`],
              ["region", `${p.definition.region} (${p.definition.dc})${p.definition.volume ? ` · volume ${p.definition.volume}` : ""}`],
              ["config", p.definition.config],
              ["models", p.definition.models.join(", ")],
              ["deadline", p.deadline ? `${dt(p.deadline)} (${until(p.deadline)})` : `${p.definition.deadline_min} min after start`],
              ["idle stop", p.definition.idle_stop_min ? `${p.definition.idle_stop_min} min` : "policy default"],
              ["$/hr now", p.pod ? fmt$(p.pod.cost_per_hr) : "–"],
              ["cost today / total", `${fmt$(p.cost.today)} / ${fmt$(p.cost.total)} (${p.cost.minutes} pod-min)`],
            ]),
          ),
        ),
        p.pod ? fvBootTimeline(p.pod.pod_id) : null,
        card(last ? `Operation: ${last.kind} ${last.status}${last.error ? `: ${last.error}` : ""}` : "Operation", last ? h("div", { class: "log" }, last.log.slice(-60).map((l) => h("div", {}, `${dt(l.at).slice(11)} ${l.msg}`))) : h("p", { class: "muted small" }, "None yet.")),
        card(
          "Log tail (shipped and Runpod, from boot)",
          p.pod ? h("div", { class: "row small", style: "margin-bottom:6px" }, h("a", { class: "btn", href: `#/logs?pod=${p.pod.pod_id}` }, "Logs page"), h("span", { class: "muted" }, `JSON: /api/pods/${p.pod.pod_id}/logs`)) : null,
          logs.lines.length ? h("div", { class: "log" }, logs.lines.map((l) => h("div", {}, `${dt(l.ts).slice(11)} ${l.level.toUpperCase()} ${l.target || ""}: ${l.msg}`))) : h("p", { class: "muted small" }, "No lines yet."),
        ),
      );
    };
    await draw();
    every(10000, () => draw().catch(() => {}));
  }

  routes.push([/^#\/standalone$/, pageStandalone]);
})();
