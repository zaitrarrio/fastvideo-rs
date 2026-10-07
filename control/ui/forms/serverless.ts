// The serverless pages' forms (#/serverless): the new-endpoint form and its
// JSON tab, the spec editor of an endpoint, the scale card. Every field is
// bound to the serverless-endpoint schema; the server's validator (POST
// /api/serverless/validate) checks the cross-field rules (lb only on GPU, the
// volume's data centre, workers_min ≤ workers_max, the name's uniqueness).
import { badge, h } from "../dom";
import { createJsonEditor } from "../editor";
import { createForm, type Api, type FieldOpts, type Form } from "../fields";
import type { Issue, Path } from "../schema";
import { issuesOf, loadDyn, loadSchemas, preflight } from "./common";

const pretty = (v: unknown) => JSON.stringify(v, null, 2);

export async function mountEndpointForm(host: HTMLElement, o: { api: Api; toast: (m: string) => void; defaults: any; onCreated: (id: string) => void }) {
  const { api } = o;
  const [schemas, dyn] = await Promise.all([loadSchemas(api), loadDyn(api)]);
  const schema = schemas["serverless-endpoint"]!;
  const create = h("button", { type: "button", class: "primary" }, "Create endpoint");
  const out = h("div", { class: "small", style: "margin-top:8px" });
  const remote = async (v: any) => {
    const r = await api("/api/serverless/validate", { method: "POST", body: { spec: v } });
    return { issues: r.ok ? [] : issuesOf(r), warnings: r.warnings || [] };
  };
  let f: Form;
  let mode: "form" | "json" = "form";
  let jsonOk = true;
  let jsonIssues: Issue[] = [];
  const formBody = h("div", {});
  const jsonBox = h("div", { hidden: true, class: "sv-json" });
  const jsonMsg = h("div", { class: "small" });
  const tabs = h("div", { class: "tabs", role: "tablist" });
  f = createForm({ schemaName: "serverless-endpoint", schema, dyn, api, value: { ...o.defaults, name: "" }, submit: create, remote, onChange: () => mode === "form" && editor?.set(pretty(f.value())) });
  let editor: ReturnType<typeof createJsonEditor> | null = null;
  const src = () => (f.get(["image", "sha"]) !== undefined ? "sha" : f.get(["image", "ref"]) !== undefined ? "ref" : "channel");
  const volOpts = [...(dyn.volumes || []).map((v: any) => ({ value: v.id, detail: `${v.dc} (weights)` })), { value: null, label: "none" }];
  const F = (p: Path, fo: FieldOpts = {}) => f.field(p, fo);
  function draw() {
    const cpu = f.get(["compute"]) === "CPU";
    const s = src();
    formBody.replaceChildren(
      h(
        "div",
        { class: "cf-fields" },
        F(["name"], { label: "Name", nameCheck: "endpoint", placeholder: "e.g. fake-test", help: h("span", {}, "The Runpod endpoint is ", h("code", {}, `fvc-${f.get(["name"]) || "<name>"}`), "; unique among the live endpoints.") }),
        F(["variant"], {
          label: "Image variant",
          allowUnset: false,
          onSet: async (v) => {
            // A variant switch takes that variant's defaults (compute, GPU / CPU fields, volume), keeping the name.
            const d = (await api(`/api/serverless/defaults?name=${encodeURIComponent(f.get(["name"]) || "x")}&variant=${encodeURIComponent(v)}`)).spec;
            for (const k of ["compute", "gpu_types", "gpu_count", "allowed_cuda", "cpu_flavors", "vcpu", "network_volume", "data_centers"]) f.set([k], d[k], true);
            if (d.compute === "CPU") f.set(["mode"], "queue", true);
            draw();
          },
        }),
        F(["compute"], { label: "Compute", onSet: () => draw(), help: "cpu: CPU workers (fake engine); the CUDA variants need GPUs." }),
        F(["mode"], { label: "Mode", onSet: (v) => (v === "lb" && f.set(["scaler_type"], "REQUEST_COUNT", true), draw()) }),
      ),
      h(
        "div",
        { class: "cf-fields", style: "margin-top:10px" },
        h(
          "div",
          { class: "cf-field", "data-schema-ignore": "" },
          h("label", {}, "Image source"),
          h(
            "div",
            { class: "cf-seg", role: "radiogroup", "aria-label": "Image source" },
            ...(["channel", "sha", "ref"] as const).map((k) =>
              h("button", { type: "button", role: "radio", "aria-checked": String(s === k), onclick: () => (f.set(["image"], { [k]: k === "channel" ? "stable" : "" }, true), draw()) }, k === "channel" ? "release channel" : k === "sha" ? "git commit" : "image ref"),
            ),
          ),
        ),
        s === "channel" ? F(["image", "channel"], { label: "Channel", allowUnset: false }) : s === "sha" ? F(["image", "sha"], { label: "Commit" }) : F(["image", "ref"], { label: "Image reference" }),
        F(["config"], { label: "Worker config in the image", placeholder: "default: the variant's", help: "Default: the variant's baked config." }),
      ),
      h(
        "div",
        { class: "cf-fields", style: "margin-top:10px" },
        ...(cpu
          ? [F(["cpu_flavors"], { label: "CPU flavors (in order)" }), F(["vcpu"], { label: "vCPUs per worker" })]
          : [F(["gpu_types"], { label: "GPU types (priority order)" }), F(["gpu_count"], { label: "GPUs per worker", unsetLabel: "1" }), F(["allowed_cuda"], { label: "CUDA versions a host may offer", help: "The images need 13.0." })]),
        F(["network_volume"], { label: "Network volume", options: volOpts, allowUnset: false, onSet: () => draw(), help: "EU weights volume jg48s6o1w0 (CLAUDE.md: EU only). With a volume the workers run in its data centre." }),
        F(["data_centers"], { label: "Data centres", help: f.get(["network_volume"]) ? "With a volume: its data centre only." : "Any of these (none: any)." }),
      ),
      h(
        "div",
        { class: "cf-fields", style: "margin-top:10px" },
        F(["workers_min"], { label: "Workers min (always on, billed while idle)" }),
        F(["workers_max"], { label: "Workers max" }),
        F(["idle_timeout_s"], { label: "Idle timeout" }),
        F(["execution_timeout_s"], { label: "Execution timeout" }),
        F(["scaler_type"], { label: "Scaler", onSet: () => draw() }),
        F(["scaler_value"], { label: f.get(["scaler_type"]) === "REQUEST_COUNT" ? "Jobs per worker" : "Queue delay", unit: f.get(["scaler_type"]) === "REQUEST_COUNT" ? "jobs" : "s" }),
        F(["flashboot"], { label: "FlashBoot", placeholder: "faster warm starts" }),
        F(["container_disk_gb"], { label: "Container disk" }),
        F(["deadline_min"], { label: "Backstop after", unsetLabel: "no backstop", allowUnset: true, placeholder: "none" }),
        F(["deadline_action"], { label: "At the backstop" }),
      ),
      h("div", { class: "cf-fields", style: "margin-top:10px" }, F(["env"], { label: "Env (plain values; secrets: Runpod secrets)", env: { shape: "plain", secrets: false, reserved: dyn.sls_reserved_env_keys || [] } })),
    );
    f.prune();
  }
  function drawTabs() {
    tabs.replaceChildren(
      h("button", { type: "button", class: mode === "form" ? "on" : "", onclick: () => switchTo("form") }, "Form"),
      h("button", { type: "button", class: mode === "json" ? "on" : "", onclick: () => switchTo("json") }, "JSON"),
    );
  }
  function switchTo(m: "form" | "json") {
    if (m === "form" && !jsonOk) return o.toast("fix the JSON first: the form needs a valid document");
    mode = m;
    formBody.hidden = m !== "form";
    jsonBox.hidden = m !== "json";
    if (m === "json") {
      if (!editor)
        editor = createJsonEditor({
          parent: jsonBox,
          doc: pretty(f.value()),
          schema,
          dynamic: dyn,
          extraIssues: () => [...f.problems(), ...jsonIssues],
          onChange: (t) => {
            try {
              const v = JSON.parse(t);
              jsonOk = true;
              jsonIssues = [];
              for (const k of Object.keys(f.value())) if (!(k in v)) f.set([k], undefined, true);
              for (const [k, x] of Object.entries(v)) f.set([k], x, true);
            } catch (e) {
              jsonOk = false;
              jsonIssues = [{ path: [], message: `not valid JSON: ${(e as Error).message}` }];
              create.disabled = true;
            }
            jsonMsg.textContent = jsonOk ? "" : jsonIssues[0]!.message;
          },
        });
      else editor.set(pretty(f.value()));
      jsonBox.append(jsonMsg, h("p", { class: "small muted" }, "The full spec (GET /api/schemas/serverless-endpoint), checked as you type; Create stays off while it is invalid."));
    } else draw();
    drawTabs();
  }
  create.addEventListener("click", async () => {
    if (!jsonOk || !(await f.ready())) return o.toast("fix the problems first");
    const s = f.value();
    create.disabled = true;
    out.replaceChildren(h("span", { class: "muted" }, "image preflight…"));
    const pf = await preflight(api, { endpoint: s });
    if (!pf.ok) {
      out.replaceChildren(badge("image preflight failed", "critical"), ...pf.errors.map((e) => h("p", { class: "small" }, e)));
      create.disabled = false;
      return;
    }
    if (!confirm(`Create the Runpod serverless endpoint fvc-${s.name}? Workers bill while they run.`)) return void (create.disabled = false);
    try {
      const r = await api("/api/serverless", { method: "POST", body: { spec: s } });
      o.onCreated(r.endpoint.id);
    } catch (e) {
      out.replaceChildren(h("span", { style: "color:var(--critical-text)" }, (e as Error).message));
      create.disabled = false;
    }
  });
  draw();
  drawTabs();
  f.el.append(tabs, formBody, jsonBox, f.summary, h("div", { class: "row", style: "margin-top:10px" }, create), out);
  host.replaceChildren(f.el);
  return f;
}

/** An endpoint's spec as JSON, checked as you type (the schema here, the server's validator for the rest); Save stays off while invalid. */
export async function mountSpecEditor(host: HTMLElement, o: { api: Api; toast: (m: string) => void; endpoint: { id: string; spec: any }; onSaved: () => void }) {
  const { api } = o;
  const [schemas, dyn] = await Promise.all([loadSchemas(api), loadDyn(api)]);
  const save = h("button", { type: "button", class: "primary", disabled: true }, "Save");
  const status = h("div", { class: "small" });
  let issues: Issue[] = [];
  let doc: any = o.endpoint.spec;
  let t: any;
  let seq = 0;
  const check = (text: string) => {
    clearTimeout(t);
    save.disabled = true;
    try {
      doc = JSON.parse(text);
    } catch (e) {
      issues = [{ path: [], message: `not valid JSON: ${(e as Error).message}` }];
      status.replaceChildren(badge("invalid JSON", "critical"), " ", issues[0]!.message);
      ed.relint();
      return;
    }
    const my = ++seq;
    status.replaceChildren(h("span", { class: "muted" }, "checking…"));
    t = setTimeout(async () => {
      const r = await api("/api/serverless/validate", { method: "POST", body: { spec: doc, id: o.endpoint.id } }).catch((e) => ({ ok: false, error: e.message }));
      if (my !== seq) return;
      issues = r.ok ? [] : issuesOf(r);
      ed.relint();
      const same = JSON.stringify(doc) === JSON.stringify(o.endpoint.spec);
      save.disabled = !r.ok || same;
      status.replaceChildren(
        r.ok ? badge("valid", "good") : badge(`${issues.length} problem(s)`, "critical"),
        " ",
        r.ok ? (same ? "no changes" : "") : "",
        r.ok ? "" : h("ul", { class: "ff-issues" }, ...issues.map((i) => h("li", {}, h("code", {}, i.path.join(".") || "spec"), " ", i.message))),
      );
    }, 300);
  };
  const box = h("div", { "data-schema-form": "serverless-endpoint-json" });
  const ed = createJsonEditor({ parent: box, doc: pretty(o.endpoint.spec), schema: schemas["serverless-endpoint"], dynamic: dyn, extraIssues: () => issues, onChange: check });
  save.addEventListener("click", async () => {
    try {
      await api(`/api/serverless/${o.endpoint.id}`, { method: "PUT", body: { spec: doc } });
      o.toast("update: ok");
      o.onSaved();
    } catch (e) {
      o.toast(`update: ${(e as Error).message}`);
    }
  });
  host.replaceChildren(box, status, h("div", { class: "row", style: "margin-top:6px" }, save));
  check(pretty(o.endpoint.spec));
}

/** The scale card's two numbers (serverless-scale), workers_min ≤ workers_max and the policy's limits checked before Apply. */
export async function mountScaleForm(host: HTMLElement, o: { api: Api; toast: (m: string) => void; endpoint: { id: string; spec: any }; policy?: { max_workers: number }; onDone: () => void }) {
  const { api } = o;
  const schemas = await loadSchemas(api);
  const apply = h("button", { type: "button", class: "primary" }, "Apply");
  const f = createForm({
    schemaName: "serverless-scale",
    schema: schemas["serverless-scale"]!,
    api,
    value: { workers_min: o.endpoint.spec.workers_min, workers_max: o.endpoint.spec.workers_max },
    submit: apply,
    remote: async (v) => {
      const r = await api("/api/serverless/validate", { method: "POST", body: { spec: { ...o.endpoint.spec, ...v }, id: o.endpoint.id } });
      return { issues: r.ok ? [] : issuesOf(r).filter((i: Issue) => ["workers_min", "workers_max"].includes(String(i.path[0]))) };
    },
  });
  apply.addEventListener("click", async () => {
    if (!(await f.ready())) return;
    try {
      await api(`/api/serverless/${o.endpoint.id}/scale`, { method: "POST", body: f.value() });
      o.toast("scale: ok");
      o.onDone();
    } catch (e) {
      o.toast(`scale: ${(e as Error).message}`);
    }
  });
  f.el.append(h("div", { class: "cf-fields" }, f.field(["workers_min"], { label: "Workers min" }), f.field(["workers_max"], { label: "Workers max", help: o.policy ? `Policy: ${o.policy.max_workers} workers over every endpoint.` : undefined })), f.summary, h("div", { class: "row", style: "margin-top:6px" }, apply));
  host.replaceChildren(f.el);
}
