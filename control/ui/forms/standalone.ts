// The standalone pod launch form (#/standalone): every field bound to the
// standalone-launch schema, the server's validator (POST
// /api/standalone/validate) for the cross-field rules and the name's
// uniqueness, the image preflight before Launch.
import { badge, h, kids } from "../dom";
import { createForm, type Api, type FieldOpts } from "../fields";
import type { Path } from "../schema";
import { issuesOf, loadDyn, loadSchemas, preflight } from "./common";

export async function mountLaunchForm(host: HTMLElement, o: { api: Api; toast: (m: string) => void; onLaunched: (name: string) => void }) {
  const { api } = o;
  const [schemas, dyn] = await Promise.all([loadSchemas(api), loadDyn(api)]);
  const launch = h("button", { type: "submit", class: "primary" }, "Launch");
  const out = h("div", { class: "small" });
  const f = createForm({
    schemaName: "standalone-launch",
    schema: schemas["standalone-launch"]!,
    dyn,
    api,
    value: { name: "", preset: "h3-turbo", channel: "stable", region: dyn.regions?.[0]?.id || "eu", deadline_min: 60, env: {} },
    submit: launch,
    remote: async (v) => {
      const r = await api("/api/standalone/validate", { method: "POST", body: v });
      return { issues: issuesOf(r), warnings: r.warnings || [] };
    },
  });
  const src = () => (f.get(["sha"]) !== undefined ? "sha" : f.get(["image"]) !== undefined ? "image" : "channel");
  const body = h("div", {});
  const F = (p: Path, opts: FieldOpts = {}) => f.field(p, opts);
  const presetOpts = (dyn.pool_presets || []).map((p: any) => ({ value: p.id, detail: p.detail }));
  const catalog = (dyn.catalog_models || []).map((m: any) => ({ value: m, label: m.id, detail: `${m.family} · ${m.recipe}` }));
  function draw() {
    const custom = !f.get(["preset"]);
    const cpu = f.get(["compute"]) === "CPU" || (f.get(["compute"]) === undefined && f.get(["variant"]) === "cpu");
    const s = src();
    body.replaceChildren(
      h(
        "div",
        { class: "ff-section" },
        h("h3", {}, "What"),
        h(
          "div",
          { class: "cf-fields" },
          F(["name"], { label: "Name", nameCheck: "cluster", placeholder: "e.g. h3-test", help: "Unique among clusters and standalone pods; the Runpod pod is fv-pod-<name>-<stamp>." }),
          F(["preset"], {
            label: "Preset",
            options: presetOpts,
            unsetLabel: "custom (variant + config)",
            allowUnset: true,
            onSet: (v) => {
              if (v) for (const k of ["variant", "config", "models", "fake_models"]) f.set([k], undefined, true);
              else f.set(["variant"], "cpu", true), f.set(["config"], "/etc/fv/runpod-fake.toml", true), f.set(["fake_models"], ["fake-wan"], true), f.set(["compute"], "CPU", true);
              draw();
            },
            help: "A preset brings its image variant, worker config and models.",
          }),
          ...(custom
            ? [
                F(["variant"], { label: "Image variant", allowUnset: false, onSet: (v) => (f.set(["compute"], v === "cpu" ? "CPU" : "GPU", true), v === "cpu" ? f.set(["models"], undefined, true) : f.set(["fake_models"], undefined, true), draw()) }),
                F(["config"], { label: "Worker config in the image", placeholder: "/etc/fv/runpod.toml" }),
                f.get(["variant"]) === "cpu" ? F(["fake_models"], { label: "Fake-engine models" }) : F(["models"], { label: "Models", kind: "chips", options: catalog, help: "The catalog's models (family and recipe filled in)." }),
              ]
            : []),
        ),
      ),
      h(
        "div",
        { class: "ff-section" },
        h("h3", {}, "Image"),
        h(
          "div",
          { class: "cf-fields" },
          h(
            "div",
            { class: "cf-field", "data-schema-ignore": "" },
            h("label", {}, "Source"),
            h(
              "div",
              { class: "cf-seg", role: "radiogroup", "aria-label": "Image source" },
              ...(["channel", "sha", "image"] as const).map((k) =>
                h(
                  "button",
                  {
                    type: "button",
                    role: "radio",
                    "aria-checked": String(s === k),
                    onclick: () => {
                      for (const x of ["channel", "sha", "image"]) f.set([x], undefined, true);
                      f.set([k], k === "channel" ? "stable" : "", true);
                      draw();
                    },
                  },
                  k === "channel" ? "release channel" : k === "sha" ? "git commit" : "image ref",
                ),
              ),
            ),
          ),
          s === "channel" ? F(["channel"], { label: "Channel", allowUnset: false }) : s === "sha" ? F(["sha"], { label: "Commit", placeholder: "7-40 hex" }) : F(["image"], { label: "Image reference", placeholder: "ghcr.io/owner/repo@sha256:…" }),
        ),
      ),
      h(
        "div",
        { class: "ff-section" },
        h("h3", {}, "Where"),
        h(
          "div",
          { class: "cf-fields" },
          F(["compute"], { label: "Compute", onSet: (v) => (v === "CPU" ? (f.set(["gpu_types"], undefined, true), f.set(["volume"], false, true)) : (f.set(["cpu_flavors"], undefined, true), f.set(["vcpu"], undefined, true)), draw()), unsetLabel: "the preset's" }),
          F(["region"], { label: "Region" }),
          ...(cpu
            ? [F(["cpu_flavors"], { label: "CPU flavors (in order)" }), F(["vcpu"], { label: "vCPUs", unsetLabel: "2 (default)" })]
            : [F(["gpu_types"], { label: "GPU types (in order; none: the region's)", help: "Live from Runpod: $/hr and stock; a type that is gone is flagged." })]),
          F(["volume"], { label: "Weights volume", placeholder: "mount at /workspace", help: "The region's weights volume (EU jg48s6o1w0); a GPU pod needs it for the weights." }),
          F(["container_disk_gb"], { label: "Container disk", unsetLabel: "default", placeholder: cpu ? "10" : "40" }),
        ),
      ),
      h(
        "div",
        { class: "ff-section" },
        h("h3", {}, "Backstops"),
        h(
          "div",
          { class: "cf-fields" },
          F(["deadline_min"], { label: "Deadline after start", help: "The pod is deleted then (the cron and its own watchdog)." }),
          F(["idle_stop_min"], { label: "Idle stop", unsetLabel: "policy default", placeholder: "policy default" }),
          F(["max_gpu_dph"], { label: "Max GPU $/hr", placeholder: "3.6", help: "A pod on a pricier GPU is deleted right after create; the check compares it with the live prices." }),
          F(["auth"], { label: "Client auth" }),
          F(["log_level"], { label: "Most verbose level shipped", unsetLabel: "info (default)" }),
        ),
      ),
      h("div", { class: "ff-section" }, h("h3", {}, "Env"), h("div", { class: "cf-fields" }, F(["env"], { label: "Variables of this pod", env: { shape: "launch", secrets: true } }))),
    );
    f.prune();
  }
  draw();
  const submit = async (ev: Event) => {
    ev.preventDefault();
    if (!(await f.ready())) return o.toast("fix the problems first");
    launch.disabled = true;
    out.replaceChildren(h("span", { class: "muted" }, "image preflight…"));
    const pf = await preflight(api, { launch: f.value() });
    if (!pf.ok) {
      out.replaceChildren(h("div", {}, badge("image preflight failed", "critical"), ...pf.errors.map((e) => h("p", { class: "small" }, e))));
      launch.disabled = false;
      return;
    }
    try {
      const r = await api("/api/standalone", { method: "POST", body: f.value() });
      o.toast("launch: ok");
      o.onLaunched(r.pod.name);
    } catch (e) {
      out.replaceChildren(h("span", { style: "color:var(--critical-text)" }, (e as Error).message));
      launch.disabled = false;
    }
  };
  f.el.append(h("form", { onsubmit: submit, novalidate: true }, body, f.summary, h("div", { class: "row", style: "margin-top:10px" }, launch, out)));
  host.replaceChildren(...kids(f.el));
  return f;
}
