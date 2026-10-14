// The standalone pod launch form (#/standalone): every field bound to the
// standalone-launch schema, the server's validator (POST
// /api/standalone/validate) for the cross-field rules and the name's
// uniqueness, the image preflight before Launch.
import { badge, h, kids } from "../dom";
import { createForm, type Api, type FieldOpts } from "../fields";
import type { Path } from "../schema";
import { issuesOf, loadDyn, loadProvider, loadSchemas, loadShas, mergeProvider, preflight, providerSummary } from "./common";
import { instanceTable, type InstanceTable, type ITRow } from "./itable";

/** The remote check's issue while no provider is picked (nothing provider-specific is loaded or checked until then). */
export const PICK_PROVIDER = "choose a provider: its GPU types, prices and stock load then";

export async function mountLaunchForm(host: HTMLElement, o: { api: Api; toast: (m: string) => void; onLaunched: (name: string) => void }) {
  const { api } = o;
  // Opening the form calls no provider's API (owner 2026-10-12: the launch screen was slow): the lists without the
  // live ones (?live=0); a provider's GPU types, prices, stock, parked VMs and budget load once it is picked.
  const [schemas, dyn] = await Promise.all([loadSchemas(api), loadDyn(api, "", { live: false })]);
  const launch = h("button", { type: "submit", class: "primary" }, "Launch");
  const out = h("div", { class: "small" });
  const f = createForm({
    schemaName: "standalone-launch",
    schema: schemas["standalone-launch"]!,
    dyn,
    api,
    value: { name: "", preset: "h3-turbo", channel: "stable", region: dyn.regions?.[0]?.id || "eu", deadline_min: 60, env: {} },
    submit: launch,
    // Debounced by the form, the stale one cancelled; ?live=0 until a provider is picked (no provider API call).
    remote: async (v, signal) => {
      const picked = !!v.provider;
      const r = await api(`/api/standalone/validate${picked ? "" : "?live=0"}`, { method: "POST", body: v, signal });
      const issues = issuesOf(r);
      if (!picked && !issues.some((i) => i.path[0] === "provider")) issues.push({ path: ["provider"], message: PICK_PROVIDER });
      return { issues, warnings: r.warnings || [] };
    },
  });
  // The picked provider's details: loaded once per page session (common.ts), the load of an earlier pick cancelled.
  const provStatus = h("div", { class: "small muted cf-provider-status", "data-provider-status": "", role: "status", "aria-live": "polite" });
  let provLoad: AbortController | null = null;
  // The picked provider's instance-type table (owner 2026-10-13): every type it lists; a launchable row's click sets
  // the fields its `launch` names. Loaded when the provider is picked (each pick) and on its Refresh button.
  const tables = new Map<string, InstanceTable>();
  // Runpod's weights (owner 2026-10-13: "I should be able to launch Runpod in any region if I am using R2"): the
  // volume pins the pod to the volume's data centre (region); r2 / none (and a CPU pod with volume off) run anywhere:
  // data_centers, in order, a last "any" letting Runpod place it.
  const volumeDcOf = (region: string | undefined) => (dyn.regions || []).find((r: any) => r.id === (region || "eu"))?.dc ?? "EU-RO-1";
  const isCpu = () => f.get(["compute"]) === "CPU" || (f.get(["compute"]) === undefined && f.get(["variant"]) === "cpu");
  const runpodWeights = (): string | null => (isCpu() ? (f.get(["volume"]) === true ? "volume" : "none") : f.get(["weights_source"]) || presetWeights());
  const presetWeights = (): string => (dyn.pool_presets || []).find((p: any) => p.id === f.get(["preset"]))?.weights_source || "volume";
  /** The pod mounts no volume: any data centre (schemas.ts noVolumePod). */
  const freeDc = () => runpodWeights() !== "volume";
  /** The data centres the pod tries, in order (unset: the volume's, EU-RO-1, the backend default). */
  const dcsTried = (): string[] => (freeDc() ? (f.get(["data_centers"]) as string[] | undefined) ?? [volumeDcOf(undefined)] : [volumeDcOf(f.get(["region"]))]);
  const gpuList = (): string[] => (Array.isArray(f.get(["gpu_types"])) ? f.get(["gpu_types"]) : []);
  const withDc = (dcs: string[], dc: string) => (dcs.includes(dc) ? dcs : [...dcs.filter((d) => d !== "any"), dc, ...(dcs.includes("any") ? ["any"] : [])]);
  const isSel: Record<string, (r: ITRow) => boolean> = {
    // Runpod: the backend tries every (data centre, GPU type) pair (cluster/payloads.ts workerPlacements), so a row is
    // selected when its type and its data centre are both picked.
    runpod: (r) => gpuList().includes(r.name) && !!r.region && dcsTried().includes(r.region),
    brev: (r) => f.get(["provider_gpu"]) === r.name,
    gmi: (r) => f.get(["provider_gpu"]) === r.name && (!f.get(["provider_region"]) || f.get(["provider_region"]) === r.region),
  };
  const tableFor = (p: string) => {
    let t = tables.get(p);
    if (!t) {
      t = instanceTable({
        api,
        provider: p,
        noun: p === "gmi" ? "product" : p === "runpod" ? "GPU type" : "instance type",
        isSelected: (r) => (isSel[p] || (() => false))(r),
        ...(p === "runpod"
          ? {
              weights: runpodWeights,
              onLoad: () => freeDc() && draw(),
              multi: {
                items: () => gpuList().map((g) => ({ key: g, label: g, detail: `in ${dcsTried().map((d) => (d === "any" ? "any data centre" : d)).join(", ")}` })),
                order: (r: ITRow) => (gpuList().indexOf(r.name) + 1 || null),
                move: (i: number, d: -1 | 1) => {
                  const g = [...gpuList()];
                  if (i + d < 0 || i + d >= g.length) return;
                  [g[i], g[i + d]] = [g[i + d]!, g[i]!];
                  f.set(["gpu_types"], g, true);
                  draw();
                },
                remove: (i: number) => {
                  const g = gpuList().filter((_, j) => j !== i);
                  f.set(["gpu_types"], g.length ? g : undefined, true);
                  draw();
                },
              },
            }
          : {}),
        onSelect: (r, how) => {
          if (p === "runpod") {
            if (r && f.get(["compute"]) === "CPU") {
              // CPU → GPU from a row: the preset's weights, r2 when the row's data centre has no volume.
              for (const k of ["cpu_flavors", "vcpu", "volume", "region", "data_centers"]) f.set([k], undefined, true);
              f.set(["compute"], "GPU", true);
              f.set(["weights_source"], r.volume_dc === false && presetWeights() === "volume" ? "r2" : presetWeights(), true);
              if (!freeDc()) f.set(["region"], dyn.regions?.[0]?.id || "eu", true);
            }
            const free = freeDc();
            if (!r) {
              f.set(["gpu_types"], undefined, true);
              if (free) f.set(["data_centers"], undefined, true);
            } else if (how?.append) {
              // A fallback: the type once (another data centre of a picked type adds the data centre, not the type).
              if (!gpuList().includes(r.name)) f.set(["gpu_types"], [...gpuList(), r.name], true);
              if (free && r.region) f.set(["data_centers"], withDc((f.get(["data_centers"]) as string[] | undefined) ?? [], r.region), true);
            } else {
              f.set(["gpu_types"], [r.name], true);
              if (free) f.set(["data_centers"], r.region ? [r.region] : undefined, true);
              else if (r.launch?.region) f.set(["region"], r.launch.region, true);
            }
          } else {
            f.set(["provider_gpu"], r ? r.name : undefined, true);
            if (p === "gmi") f.set(["provider_region"], r ? (r.launch?.provider_region as string | undefined) : undefined, true);
          }
          draw();
        },
      });
      tables.set(p, t);
    }
    return t;
  };
  function pickProvider(p: string | undefined) {
    // A switch cancels the other tables' loads in flight; the picked one loads (again).
    for (const [k, t] of tables) if (k !== p) t.abort();
    if (p) void tableFor(p).load(false);
    provLoad?.abort();
    provLoad = null;
    provStatus.removeAttribute("data-loading");
    if (!p) return void provStatus.replaceChildren("Pick a provider: its GPU types, prices, stock (Brev: parked VMs; GMI / Brev: the budget) load then.");
    const have = dyn.provider_details?.[p];
    if (have) return void provStatus.replaceChildren(providerSummary(have));
    const ac = (provLoad = new AbortController());
    provStatus.setAttribute("data-loading", p);
    provStatus.replaceChildren(h("span", { class: "spin", "aria-hidden": "true" }), ` loading ${p}: GPU types, prices and stock…`);
    loadProvider(api, p, ac.signal)
      .then((d) => {
        if (ac.signal.aborted) return;
        provLoad = null;
        provStatus.removeAttribute("data-loading");
        mergeProvider(dyn, d);
        draw();
      })
      .catch((e) => {
        if (ac.signal.aborted) return;
        provLoad = null;
        provStatus.removeAttribute("data-loading");
        provStatus.replaceChildren(`could not load ${p}'s details: ${(e as Error).message} (pick it again to retry)`);
      });
  }
  const src = () => (f.get(["sha"]) !== undefined ? "sha" : f.get(["image"]) !== undefined ? "image" : "channel");
  const body = h("div", {});
  const F = (p: Path, opts: FieldOpts = {}) => f.field(p, opts);
  const presetOpts = (dyn.pool_presets || []).map((p: any) => ({ value: p.id, detail: p.detail }));
  const catalog = (dyn.catalog_models || []).map((m: any) => ({ value: m, label: m.id, detail: `${m.family} · ${m.recipe}` }));
  // GMI Cloud / NVIDIA Brev (docs/serve/deploy-gmi-brev.md): a provider that is off says why in its option.
  const providerOpts = (dyn.providers || [{ id: "runpod", detail: "" }]).map((p: any) => ({ value: p.id, detail: p.detail }));
  const RUNPOD_ONLY = ["compute", "region", "data_centers", "gpu_types", "cpu_flavors", "vcpu", "volume", "container_disk_gb"];
  const OTHER_ONLY = ["provider_gpu", "provider_region", "weights_download_approved"];
  /** Runpod's weights switched (or the preset / compute changed it): the location fields follow, the table recomputes. */
  function weightsChanged() {
    if (freeDc()) {
      f.set(["region"], undefined, true);
      if (!isCpu()) f.set(["volume"], undefined, true);
    } else {
      f.set(["data_centers"], undefined, true);
      if (f.get(["region"]) === undefined) f.set(["region"], dyn.regions?.[0]?.id || "eu", true);
    }
    draw();
  }
  /** The data centres the location control offers: every one Runpod lists (its location and the picked types' stock),
   * then the rest of fv-control's list, and "any". */
  function dcOptions() {
    const rows = tables.get("runpod")?.view()?.rows || [];
    const picked = new Set(gpuList());
    const info = new Map<string, { where: string; stock: string[] }>();
    for (const r of rows) {
      if (!r.region) continue;
      const x = info.get(r.region) ?? { where: r.cloud?.split(" · ")[1] ?? "", stock: [] };
      if (picked.has(r.name) && r.available) x.stock.push(`${r.gpu_model ?? r.name}: ${r.stock ?? "yes"}`);
      info.set(r.region, x);
    }
    const all = [...new Set([...info.keys(), ...(dyn.data_centers || []).map((d: any) => d.id)])];
    const rank = (d: string) => (info.get(d)?.stock.length ? 0 : info.has(d) ? 1 : 2);
    all.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
    return [
      ...all.map((d) => {
        const x = info.get(d);
        const vol = (dyn.regions || []).some((r: any) => r.dc === d);
        return { value: d, detail: [x?.where, x?.stock.length ? `stock ${x.stock.join(", ")}` : picked.size && x ? "no stock for the picked GPUs" : "", vol ? "weights volume here" : "", x || !rows.length ? "" : "not listed now"].filter(Boolean).join(" · ") };
      }),
      { value: "any", detail: "Runpod places it (last)" },
    ];
  }
  let shasAsked = false;
  function draw() {
    const custom = !f.get(["preset"]);
    const picked: string | undefined = f.get(["provider"]);
    const prov = picked || "runpod";
    const other = prov !== "runpod";
    const cpu = !other && (f.get(["compute"]) === "CPU" || (f.get(["compute"]) === undefined && f.get(["variant"]) === "cpu"));
    if (picked && !provLoad && dyn.provider_details?.[picked]) provStatus.replaceChildren(providerSummary(dyn.provider_details[picked]));
    const where = !picked
      ? []
      : other
      ? [
          tableFor(prov).el,
          F(["provider_gpu"], {
            label: prov === "gmi" ? "Selected GPU product" : "Selected instance type",
            options: (dyn.provider_gpus || []).filter((g: any) => g.provider === prov).map((g: any) => ({ value: g.id, detail: g.detail })),
            onSet: () => tableFor(prov).sync(),
            help: prov === "gmi" ? "Pick a row in the table above (or here): a GMI container product the account may use (GMI_PRODUCTS)." : "Pick a row in the table above (or here): a Brev instance type the account may use (BREV_INSTANCE_TYPES).",
          }),
          ...(prov === "gmi" ? [F(["provider_region"], { label: "IDC", placeholder: (dyn.provider_regions || [])[0]?.id || "the default IDC", help: "GMI data centre (GET /v1/idcs)." })] : []),
          F(["weights_source"], {
            label: "Weights",
            options: [
              { value: "r2", detail: "our R2 weights store, fetched at boot" },
              { value: "hub", detail: "download at boot" },
              { value: "none", detail: "fake engine" },
            ],
            unsetLabel: "default: r2 when every tree is published, else none",
            onSet: () => draw(),
            help: "No Runpod volume here: r2 (our R2 weights store, docs/ops/r2-weights.md; no approval, published trees only), hub (downloaded from the Hub at every boot; the owner approves) or none (the fake engine).",
          }),
          ...(f.get(["weights_source"]) === "hub" ? [F(["weights_download_approved"], { label: "Owner approved the Hub download", help: "Tens to hundreds of GB per boot, paid at the GPU's rate (CLAUDE.md: large downloads need approval)." })] : []),
        ]
      : [
          tableFor("runpod").el,
          F(["compute"], {
            label: "Compute",
            onSet: (v) => (v === "CPU" ? (f.set(["gpu_types"], undefined, true), f.set(["volume"], false, true), f.set(["weights_source"], undefined, true)) : (f.set(["cpu_flavors"], undefined, true), f.set(["vcpu"], undefined, true), f.set(["volume"], undefined, true), f.set(["weights_source"], presetWeights(), true)), weightsChanged()),
            unsetLabel: "the preset's",
          }),
          ...(cpu
            ? [F(["cpu_flavors"], { label: "CPU flavors (in order)" }), F(["vcpu"], { label: "vCPUs", unsetLabel: "2 (default)" })]
            : [
                F(["weights_source"], {
                  label: "Weights",
                  allowUnset: false,
                  options: [
                    { value: "volume", detail: "the EU weights volume (H3 + LTX trees): EU-RO-1 only" },
                    { value: "r2", detail: "our R2 weights store, fetched at boot: any data centre" },
                    { value: "none", detail: "fake engine, no weights: any data centre" },
                  ],
                  onSet: () => weightsChanged(),
                  help: `volume: the network volume (EU bz0ofkfx45 in EU-RO-1, H3 + LTX trees only). r2: the trees from our R2 weights store onto the container disk at boot (docs/ops/r2-weights.md; published trees only, no approval), in any data centre. none: no weights (fake engine). The preset's default: ${presetWeights()}. (Hub downloads at boot are GMI / Brev only.)`,
                }),
              ]),
          ...(freeDc()
            ? [
                F(["data_centers"], {
                  label: "Data centres (in order)",
                  options: dcOptions(),
                  onSet: (v) => {
                    // "any" goes last (the fallback after the named ones).
                    if (Array.isArray(v) && v.includes("any") && v[v.length - 1] !== "any") f.set(["data_centers"], [...v.filter((d) => d !== "any"), "any"], true);
                    draw();
                  },
                  help: `No volume here: any Runpod data centre, tried in order (each with the picked ${cpu ? "flavors" : "GPU types"} in order); "any" lets Runpod place it. None picked: ${volumeDcOf(undefined)}. A row click in the table sets its data centre; "+ fallback" adds one.`,
                }),
              ]
            : [F(["region"], { label: "Region", onSet: () => tableFor("runpod").sync(), help: "The weights volume's region (EU-RO-1). For another data centre: Weights r2 or none." })]),
          ...(cpu ? [F(["volume"], { label: "Weights volume", placeholder: "mount at /workspace (EU-RO-1 only)", onSet: () => weightsChanged(), help: "A CPU pod needs no weights: off, it runs in any data centre." })] : freeDc() ? [] : [F(["volume"], { label: "Weights volume", placeholder: "mount at /workspace", help: "The region's weights volume (EU bz0ofkfx45 in EU-RO-1: H3 + LTX trees only); a GPU pod needs it for the weights." })]),
          F(["container_disk_gb"], { label: "Container disk", unsetLabel: "default", placeholder: cpu ? "10" : "40", help: f.get(["weights_source"]) === "r2" && !cpu ? "Unset: fv-control sizes it for the R2 trees." : undefined }),
        ];
    const s = src();
    // A commit: the commits with images (GHCR) load now, once (not on open).
    if (s === "sha" && !shasAsked) (shasAsked = true), loadShas(api, dyn).then(() => src() === "sha" && draw());
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
              else f.set(["variant"], "cpu", true), f.set(["config"], "/etc/fv/runpod-fake.toml", true), f.set(["fake_models"], ["fake-wan"], true), f.set(["compute"], "CPU", true), f.set(["volume"], false, true);
              // Runpod: the new preset's default weights (the backend's: standalone.ts presetWeightsDefault).
              if ((f.get(["provider"]) || "runpod") === "runpod") f.set(["weights_source"], isCpu() ? undefined : presetWeights(), true), weightsChanged();
              else draw();
            },
            help: "A preset brings its image variant, worker config and models.",
          }),
          ...(custom
            ? [
                F(["variant"], { label: "Image variant", allowUnset: false, onSet: (v) => (f.set(["compute"], v === "cpu" ? "CPU" : "GPU", true), v === "cpu" ? f.set(["models"], undefined, true) : f.set(["fake_models"], undefined, true), (f.get(["provider"]) || "runpod") === "runpod" ? (f.set(["volume"], v === "cpu" ? false : undefined, true), f.set(["weights_source"], v === "cpu" ? undefined : presetWeights(), true), weightsChanged()) : draw()) }),
                F(["config"], { label: "Worker config in the image", placeholder: "/etc/fv/runpod.toml" }),
                f.get(["variant"]) === "cpu" ? F(["fake_models"], { label: "Fake-engine models" }) : F(["models"], { label: "Models", kind: "chips", options: catalog, help: "The catalog's models (family and recipe filled in)." }),
              ]
            : []),
          F(["upsampler_local"], {
            label: "Prompt upsampler (local)",
            placeholder: "llama-server sidecar on the GPU",
            help: "Qwen3-VL-4B-Instruct (vision) in llama-server next to fv-serve on the same GPU (see the doc for its GPU memory; its 5.1 GB GGUF from the R2 weights store). If it fails, fv-serve starts anyway and upsampling falls back. Default off; GPU only. docs/serve/upsampler-local.md.",
          }),
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
          F(["provider"], {
            label: "Provider",
            options: providerOpts,
            unsetLabel: "choose a provider…",
            allowUnset: true,
            onSet: (v) => {
              // Each provider's own fields: the other side's are cleared (the server refuses a mix).
              for (const k of v && v !== "runpod" ? RUNPOD_ONLY : OTHER_ONLY) f.set([k], undefined, true);
              // The weights: Runpod's default per preset (volume / r2), GMI / Brev their own default (unset).
              f.set(["weights_source"], v === "runpod" && !isCpu() ? presetWeights() : undefined, true);
              pickProvider(v || undefined);
              if (v === "runpod") weightsChanged();
              else draw();
            },
            help: "Runpod (the EU weights volume in EU-RO-1, or the R2 weights store / no weights in any data centre), or GMI Cloud / NVIDIA Brev: docs/serve/deploy-gmi-brev.md. Its GPU types, prices and stock load once picked.",
          }),
          provStatus,
          ...where,
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
    if (picked) tableFor(prov).sync();
  }
  pickProvider(undefined);
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
