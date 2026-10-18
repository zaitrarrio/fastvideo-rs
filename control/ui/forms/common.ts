// Shared bits of the schema-bound forms: the schemas and live lists (cached
// per page load), the field manifests the coverage test reads, and the
// image preflight before a save or launch.
import { h } from "../dom";
import type { Dynamic } from "../editor";
import type { Api } from "../fields";
import type { Issue, Path, Schema } from "../schema";

let schemas: Promise<Record<string, Schema>> | null = null;
const dyns = new Map<string, Promise<Dynamic>>();
export const loadSchemas = (api: Api) => (schemas ||= api("/api/v1/schemas").then((r) => r.schemas));
/** The lists; `live: false` (the launch forms): no provider's API is called, see loadProvider. */
export const loadDyn = (api: Api, cluster = "", o: { live?: boolean } = {}) => {
  const live = o.live !== false;
  const k = `${cluster}|${live}`;
  const q = [cluster ? `cluster=${encodeURIComponent(cluster)}` : "", live ? "" : "live=0"].filter(Boolean).join("&");
  if (!dyns.has(k)) dyns.set(k, api(`/api/v1/schemas/dynamic${q ? `?${q}` : ""}`).catch(() => ({})));
  return dyns.get(k)!;
};
/** Forget the live lists (after a create: names, endpoints). */
export const resetDyn = () => (dyns.clear(), providers.clear());

/**
 * One launch provider's live details (GET /api/v1/schemas/dynamic?provider=<p>: GPU / instance types with $/hr and stock,
 * GMI's IDC, Brev's parked VMs, the budget), asked only once that provider is picked; kept for the page's session.
 * A load aborted by its signal (the pick changed first) is forgotten, so the next pick asks again.
 */
const providers = new Map<string, Promise<ProviderDetails>>();
export interface ProviderDetails {
  provider: string;
  enabled: boolean;
  reason: string | null;
  live_ok: boolean;
  gpu_types?: any[];
  provider_gpus?: { id: string; detail: string; provider: string }[];
  provider_regions?: { id: string; detail: string }[];
  parked?: any[];
  parked_usd_per_day?: number;
  /** GMI / Brev: the monthly budget (budget_source: policy = entered in the console, worker_var = the Worker's var). */
  budget?: { budget_usd: number | null; budget_source?: "policy" | "worker_var" | null; month_usd: number; running_dph: number; parked_dph: number };
}
export const providerLoaded = (p: string) => providers.has(p);
export function loadProvider(api: Api, p: string, signal?: AbortSignal): Promise<ProviderDetails> {
  if (!providers.has(p)) {
    const pr = api(`/api/v1/schemas/dynamic?provider=${encodeURIComponent(p)}`, { signal });
    providers.set(p, pr);
    pr.catch(() => providers.get(p) === pr && providers.delete(p));
  }
  return providers.get(p)!;
}
let shas: Promise<string[]> | null = null;
/** The commits with images (GHCR tags too: ?images=1), folded into a form's lists once its image source is a commit. */
export async function loadShas(api: Api, dyn: Record<string, any>): Promise<void> {
  shas ||= api("/api/v1/schemas/dynamic?images=1").then((r) => r.shas || [], () => ((shas = null), []));
  const got = await shas;
  dyn.shas = [...new Set([...(dyn.shas || []), ...got])];
}
/** Folds a provider's details into a form's lists (its GPU types; its share of provider_gpus / provider_regions). */
export function mergeProvider(dyn: Record<string, any>, d: ProviderDetails) {
  if (d.gpu_types) dyn.gpu_types = d.gpu_types;
  if (d.provider_gpus) dyn.provider_gpus = [...(dyn.provider_gpus || []).filter((g: any) => g.provider !== d.provider), ...d.provider_gpus];
  if (d.provider === "gmi") dyn.provider_regions = d.provider_regions || [];
  (dyn.provider_details ||= {})[d.provider] = d;
}
/** One line on what was loaded for a provider (under the provider field): budget, parked VMs, a list that failed. */
export function providerSummary(d: ProviderDetails): string {
  const bits: string[] = [];
  if (!d.enabled) return d.reason || `${d.provider} is off`;
  if (d.provider === "runpod") bits.push(d.live_ok ? `${(d.gpu_types || []).length} GPU types with live $/hr and stock` : "Runpod's live GPU list is unavailable now (the types are still offered)");
  else {
    const n = (d.provider_gpus || []).length;
    bits.push(d.live_ok ? `${n} ${d.provider === "brev" ? "instance type" : "product"}${n === 1 ? "" : "s"} with live prices` : "the live price list is unavailable now");
    const b = d.budget;
    if (b) bits.push(`this month $${b.month_usd.toFixed(2)}${b.budget_usd != null ? ` of the $${b.budget_usd} monthly budget${b.budget_source === "worker_var" ? ` (the Worker var ${d.provider.toUpperCase()}_BUDGET_USD)` : ""}` : " (no budget set: launches are refused)"}${b.running_dph ? `, $${b.running_dph.toFixed(2)}/hr running` : ""}`);
    if (d.provider === "brev") bits.push(d.parked?.length ? `${d.parked.length} parked / held VM${d.parked.length > 1 ? "s" : ""} ($${(d.parked_usd_per_day || 0).toFixed(2)}/day storage)` : "no parked VMs");
  }
  return bits.join(" · ");
}

/**
 * GMI / Brev: the provider's monthly budget as an editable field (owner report 2026-10-18: "it is not recognizing the
 * budget I enter, it always sets the budget to $10": the console had no place to enter it, only the Worker var). Saving
 * PUTs /api/v1/providers/<p>/budget (the policy brev_budget_usd / gmi_budget_usd, the store every check reads); the
 * saved value is what the field, the provider line and the next launch's check use. Not a field of the launch
 * document (data-schema-ignore): it is account-wide, not per pod. The element is meant to be kept across redraws.
 */
export function budgetField(api: Api, d: ProviderDetails, onSaved: (d: ProviderDetails) => void): HTMLElement {
  const title = d.provider === "brev" ? "NVIDIA Brev" : d.provider === "gmi" ? "GMI Cloud" : d.provider;
  const cur = () => d.budget?.budget_usd;
  const input = h("input", { type: "number", min: "0", max: "100000", step: "1", value: cur() ?? "", placeholder: "unset: launches are refused", "aria-label": `${title} monthly budget ($)`, "data-budget-input": d.provider }) as HTMLInputElement;
  const save = h("button", { type: "button", "data-budget-save": d.provider }, "Save budget");
  const msg = h("span", { class: "small", "data-budget-msg": d.provider, role: "status" });
  const help = () => {
    const b = d.budget;
    const src = b?.budget_source === "policy" ? "set here (Settings → Policies " + d.provider + "_budget_usd)" : b?.budget_source === "worker_var" ? `from the Worker var ${d.provider.toUpperCase()}_BUDGET_USD until you save one here` : "not set";
    return `The most fv-control may spend on ${title} per UTC month, for every pod and cluster there (not per pod: the per-pod cap is Max GPU $/hr). This month's spend + the running and parked VMs + this launch until its deadline must fit. Now ${cur() != null ? `$${cur()}` : "unset"}, ${src}.`;
  };
  const helpEl = h("span", { class: "cf-help" }, help());
  const dirty = () => input.value.trim() !== String(cur() ?? "");
  const sync = () => (save.disabled = !dirty() || (input.value.trim() !== "" && !(Number(input.value) >= 0)));
  input.addEventListener("input", () => (msg.replaceChildren(), sync()));
  input.addEventListener("keydown", (e) => e.key === "Enter" && (e.preventDefault(), !save.disabled && save.click()));
  save.addEventListener("click", async () => {
    const raw = input.value.trim();
    const v = raw === "" ? null : Number(raw);
    if (v !== null && !(Number.isFinite(v) && v >= 0)) return void msg.replaceChildren("a number of $ (0 or more)");
    save.disabled = true;
    msg.replaceChildren("saving…");
    try {
      const r = await api(`/api/v1/providers/${encodeURIComponent(d.provider)}/budget`, { method: "PUT", body: { budget_usd: v } });
      const pv = r.provider || {};
      d.budget = { budget_usd: pv.budget_usd ?? null, budget_source: pv.budget_source ?? null, month_usd: pv.month_usd ?? d.budget?.month_usd ?? 0, running_dph: pv.running_dph ?? 0, parked_dph: pv.parked_dph ?? 0 };
      input.value = String(d.budget.budget_usd ?? "");
      helpEl.textContent = help();
      msg.replaceChildren(`saved: $${d.budget.budget_usd ?? "unset"} per month`);
      onSaved(d);
    } catch (e) {
      msg.replaceChildren(`not saved: ${(e as Error).message}`);
    }
    sync();
  });
  sync();
  return h("div", { class: "cf-field", "data-schema-ignore": "", "data-budget-field": d.provider }, h("label", {}, `Monthly budget (${title}, $)`), h("div", { class: "row", style: "gap:8px;align-items:center" }, input, save, msg), helpEl);
}

/**
 * Every field each form offers, by schema: the UI coverage test
 * (test/unit/config-validation.test.ts) checks each path exists in its
 * schema and that every schema property is offered or listed in NOT_OFFERED
 * with the reason.
 */
export const FORM_FIELDS: Record<string, { schema: string; fields: Path[]; notOffered?: Record<string, string> }> = {
  "standalone-launch": {
    schema: "standalone-launch",
    fields: [["name"], ["preset"], ["variant"], ["config"], ["models"], ["fake_models"], ["channel"], ["sha"], ["image"], ["compute"], ["cpu_flavors"], ["vcpu"], ["region"], ["data_centers"], ["volume"], ["container_disk_gb"], ["deadline_min"], ["idle_stop_min"], ["max_gpu_dph"], ["auth"], ["log_level"], ["env"], ["provider"], ["provider_gpu"], ["provider_region"], ["weights_source"], ["weights_download_approved"], ["upsampler_local"]],
    notOffered: {
      config_toml: "inline worker configs come with the presets; a custom one is a cluster pool (the cluster page's TOML editor)",
      gpu_types: "the instance table: a row click picks the primary GPU type, \"+ fallback\" appends; the Selected GPUs strip orders them",
      gpu_type: "API shorthand of gpu_types",
      dc: "API shorthand of data_centers (one data centre) / alias of region",
      min_balance: "the account defaults (Settings: policies); API only",
      balance_floor: "the account defaults; API only",
      min_start: "the account defaults; API only",
      start: "the form always starts the pod (Launch)",
      skip_image_check: "never from the UI: the preflight must pass",
    },
  },
  "serverless-endpoint": {
    schema: "serverless-endpoint",
    fields: [["name"], ["preset"], ["mode"], ["image", "channel"], ["image", "sha"], ["image", "ref"], ["variant"], ["compute"], ["config"], ["env"], ["gpu_types"], ["gpu_count"], ["cpu_flavors"], ["vcpu"], ["data_centers"], ["weights_source"], ["weights_download_approved"], ["network_volume"], ["workers_min"], ["workers_max"], ["idle_timeout_s"], ["flashboot"], ["execution_timeout_s"], ["scaler_type"], ["scaler_value"], ["allowed_cuda"], ["container_disk_gb"], ["deadline_min"], ["deadline_action"], ["upsampler_local"]],
    notOffered: { config_toml: "the JSON tab (an inline worker config)" },
  },
  "token-create": { schema: "token-create", fields: [["name"], ["scope"], ["ttl_days"]] },
  "release-dispatch": { schema: "release-dispatch", fields: [["action"], ["target"], ["channel"], ["to"], ["notes"], ["templates"], ["allow_partial"], ["dry_run"]] },
  extend: { schema: "extend", fields: [["minutes"]] },
  scale: { schema: "scale", fields: [["pool"], ["count"]] },
  roll: { schema: "roll", fields: [["target"], ["pools"]] },
  "mint-key": { schema: "mint-key", fields: [["name"]] },
  "serverless-scale": { schema: "serverless-scale", fields: [["workers_min"], ["workers_max"]] },
  "serverless-cancel": { schema: "serverless-cancel", fields: [["job"], ["fv_job"], ["fv_api"], ["stop_fv_job"]] },
  "serverless-purge": { schema: "serverless-purge", fields: [["confirm"]], notOffered: { expected: "set from the queued count the dialog shows (a purge is refused when the queue grew since)" } },
  "job-cancel": { schema: "job-cancel", fields: [["job"], ["cluster"]] },
  "jobs-cancel-queued": { schema: "jobs-cancel-queued", fields: [["pool"], ["max"]] },
  "jobs-query": { schema: "jobs-query", fields: [], notOffered: { status: "the Jobs card's status buttons", pool: "the Jobs card lists every pool (each row says its pool); API and CLI filter", pod: "the pod pages link to their own view; API and CLI filter", limit: "the card shows the latest 100; API and CLI" } },
};

export interface Preflight {
  ok: boolean;
  errors: string[];
  warnings: string[];
}
/** The image preflight (POST /api/v1/preflight): the images resolve and have what the spec needs. */
export async function preflight(api: Api, body: { spec?: unknown; launch?: unknown; endpoint?: unknown }): Promise<Preflight> {
  try {
    const r = await api("/api/v1/preflight", { method: "POST", body });
    return { ok: !!r.ok, errors: r.errors || [], warnings: r.warnings || [] };
  } catch (e) {
    return { ok: false, errors: [(e as Error).message], warnings: [] };
  }
}
export const issuesOf = (r: any): Issue[] => (Array.isArray(r?.issues) ? r.issues : r?.error ? [{ path: [], message: r.error }] : []);

/** The server's own check of a schema (POST /api/v1/schemas/<name>/validate): refinements the JSON Schema cannot say. */
export const schemaRemote = (api: Api, name: string) => async (v: unknown) => {
  const r = await api(`/api/v1/schemas/${name}/validate`, { method: "POST", body: { doc: v } });
  return { issues: (r.issues || []) as Issue[] };
};
