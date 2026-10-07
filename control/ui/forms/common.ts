// Shared bits of the schema-bound forms: the schemas and live lists (cached
// per page load), the field manifests the coverage test reads, and the
// image preflight before a save or launch.
import type { Dynamic } from "../editor";
import type { Api } from "../fields";
import type { Issue, Path, Schema } from "../schema";

let schemas: Promise<Record<string, Schema>> | null = null;
const dyns = new Map<string, Promise<Dynamic>>();
export const loadSchemas = (api: Api) => (schemas ||= api("/api/schemas").then((r) => r.schemas));
export const loadDyn = (api: Api, cluster = "") => {
  if (!dyns.has(cluster)) dyns.set(cluster, api(`/api/schemas/dynamic${cluster ? `?cluster=${encodeURIComponent(cluster)}` : ""}`).catch(() => ({})));
  return dyns.get(cluster)!;
};
/** Forget the live lists (after a create: names, endpoints). */
export const resetDyn = () => dyns.clear();

/**
 * Every field each form offers, by schema: the UI coverage test
 * (test/unit/config-validation.test.ts) checks each path exists in its
 * schema and that every schema property is offered or listed in NOT_OFFERED
 * with the reason.
 */
export const FORM_FIELDS: Record<string, { schema: string; fields: Path[]; notOffered?: Record<string, string> }> = {
  "standalone-launch": {
    schema: "standalone-launch",
    fields: [["name"], ["preset"], ["variant"], ["config"], ["models"], ["fake_models"], ["channel"], ["sha"], ["image"], ["compute"], ["gpu_types"], ["cpu_flavors"], ["vcpu"], ["region"], ["volume"], ["container_disk_gb"], ["deadline_min"], ["idle_stop_min"], ["max_gpu_dph"], ["auth"], ["log_level"], ["env"]],
    notOffered: {
      config_toml: "inline worker configs come with the presets; a custom one is a cluster pool (the cluster page's TOML editor)",
      gpu_type: "API shorthand of gpu_types",
      dc: "API alias of region",
      min_balance: "the account defaults (Settings: policies); API only",
      balance_floor: "the account defaults; API only",
      min_start: "the account defaults; API only",
      start: "the form always starts the pod (Launch)",
      skip_image_check: "never from the UI: the preflight must pass",
    },
  },
  "serverless-endpoint": {
    schema: "serverless-endpoint",
    fields: [["name"], ["mode"], ["image", "channel"], ["image", "sha"], ["image", "ref"], ["variant"], ["compute"], ["config"], ["env"], ["gpu_types"], ["gpu_count"], ["cpu_flavors"], ["vcpu"], ["data_centers"], ["network_volume"], ["workers_min"], ["workers_max"], ["idle_timeout_s"], ["flashboot"], ["execution_timeout_s"], ["scaler_type"], ["scaler_value"], ["allowed_cuda"], ["container_disk_gb"], ["deadline_min"], ["deadline_action"]],
    notOffered: { config_toml: "the JSON tab (an inline worker config)" },
  },
  "token-create": { schema: "token-create", fields: [["name"], ["scope"], ["ttl_days"]] },
  "release-dispatch": { schema: "release-dispatch", fields: [["action"], ["target"], ["channel"], ["to"], ["notes"], ["templates"], ["allow_partial"], ["dry_run"]] },
  extend: { schema: "extend", fields: [["minutes"]] },
  scale: { schema: "scale", fields: [["pool"], ["count"]] },
  roll: { schema: "roll", fields: [["target"], ["pools"]] },
  "mint-key": { schema: "mint-key", fields: [["name"]] },
  "serverless-scale": { schema: "serverless-scale", fields: [["workers_min"], ["workers_max"]] },
};

export interface Preflight {
  ok: boolean;
  errors: string[];
  warnings: string[];
}
/** The image preflight (POST /api/preflight): the images resolve and have what the spec needs. */
export async function preflight(api: Api, body: { spec?: unknown; launch?: unknown; endpoint?: unknown }): Promise<Preflight> {
  try {
    const r = await api("/api/preflight", { method: "POST", body });
    return { ok: !!r.ok, errors: r.errors || [], warnings: r.warnings || [] };
  } catch (e) {
    return { ok: false, errors: [(e as Error).message], warnings: [] };
  }
}
export const issuesOf = (r: any): Issue[] => (Array.isArray(r?.issues) ? r.issues : r?.error ? [{ path: [], message: r.error }] : []);
