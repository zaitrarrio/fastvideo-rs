// Every instance type a launch provider lists, one normalized row each, for
// the launch screen's tables (GET /api/providers/:id/instance-types,
// docs/control/README.md "Instance types"): Runpod's GPU types per data centre
// (with stock), every Brev instance type of the org's listing (CPU-only ones
// too), every GMI container product. Rows outside what fv-control can launch
// are listed as well, `supported: false` with the reason. The normalizers are
// pure (unit-tested on recorded shapes); the fetchers use the per-isolate ttl
// cache, and `?refresh=1` reads the upstream again (at most once per
// REFRESH_MIN_MS per provider and isolate: a faster refresh is served from the
// cache and says so).
import { brev, brevOff, brevInstanceTypes, brevPrice, parseDuration, parseGiB } from "./brev";
import { AVAILABLE_REGIONS, REGIONS, type RegionId } from "./cluster/regions";
import { RUNPOD_DATA_CENTERS, RUNPOD_GPU_TYPES } from "./enums";
import type { Env } from "./env";
import { gmi, gmiPriceDivisor, gmiProducts } from "./gmi";
import { gpuClass, shortGpuName } from "./gpu-family";
import { providerImpl } from "./providers";
import { runpod } from "./runpod";
import { ttlCached, ttlCachedAt } from "./ttlcache";
import { HttpError } from "./util";

export interface InstanceTypeRow {
  /** Unique within the table (Runpod: "<gpu type>@<data centre>"; GMI: "<product>@<idc>"; Brev: the type). */
  id: string;
  /** The provider's own id of the type (Runpod GPU type id, Brev instance type, GMI product). */
  name: string;
  /** A short human summary: GPUs, VRAM, vCPU, RAM. */
  spec: string;
  /** The provider says it can be had now (null: not reported). */
  available: boolean | null;
  /** The stock level where the provider reports one (Runpod High / Medium / Low). */
  stock: string | null;
  /** $/hr of the whole instance (Runpod: one GPU, Secure Cloud on-demand). */
  usd_per_hr: number | null;
  usd_per_gpu_hr: number | null;
  /** The GPU model ("H100 SXM", "RTX PRO 6000"); null: no GPU. */
  gpu_model: string | null;
  /** The GPU's architecture (gpu-family.ts): Blackwell, Hopper, Ada Lovelace, Ampere, … */
  model_family: string | null;
  gpu_count: number;
  /** "<count> × <model>" ("-" without a GPU). */
  gpu: string;
  /** The GPU's segment (gpu-family.ts): datacenter, workstation, consumer. */
  gpu_family: string | null;
  gpu_vendor: string | null;
  /** GPU memory, GB: per GPU and in all. */
  vram_gb: number | null;
  vram_total_gb: number | null;
  /** System memory, GB (Runpod: the least it gives a one-GPU pod of the type). */
  ram_gb: number | null;
  vcpu: number | null;
  /** The storage as the provider reports it (container disk, network volume, gp3, ssd, nvme, …). */
  storage_type: string | null;
  /** fixed: the size comes with the type; elastic: chosen at create (min-max). */
  storage_kind: "fixed" | "elastic" | null;
  storage_gb: number | null;
  storage_min_gb: number | null;
  storage_max_gb: number | null;
  /** Data centre / location / IDC. */
  region: string | null;
  /** The cloud behind it (Brev: aws, gcp, shadeform/massedcompute…; GMI; Runpod: secure cloud) and the region's place. */
  cloud: string | null;
  /** Brev: the type can be stopped and kept (null: not a Brev notion / not reported). */
  stoppable: boolean | null;
  /** Brev: its estimated deploy time, seconds. */
  deploy_s: number | null;
  /** fv-control can launch it from the launch form; else `reason` says why not. */
  supported: boolean;
  reason: string | null;
  /** The launch form's fields a click on the row sets (null: not launchable). */
  launch: Record<string, unknown> | null;
}

export interface InstanceTypesView {
  provider: string;
  enabled: boolean;
  reason: string | null;
  /** When the upstream list was read (ms). */
  at: number | null;
  /** Served from the cache (not read for this request). */
  cached: boolean;
  /** A refresh was asked within REFRESH_MIN_MS of the last one: seconds until one is allowed (served from the cache). */
  refresh_limited: number | null;
  live_ok: boolean;
  /** Why the live list could not be read (the rows are then the configured ones). */
  error: string | null;
  rows: InstanceTypeRow[];
  counts: { total: number; supported: number; available: number };
}

const r2 = (v: number) => Math.round(v * 100) / 100;
const pos = (x: unknown): number | null => {
  const v = Number((x as any)?.amount ?? x);
  return x !== null && x !== undefined && x !== "" && Number.isFinite(v) && v > 0 ? v : null;
};
const gb = (v: number | null) => (v === null ? null : Math.round(v * 10) / 10);
/** {value, unit} (GB / GiB / MiB / TB …) -> GB; null when absent. */
const bytesGb = (b: any): number | null => {
  const v = Number(b?.value);
  if (!b || !Number.isFinite(v) || v <= 0) return null;
  const u = String(b.unit || "").toLowerCase();
  return gb(u.startsWith("t") ? v * 1024 : u.startsWith("m") ? v / 1024 : u.startsWith("k") ? v / 1048576 : v);
};
function specLine(r: { gpu_count: number; gpu_model: string | null; vram_gb: number | null; vcpu: number | null; ram_gb: number | null; ramMin?: boolean }): string {
  const bits = [r.gpu_count && r.gpu_model ? `${r.gpu_count}× ${r.gpu_model}${r.vram_gb ? ` ${r.vram_gb} GB` : ""}` : "CPU only"];
  if (r.vcpu) bits.push(`${r.ramMin ? "≥ " : ""}${r.vcpu} vCPU`);
  if (r.ram_gb) bits.push(`${r.ramMin ? "≥ " : ""}${r.ram_gb} GB RAM`);
  return bits.join(" · ");
}
function gpuCols(model: string | null, count: number, vram: number | null, ...names: (string | null | undefined)[]) {
  // The provider's full names first (Runpod's id says "Server Edition", its display name does not), then the model.
  const c = model ? gpuClass(...names, model) : null;
  return {
    gpu_model: model,
    model_family: c?.arch ?? null,
    gpu_count: model ? count : 0,
    gpu: model && count ? `${count} × ${model}` : "-",
    gpu_family: c?.segment ?? null,
    gpu_vendor: (c?.vendor ?? null) as string | null,
    vram_gb: model ? vram : null,
    vram_total_gb: model && vram !== null && count ? r2(vram * count) : null,
  };
}
const hardwareReasons = (g: { gpu_model: string | null; gpu_vendor: string | null }, archs?: unknown): string[] => {
  const out: string[] = [];
  if (!g.gpu_model) out.push("no GPU (a CPU-only type)");
  else if (g.gpu_vendor === "AMD") out.push("an AMD GPU: the serve images need NVIDIA CUDA");
  if (Array.isArray(archs) && archs.length && !archs.some((a) => /x86_64|amd64/i.test(String(a)))) out.push(`${archs.join("/")} only: the serve images are x86_64`);
  return out;
};
const finish = (row: Omit<InstanceTypeRow, "supported" | "reason" | "launch">, reasons: string[], launch: Record<string, unknown>): InstanceTypeRow => ({
  ...row,
  supported: !reasons.length,
  reason: reasons.length ? reasons.join("; ") : null,
  launch: reasons.length ? null : launch,
});

// ---------------------------------------------------------------- Runpod
export interface RunpodListing {
  gpuTypes: any[];
  /** null: the data-centre stock could not be read (one row per GPU type then). */
  dataCenters: any[] | null;
}
/** The data centres with an fv weights volume (CLAUDE.md: EU only now): where the launch form may place a Runpod GPU pod. */
export const runpodVolumeDcs = (): Map<string, RegionId> => new Map(AVAILABLE_REGIONS.map((r) => [REGIONS[r].dc, r] as [string, RegionId]));

/** Runpod: one row per (GPU type, data centre) Runpod lists stock for, and one per GPU type no data centre lists. */
export function normalizeRunpod(d: RunpodListing, volumeDcs = runpodVolumeDcs()): InstanceTypeRow[] {
  const known = new Set<string>(RUNPOD_GPU_TYPES);
  const knownDcs = new Set<string>(RUNPOD_DATA_CENTERS);
  const types = new Map<string, any>();
  for (const g of d.gpuTypes || []) if (g?.id && g.id !== "unknown") types.set(String(g.id), g);
  const rows: InstanceTypeRow[] = [];
  const seen = new Set<string>();
  const haveDcs = Array.isArray(d.dataCenters);
  const row = (g: any, dc: any | null, a: any | null): InstanceTypeRow => {
    const id = String(g.id);
    const display = String(g.displayName || shortGpuName(id));
    const vram = pos(g.memoryInGb);
    const price = pos(g.securePrice);
    const lp = g.lowestPrice || {};
    const cols = gpuCols(display, 1, vram, id);
    const ram = pos(lp.minMemory);
    const vcpu = pos(lp.minVcpu);
    const dcId: string | null = dc?.id ? String(dc.id) : null;
    const region = dcId ? volumeDcs.get(dcId) : undefined;
    const reasons = hardwareReasons({ gpu_model: cols.gpu_model, gpu_vendor: cols.gpu_vendor ?? (/^AMD/i.test(id) ? "AMD" : null) });
    if (!known.has(id)) reasons.push("not in fv-control's Runpod GPU type list (enums.ts RUNPOD_GPU_TYPES)");
    if (price === null) reasons.push("not offered in Secure Cloud (fv-control creates Secure Cloud pods)");
    if (!dcId) reasons.push(haveDcs ? "no data centre lists it now" : "the data centres' stock is unknown now (Refresh)");
    else if (dc.listed === false) reasons.push(`${dcId} is not a listed data centre`);
    else if (!knownDcs.has(dcId)) reasons.push(`${dcId} is not in fv-control's data centre list (enums.ts RUNPOD_DATA_CENTERS)`);
    else if (!region) reasons.push(`no fv weights volume in ${dcId}: the launch form places Runpod GPU pods where one is (${[...volumeDcs.keys()].join(", ") || "none now"}); weights_source r2 pods may use any data centre (JSON / CLI)`);
    // Without the data centres' answer: the type's own (any data centre) stock.
    const stock = a ? (a.stockStatus ? String(a.stockStatus) : null) : haveDcs ? null : lp.stockStatus ? String(lp.stockStatus) : null;
    return finish(
      {
        id: `${id}@${dcId ?? "-"}`,
        name: id,
        spec: specLine({ ...cols, vcpu, ram_gb: ram, ramMin: true }),
        available: a ? a.available === true && !!a.stockStatus : haveDcs ? false : stock ? true : null,
        stock,
        usd_per_hr: price,
        usd_per_gpu_hr: price,
        ...cols,
        ram_gb: ram,
        vcpu,
        storage_type: region ? `container disk + network volume (${REGIONS[region].volume})` : dc?.storageSupport ? "container disk (network volumes offered)" : "container disk",
        storage_kind: "elastic",
        storage_gb: 40,
        storage_min_gb: 5,
        storage_max_gb: 500,
        region: dcId,
        cloud: `Runpod Secure Cloud${dc?.location ? ` · ${dc.location}` : ""}`,
        stoppable: null,
        deploy_s: null,
      },
      reasons,
      { gpu_types: [id], region },
    );
  };
  for (const dc of d.dataCenters || []) {
    for (const a of dc?.gpuAvailability || []) {
      const g = types.get(String(a?.gpuTypeId || a?.id));
      if (!g) continue;
      seen.add(g.id);
      rows.push(row(g, dc, a));
    }
  }
  for (const g of types.values()) if (!seen.has(g.id)) rows.push(row(g, null, null));
  return rows;
}
const RUNPOD_TYPES_Q = "{ gpuTypes { id displayName manufacturer memoryInGb securePrice communityPrice lowestPrice(input: {gpuCount: 1}) { stockStatus minVcpu minMemory } } }";
const RUNPOD_DCS_Q = "{ dataCenters { id name location storageSupport listed gpuAvailability { available stockStatus gpuTypeId } } }";
async function runpodListing(env: Env): Promise<RunpodListing> {
  const [t, dcs] = await Promise.all([runpod.gql<any>(env, RUNPOD_TYPES_Q), runpod.gql<any>(env, RUNPOD_DCS_Q).catch(() => null)]);
  return { gpuTypes: t?.gpuTypes || [], dataCenters: Array.isArray(dcs?.dataCenters) ? dcs.dataCenters : null };
}

// ---------------------------------------------------------------- Brev
export interface BrevCtx {
  allowed: string[];
  off: string | null;
  /** The owner's BREV_PRICES override of a type's $/hr. */
  price?: (type: string) => number | null;
}
const STORAGE_SHOWN = 3;
/** Brev: every instance type of the org's listing (api/instances/alltypesavailable/{org}). */
export function normalizeBrev(raw: any[], ctx: BrevCtx): InstanceTypeRow[] {
  const allowed = new Set(ctx.allowed);
  return (raw || [])
    .filter((x) => x && x.type)
    .map((x) => {
      const type = String(x.type);
      const g = Array.isArray(x.supported_gpus) && x.supported_gpus.length ? x.supported_gpus[0] : null;
      const count = g ? Number(g.count) || 1 : 0;
      const model = g?.name ? String(g.name) : null;
      const vram = g ? bytesGb(g.memory_bytes) ?? parseGiB(g.memory) : null;
      const cols = gpuCols(model, count, vram, g?.manufacturer ? `${g.manufacturer} ${model}` : null);
      if (g?.manufacturer && !cols.gpu_vendor) cols.gpu_vendor = String(g.manufacturer).toUpperCase() === "NVIDIA" ? "NVIDIA" : String(g.manufacturer);
      const ram = bytesGb(x.memory_bytes) ?? parseGiB(x.memory);
      const vcpu = pos(x.vcpu);
      const price = ctx.price?.(type) ?? pos(x.base_price);
      const st: any[] = Array.isArray(x.supported_storage) ? x.supported_storage : [];
      const kinds = [...new Set(st.map((s) => String(s?.type || "")).filter(Boolean))];
      const fixed = st.map((s) => bytesGb(s?.size_bytes) ?? parseGiB(s?.size)).filter((v): v is number => v !== null);
      const mins = st.map((s) => bytesGb(s?.min_size_bytes) ?? parseGiB(s?.min_size)).filter((v): v is number => v !== null);
      const maxs = st.map((s) => bytesGb(s?.max_size_bytes) ?? parseGiB(s?.max_size)).filter((v): v is number => v !== null);
      const elastic = x.elastic_root_volume === true || (!fixed.length && (mins.length > 0 || maxs.length > 0));
      const reasons = ctx.off ? [`NVIDIA Brev is off: ${ctx.off}`] : [];
      reasons.push(...hardwareReasons(cols, x.supported_architectures));
      if (!allowed.has(type)) reasons.push("not in BREV_INSTANCE_TYPES (the account's allowed types)");
      return finish(
        {
          id: type,
          name: type,
          spec: specLine({ ...cols, vcpu, ram_gb: ram }),
          available: typeof x.is_available === "boolean" ? x.is_available : null,
          stock: null,
          usd_per_hr: price,
          usd_per_gpu_hr: price !== null && cols.gpu_count ? r2(price / cols.gpu_count) : null,
          ...cols,
          ram_gb: ram,
          vcpu,
          storage_type: kinds.length ? kinds.slice(0, STORAGE_SHOWN).join(", ") + (kinds.length > STORAGE_SHOWN ? ` +${kinds.length - STORAGE_SHOWN}` : "") : null,
          storage_kind: fixed.length ? "fixed" : elastic ? "elastic" : null,
          storage_gb: fixed.length ? Math.max(...fixed) : null,
          storage_min_gb: !fixed.length && mins.length ? Math.min(...mins) : null,
          storage_max_gb: !fixed.length && maxs.length ? Math.max(...maxs) : null,
          region: x.location ? String(x.location) : null,
          cloud: [x.provider, x.cloud].filter(Boolean).map(String).join("/") || null,
          stoppable: x.stoppable === true,
          deploy_s: parseDuration(x.estimated_deploy_time),
        },
        reasons,
        { provider_gpu: type },
      );
    });
}

// ---------------------------------------------------------------- GMI
export interface GmiCtx {
  allowed: string[];
  off: string | null;
  /** GMI's price ÷ this = $/hr (GMI_PRICE_DIVISOR). */
  divisor: number;
}
/** GPU memory per GPU model (GB) where GMI's product spec does not say it (its spec names the model only). */
export const KNOWN_VRAM_GB: Record<string, number> = { H100: 80, H200: 141, B200: 180, B300: 288, GB200: 186, A100: 80, L40S: 48, L40: 48, L4: 24, A10: 24, A10G: 24, T4: 16, V100: 16, A6000: 48, "RTX 4090": 24, "RTX 5090": 32, "RTX PRO 6000": 96 };
const qty = (q: any): number | null => pos(q?.qty ?? q);
const giQty = (q: any): number | null => {
  const v = qty(q);
  if (v === null) return null;
  const u = String(q?.unit || "Gi");
  return gb(/^T/i.test(u) ? v * 1024 : /^M/i.test(u) ? v / 1024 : v);
};
/** GMI: every container product (GET /v1/containers/products, all IDCs). */
export function normalizeGmi(raw: any[], ctx: GmiCtx): InstanceTypeRow[] {
  const allowed = new Set(ctx.allowed);
  return (raw || [])
    .filter((p) => p && p.name)
    .map((p) => {
      const name = String(p.name);
      const idc = p.idc ? String(p.idc) : null;
      const res = p.spec?.resource || {};
      const basic: any[] = Array.isArray(p.spec?.basic) ? p.spec.basic : [];
      const gpuLine = basic.find((b) => /^GPU$/i.test(String(b?.name || "")));
      const rawModel = p.gpuModel ? String(p.gpuModel) : gpuLine?.model ? String(gpuLine.model) : null;
      const cpuOnly = !rawModel || /cpu[- ]?only/i.test(rawModel);
      const model = cpuOnly ? null : shortGpuName(rawModel!);
      const count = cpuOnly ? 0 : qty(res.gpu) ?? qty(gpuLine) ?? Number(/\.x(\d{1,2})(?:\.|$)/.exec(name)?.[1] || 1);
      const vram = model ? giQty(gpuLine?.memory) ?? KNOWN_VRAM_GB[model.toUpperCase()] ?? KNOWN_VRAM_GB[model] ?? null : null;
      const cols = gpuCols(model, count, vram, name);
      const ram = giQty(res.memory) ?? giQty(basic.find((b) => /^Memory$/i.test(String(b?.name || ""))));
      const vcpu = qty(res.cpu) ?? qty(basic.find((b) => /^CPU$/i.test(String(b?.name || ""))));
      const disks = basic.filter((b) => /storage/i.test(String(b?.name || "")));
      const price = typeof p.price === "number" && p.price > 0 ? Math.round((p.price / ctx.divisor) * 10000) / 10000 : null;
      const data = disks.find((b) => /data/i.test(String(b.name)));
      const reasons = ctx.off ? [`GMI Cloud is off: ${ctx.off}`] : [];
      reasons.push(...hardwareReasons(cols));
      if (!allowed.has(name)) reasons.push("not in GMI_PRODUCTS (the account's allowed products)");
      return finish(
        {
          id: `${name}@${idc ?? "-"}`,
          name,
          spec: specLine({ ...cols, vcpu, ram_gb: ram }),
          available: p.valid === false ? false : p.valid === true ? true : null,
          stock: null,
          usd_per_hr: price,
          usd_per_gpu_hr: price !== null && cols.gpu_count ? r2(price / cols.gpu_count) : null,
          ...cols,
          ram_gb: ram,
          vcpu,
          storage_type: disks.length ? disks.map((b) => `${String(b.model || "disk").trim()} ${/data/i.test(String(b.name)) ? "data" : /os/i.test(String(b.name)) ? "OS" : "disk"}${/ephemeral/i.test(String(b.name)) ? " (ephemeral)" : ""}`).join(" + ") : giQty(res.localStorage) ? "local" : null,
          storage_kind: disks.length || res.localStorage ? "fixed" : null,
          storage_gb: data ? giQty({ qty: Number(data.qty), unit: data.unit }) : giQty(res.localStorage),
          storage_min_gb: null,
          storage_max_gb: null,
          region: idc,
          cloud: "GMI Cloud",
          stoppable: null,
          deploy_s: null,
        },
        reasons,
        { provider_gpu: name, ...(idc ? { provider_region: idc } : {}) },
      );
    });
}

// ---------------------------------------------------------------- the endpoint
export const INSTANCE_TYPE_PROVIDERS = ["runpod", "gmi", "brev"] as const;
/** A refresh (cache bypass) per provider and isolate at most this often. */
export const REFRESH_MIN_MS = 30_000;
const lastRefresh = new Map<string, number>();
/** Tests: forget the refresh rate limit. */
export const resetRefreshLimit = () => lastRefresh.clear();
const TTL = { runpod: 300_000, gmi: 60_000 };

const sortRows = (rows: InstanceTypeRow[]) =>
  rows.sort((a, b) => Number(b.supported) - Number(a.supported) || Number(b.available === true) - Number(a.available === true) || (a.usd_per_hr ?? 1e9) - (b.usd_per_hr ?? 1e9) || a.id.localeCompare(b.id));

/** GET /api/providers/:id/instance-types: every type the provider lists, normalized; `refresh` reads the upstream again (rate-limited). */
export async function instanceTypes(env: Env, p: string, o: { refresh?: boolean } = {}): Promise<InstanceTypesView> {
  if (!(INSTANCE_TYPE_PROVIDERS as readonly string[]).includes(p)) throw new HttpError(404, `no launch provider ${p} (${INSTANCE_TYPE_PROVIDERS.join(", ")})`);
  const t0 = Date.now();
  let refresh = !!o.refresh;
  let limited: number | null = null;
  if (refresh) {
    const last = lastRefresh.get(p) ?? 0;
    if (t0 - last < REFRESH_MIN_MS) {
      refresh = false;
      limited = Math.ceil((REFRESH_MIN_MS - (t0 - last)) / 1000);
    } else lastRefresh.set(p, t0);
  }
  const off = p === "runpod" ? null : providerImpl(p as "gmi" | "brev").off(env);
  const view = (rows: InstanceTypeRow[], at: number | null, extra: Partial<InstanceTypesView> = {}): InstanceTypesView => ({
    provider: p,
    enabled: !off,
    reason: off ? `${providerImpl(p as "gmi" | "brev").title} is off: ${off} (docs/serve/deploy-gmi-brev.md §8)` : null,
    at,
    cached: at !== null && at < t0,
    refresh_limited: limited,
    live_ok: true,
    error: null,
    rows: sortRows(rows),
    counts: { total: rows.length, supported: rows.filter((r) => r.supported).length, available: rows.filter((r) => r.available === true).length },
    ...extra,
  });
  // A provider whose credentials are unset cannot be listed: its configured types, each not launchable (why: off).
  const configured = (ids: string[], why: string) =>
    ids.map((id) => finish({ id, name: id, spec: "-", available: null, stock: null, usd_per_hr: null, usd_per_gpu_hr: null, ...gpuCols(null, 0, null), ram_gb: null, vcpu: null, storage_type: null, storage_kind: null, storage_gb: null, storage_min_gb: null, storage_max_gb: null, region: null, cloud: null, stoppable: null, deploy_s: null }, [why], {}));
  const load = async (fresh: boolean): Promise<InstanceTypesView> => {
    if (p === "runpod") {
      const key = "runpod:instance-types";
      const d = await ttlCached(key, TTL.runpod, () => runpodListing(env), { refresh: fresh });
      return view(normalizeRunpod(d), ttlCachedAt(key), d.dataCenters ? {} : { error: "Runpod's data-centre stock could not be read: one row per GPU type" });
    }
    if (p === "brev") {
      if (off) return view(configured(brevInstanceTypes(env), `NVIDIA Brev is off: ${off}`), null, { live_ok: false });
      const { at, rows } = await brev.rawTypes(env, { refresh: fresh });
      return view(normalizeBrev(rows, { allowed: brevInstanceTypes(env), off, price: (t) => brevPrice(env, t) }), at);
    }
    if (off) return view(configured(gmiProducts(env), `GMI Cloud is off: ${off}`), null, { live_ok: false });
    const key = "gmi:instance-types";
    const raw = await ttlCached(key, TTL.gmi, () => gmi.rawProducts(env), { refresh: fresh });
    return view(normalizeGmi(raw, { allowed: gmiProducts(env), off, divisor: gmiPriceDivisor(env) }), ttlCachedAt(key));
  };
  try {
    return await load(refresh);
  } catch (e) {
    const msg = (e as Error).message;
    // A failed refresh: the cached list (when one is still fresh), saying the refresh failed.
    if (refresh) {
      const cached = await load(false).catch(() => null);
      if (cached) return { ...cached, error: `the refresh failed: ${msg} (showing the cached list)` };
    }
    const ids = p === "runpod" ? [...RUNPOD_GPU_TYPES] : p === "brev" ? brevInstanceTypes(env) : gmiProducts(env);
    return view(configured(ids, "the live list is unavailable: launchable once it loads (Refresh)"), null, { live_ok: false, error: msg });
  }
}
