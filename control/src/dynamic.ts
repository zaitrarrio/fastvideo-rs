// Live values for the editors' autocomplete and dropdowns
// (GET /api/schemas/dynamic): GPU types with price and stock, regions and
// volumes, CPU flavors, release channels and their digests, image
// variants and the pool presets that use them, pools, model ids, recipes and
// fal apps (cluster/catalog.json), env keys (the engine's, with what they do,
// and those in use) and the controller's reserved keys.
import CATALOG from "./cluster/catalog.json";
import { RESERVED_KEYS } from "./cluster/payloads";
import { POOL_PRESETS, REGIONS, regionAvailable, STANDARD_POOLS } from "./cluster/spec";
import { getCluster, listClusters } from "./cluster/store";
import type { Env } from "./env";
import { releaseHeads } from "./releases";
import { CPU_DPH_PER_VCPU, runpod } from "./runpod";
import { ENV_VALUE_TYPES, FAKE_MODELS, OTHER_PROVIDERS, RUNPOD_DATA_CENTERS, VARIANTS, type OtherProviderId } from "./enums";
import { brevGpuOptions, providerImpl, providerView } from "./providers";
import { brevInstanceTypes } from "./brev";
import { presetWeightsDefault } from "./standalone";
import { parkedView } from "./brev-park";
import { listTags } from "./ghcr";
import { knownVolumes, SLS_RESERVED } from "./serverless/spec";
import { SLS_PRESETS } from "./serverless/presets";
import { ttlCached } from "./ttlcache";
import { HttpError } from "./util";

export interface GpuType {
  id: string;
  display: string;
  memory_gb: number | null;
  secure_price: number | null;
  community_price: number | null;
  stock: string | null;
}
/** Runpod GPU types; a price of 0 means "not offered" in that cloud (null here); cheapest secure first. 5-minute cache, one call in flight. */
export function gpuTypes(env: Env): Promise<GpuType[]> {
  return ttlCached("runpod:gpu-types", 300_000, async () => {
    const d = await runpod.gql<any>(env, "{ gpuTypes { id displayName memoryInGb securePrice communityPrice lowestPrice(input: {gpuCount: 1}) { stockStatus } } }");
    return (d?.gpuTypes || [])
      .filter((g: any) => g.id && g.id !== "unknown")
      .map((g: any) => ({ id: g.id, display: g.displayName || g.id, memory_gb: g.memoryInGb ?? null, secure_price: g.securePrice > 0 ? g.securePrice : null, community_price: g.communityPrice > 0 ? g.communityPrice : null, stock: g.lowestPrice?.stockStatus ?? null }))
      .sort((a: GpuType, b: GpuType) => (a.secure_price ?? 99) - (b.secure_price ?? 99));
  });
}

export { VARIANTS, FAKE_MODELS } from "./enums";
const variantDetail = (v: string) => {
  const ps = POOL_PRESETS.filter((p) => p.pool.variant === v).map((p) => p.id);
  return v === "cpu" ? "CPU: the fake engine" : ps.length ? `presets: ${ps.join(", ")}` : "";
};
/** Env keys with what they do: the engine's (catalog.json) and the controller-facing serve ones. */
export function envKeyOptions(inUse: { key: string; scope: string; n: number }[]) {
  const use = new Map<string, string[]>();
  for (const k of inUse) (use.get(k.key) || use.set(k.key, []).get(k.key)!).push(`${k.scope}×${k.n}`);
  const out = CATALOG.engine_env.map((e) => ({ id: e.id, detail: use.has(e.id) ? `${e.detail} (in use: ${use.get(e.id)!.join(", ")})` : e.detail }));
  for (const [k, v] of use) if (!out.some((o) => o.id === k)) out.push({ id: k, detail: `in use: ${v.join(", ")}` });
  return out;
}

let tagCache: { at: number; shas: string[] } | null = null;
/** Commits with serve images in GHCR (tags <variant>-sha-<sha7>), newest first as the registry lists them; 5-minute cache, 4 s budget. */
async function imageShas(env: Env): Promise<string[]> {
  if (tagCache && Date.now() - tagCache.at < 300_000) return tagCache.shas;
  const tags = await Promise.race([listTags(env), new Promise<string[]>((_, rej) => setTimeout(() => rej(new Error("timeout")), 4000))]);
  const shas = [...new Set(tags.map((t) => /(?:^|-)sha-([0-9a-f]{7,40})$/.exec(t)?.[1]).filter((x): x is string => !!x))].reverse().slice(0, 50);
  tagCache = { at: Date.now(), shas };
  return shas;
}

/**
 * The editors' lists. `live: false` (GET /api/schemas/dynamic?live=0, the launch forms on open) makes no upstream
 * call: no Runpod GPU types, no GHCR tags, no provider listing (only what the config says); the forms then ask
 * providerDetails for the provider picked. The default (live) is the full view (the JSON editor, the CLI).
 */
export async function dynamicEnums(env: Env, clusterId?: string, opts: { live?: boolean } = {}) {
  const live = opts.live !== false;
  const [gpus, heads, clusters, keys, tagShas, endpoints] = await Promise.all([
    live ? gpuTypes(env).catch(() => [] as GpuType[]) : ([] as GpuType[]),
    releaseHeads(env).catch(() => ({ available: false, heads: [] as any[], history: [] })),
    listClusters(env),
    env.DB.prepare("SELECT key, scope, COUNT(*) AS n FROM env_vars GROUP BY key, scope ORDER BY key").all<{ key: string; scope: string; n: number }>(),
    live ? imageShas(env).catch(() => [] as string[]) : ([] as string[]),
    env.DB.prepare("SELECT name FROM serverless_endpoints WHERE deleted_at IS NULL ORDER BY name").all<{ name: string }>().catch(() => ({ results: [] as { name: string }[] })),
  ]);
  const vols = knownVolumes();
  const channels = [...new Set(["stable", "latest", ...heads.heads.map((h: any) => h.channel)])];
  const pools = clusterId ? (await getCluster(env, clusterId).catch(() => null))?.spec.pools.map((p) => p.id) ?? [] : [...new Set(clusters.flatMap((c) => c.spec.pools.map((p) => p.id)))];
  return {
    live,
    gpu_types: gpus,
    regions: Object.entries(REGIONS)
      .filter(([id]) => regionAvailable(id))
      .map(([id, r]) => ({ id, dc: r.dc, volume: r.volume, gpus: r.gpus })),
    cpu_flavors: Object.entries(CPU_DPH_PER_VCPU).map(([id, per]) => ({ id, dph_per_vcpu: per })),
    channels: channels.map((ch) => {
      const h = heads.heads.find((x: any) => x.channel === ch);
      return { id: ch, sha: h?.git_sha?.slice(0, 7) ?? null, promoted_at: h?.promoted_at ?? null, digests: h?.digests ?? {} };
    }),
    shas: shaList(heads.history, tagShas),
    releases: heads.history.slice(0, 30).map((h: any) => ({ id: String(h.id), detail: `${h.channel} → ${String(h.git_sha).slice(0, 7)} (${h.action})` })),
    volumes: vols.map((v) => ({ id: v.id, dc: v.dc, region: v.region, detail: `${v.region} weights volume in ${v.dc}` })),
    data_centers: RUNPOD_DATA_CENTERS.map((dc) => ({ id: dc, detail: vols.some((v) => v.dc === dc) ? "weights volume here" : "" })),
    dc_prefixes: [...new Set(RUNPOD_DATA_CENTERS.flatMap((dc) => [dc.split("-")[0]!, dc.split("-").slice(0, 2).join("-"), dc]))].sort(),
    config_paths: [...new Set([...POOL_PRESETS, ...STANDARD_POOLS.map((pool) => ({ pool }))].map((p) => p.pool.config).filter((x): x is string => !!x).concat(["/etc/fv/runpod-fake.toml"]))].sort(),
    endpoints: (endpoints.results || []).map((e) => e.name),
    variants: VARIANTS.map((v) => ({ id: v, detail: variantDetail(v) })),
    // weights_source: the preset's default on a Runpod GPU pod (the launch form's Weights default; standalone.ts).
    pool_presets: POOL_PRESETS.map((p) => ({ id: p.id, detail: `${p.title}${p.licence ? ` · ${p.licence}` : ""}`, weights_source: presetWeightsDefault(p.id) })),
    sls_presets: SLS_PRESETS.map((p) => ({ id: p.id, detail: `${p.title} · ${p.variant}${p.config_toml ? " + inline config" : ""}${p.min_vram_gb ? ` · ≥ ${p.min_vram_gb} GB GPU` : ""}${p.licence ? ` · ${p.licence}` : ""}` })),
    fake_models: FAKE_MODELS,
    models: POOL_PRESETS.flatMap((p) => p.pool.models || []),
    catalog_models: CATALOG.models.map((m) => ({ id: m.id, family: m.family, recipe: m.recipe })),
    model_ids: CATALOG.models.map((m) => ({ id: m.id, detail: `${m.family} · ${m.recipe} · ${m.detail}` })),
    families: CATALOG.families,
    recipes: CATALOG.recipes.map((r) => ({ id: r.id, detail: `${r.family}${r.serve ? "" : " · NOT servable"} · ${r.detail}` })),
    fal_apps: CATALOG.fal_apps,
    pools,
    clusters: clusters.map((c) => ({ id: c.id, name: c.name })),
    env_keys: envKeyOptions(keys.results || []),
    reserved_env_keys: [...RESERVED_KEYS],
    sls_reserved_env_keys: [...SLS_RESERVED],
    env_types: ENV_VALUE_TYPES,
    // Launch providers (docs/serve/deploy-gmi-brev.md): Runpod, and GMI / Brev when their secrets are set (detail says why not).
    providers: [{ id: "runpod", detail: "Runpod (default): the EU weights volume, or R2 / Hub / none in any data centre" }, ...OTHER_PROVIDERS.map((p) => ({ id: p, detail: providerImpl(p).off(env) ? `off: ${providerImpl(p).off(env)}` : providerImpl(p).title }))],
    // Brev: live prices, stoppable types first (keep-on-stop), parked instances (docs/serve/deploy-gmi-brev.md §3.4).
    provider_gpus: [...configuredGpus(env, "gmi"), ...(live && !providerImpl("brev").off(env) ? await brevGpuOptions(env).catch(() => []) : configuredGpus(env, "brev"))],
    provider_regions: env.GMI_DEFAULT_IDC ? [{ id: env.GMI_DEFAULT_IDC, detail: "GMI default IDC" }] : [],
  };
}

const shaList = (history: any[], tagShas: string[]) => [...new Set([...history.map((h: any) => String(h.git_sha).slice(0, 7)), ...tagShas.map((x) => x.slice(0, 7))])].slice(0, 60);
/** The commits with images (GET /api/schemas/dynamic?images=1): releases and GHCR's sha tags, asked when a form's image source is a commit. */
export async function imageCommits(env: Env) {
  const [heads, tagShas] = await Promise.all([releaseHeads(env).catch(() => ({ history: [] as any[] })), imageShas(env).catch(() => [] as string[])]);
  return { shas: shaList(heads.history, tagShas) };
}

/** A provider's allowed GPU products / instance types as the config lists them (no upstream call). */
const configuredGpus = (env: Env, p: OtherProviderId) => (p === "brev" ? brevInstanceTypes(env) : providerImpl(p).gpus(env)).map((g) => ({ id: g, detail: providerImpl(p).title, provider: p }));

export const LAUNCH_PROVIDERS = ["runpod", ...OTHER_PROVIDERS] as const;

/**
 * One launch provider's live details (GET /api/schemas/dynamic?provider=<p>), asked by the launch forms once that
 * provider is picked: Runpod's GPU types with $/hr and stock; GMI's products with price and availability in the
 * default IDC; Brev's instance types (stoppable first, live price, parked count) and its parked / held VMs; for GMI
 * and Brev the month's spend against the budget. Only that provider's API is called (an off one: none). The upstream
 * lists are cached (Runpod 5 min, Brev 1 h, GMI 60 s); the D1 parts (spend, parked) are read fresh.
 */
export async function providerDetails(env: Env, p: string) {
  if (!(LAUNCH_PROVIDERS as readonly string[]).includes(p)) throw new HttpError(404, `no launch provider ${p} (${LAUNCH_PROVIDERS.join(", ")})`);
  if (p === "runpod") {
    const gpus = await gpuTypes(env).catch(() => null);
    return { provider: "runpod", enabled: true, reason: null, live_ok: !!gpus?.length, gpu_types: gpus || [] };
  }
  const id = p as OtherProviderId;
  const impl = providerImpl(id);
  const off = impl.off(env);
  const view = await providerView(env, id);
  const base = { provider: id, enabled: !off, reason: view.reason, budget: { budget_usd: view.budget_usd, month_usd: view.month_usd, running_dph: view.running_dph, parked_dph: view.parked_dph } };
  if (off) return { ...base, live_ok: false, provider_gpus: configuredGpus(env, id), provider_regions: [], ...(id === "brev" ? { parked: [] } : {}) };
  if (id === "brev") {
    const [gpus, pk] = await Promise.all([brevGpuOptions(env).then((x) => ({ ok: true, x }), () => ({ ok: false, x: configuredGpus(env, "brev") })), parkedView(env)]);
    return { ...base, live_ok: gpus.ok, provider_gpus: gpus.x, provider_regions: [], parked: pk.parked, parked_usd_per_day: pk.storage_usd_per_day, park_policy: pk.policy.line };
  }
  // GMI: each allowed product's price and availability in the default IDC (one products call, cached 60 s).
  const offers = await ttlCached(`gmi:offers:${env.GMI_API || ""}:${env.GMI_DEFAULT_IDC || ""}:${impl.gpus(env).join(",")}`, 60_000, () => Promise.all(impl.gpus(env).map((g) => impl.offer(env, g)))).catch(() => null);
  const provider_gpus = impl.gpus(env).map((g) => {
    const o = offers?.find((x) => x.gpu === g);
    const bits = [impl.title, o?.usd_per_hr != null ? `$${o.usd_per_hr.toFixed(2)}/hr` : "price unknown", ...(o?.in_stock === false ? [`not available${o.region ? ` in ${o.region}` : ""} now`] : o?.in_stock ? ["available"] : [])];
    return { id: g, detail: bits.join(" · "), provider: id, usd_per_hr: o?.usd_per_hr ?? null, in_stock: o?.in_stock ?? null };
  });
  return { ...base, live_ok: !!offers, provider_gpus, provider_regions: env.GMI_DEFAULT_IDC ? [{ id: env.GMI_DEFAULT_IDC, detail: "GMI default IDC" }] : [] };
}
