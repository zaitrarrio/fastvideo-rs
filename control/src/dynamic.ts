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
import { ENV_VALUE_TYPES, FAKE_MODELS, OTHER_PROVIDERS, RUNPOD_DATA_CENTERS, VARIANTS } from "./enums";
import { providerImpl } from "./providers";
import { listTags } from "./ghcr";
import { knownVolumes, SLS_RESERVED } from "./serverless/spec";
import { SLS_PRESETS } from "./serverless/presets";

export interface GpuType {
  id: string;
  display: string;
  memory_gb: number | null;
  secure_price: number | null;
  community_price: number | null;
  stock: string | null;
}
let gpuCache: { at: number; list: GpuType[] } | null = null;
/** Runpod GPU types; a price of 0 means "not offered" in that cloud (null here); cheapest secure first. */
export async function gpuTypes(env: Env): Promise<GpuType[]> {
  if (gpuCache && Date.now() - gpuCache.at < 300_000) return gpuCache.list;
  const d = await runpod.gql<any>(env, "{ gpuTypes { id displayName memoryInGb securePrice communityPrice lowestPrice(input: {gpuCount: 1}) { stockStatus } } }");
  const list: GpuType[] = (d?.gpuTypes || [])
    .filter((g: any) => g.id && g.id !== "unknown")
    .map((g: any) => ({ id: g.id, display: g.displayName || g.id, memory_gb: g.memoryInGb ?? null, secure_price: g.securePrice > 0 ? g.securePrice : null, community_price: g.communityPrice > 0 ? g.communityPrice : null, stock: g.lowestPrice?.stockStatus ?? null }))
    .sort((a: GpuType, b: GpuType) => (a.secure_price ?? 99) - (b.secure_price ?? 99));
  gpuCache = { at: Date.now(), list };
  return list;
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

export async function dynamicEnums(env: Env, clusterId?: string) {
  const [gpus, heads, clusters, keys, tagShas, endpoints] = await Promise.all([
    gpuTypes(env).catch(() => [] as GpuType[]),
    releaseHeads(env).catch(() => ({ available: false, heads: [] as any[], history: [] })),
    listClusters(env),
    env.DB.prepare("SELECT key, scope, COUNT(*) AS n FROM env_vars GROUP BY key, scope ORDER BY key").all<{ key: string; scope: string; n: number }>(),
    imageShas(env).catch(() => [] as string[]),
    env.DB.prepare("SELECT name FROM serverless_endpoints WHERE deleted_at IS NULL ORDER BY name").all<{ name: string }>().catch(() => ({ results: [] as { name: string }[] })),
  ]);
  const vols = knownVolumes();
  const channels = [...new Set(["stable", "latest", ...heads.heads.map((h: any) => h.channel)])];
  const pools = clusterId ? (await getCluster(env, clusterId).catch(() => null))?.spec.pools.map((p) => p.id) ?? [] : [...new Set(clusters.flatMap((c) => c.spec.pools.map((p) => p.id)))];
  return {
    gpu_types: gpus,
    regions: Object.entries(REGIONS)
      .filter(([id]) => regionAvailable(id))
      .map(([id, r]) => ({ id, dc: r.dc, volume: r.volume, gpus: r.gpus })),
    cpu_flavors: Object.entries(CPU_DPH_PER_VCPU).map(([id, per]) => ({ id, dph_per_vcpu: per })),
    channels: channels.map((ch) => {
      const h = heads.heads.find((x: any) => x.channel === ch);
      return { id: ch, sha: h?.git_sha?.slice(0, 7) ?? null, promoted_at: h?.promoted_at ?? null, digests: h?.digests ?? {} };
    }),
    shas: [...new Set([...heads.history.map((h: any) => String(h.git_sha).slice(0, 7)), ...tagShas.map((x) => x.slice(0, 7))])].slice(0, 60),
    releases: heads.history.slice(0, 30).map((h: any) => ({ id: String(h.id), detail: `${h.channel} → ${String(h.git_sha).slice(0, 7)} (${h.action})` })),
    volumes: vols.map((v) => ({ id: v.id, dc: v.dc, region: v.region, detail: `${v.region} weights volume in ${v.dc}` })),
    data_centers: RUNPOD_DATA_CENTERS.map((dc) => ({ id: dc, detail: vols.some((v) => v.dc === dc) ? "weights volume here" : "" })),
    dc_prefixes: [...new Set(RUNPOD_DATA_CENTERS.flatMap((dc) => [dc.split("-")[0]!, dc.split("-").slice(0, 2).join("-"), dc]))].sort(),
    config_paths: [...new Set([...POOL_PRESETS, ...STANDARD_POOLS.map((pool) => ({ pool }))].map((p) => p.pool.config).filter((x): x is string => !!x).concat(["/etc/fv/runpod-fake.toml"]))].sort(),
    endpoints: (endpoints.results || []).map((e) => e.name),
    variants: VARIANTS.map((v) => ({ id: v, detail: variantDetail(v) })),
    pool_presets: POOL_PRESETS.map((p) => ({ id: p.id, detail: `${p.title}${p.licence ? ` · ${p.licence}` : ""}` })),
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
    providers: [{ id: "runpod", detail: "Runpod (default): the EU weights volume" }, ...OTHER_PROVIDERS.map((p) => ({ id: p, detail: providerImpl(p).off(env) ? `off: ${providerImpl(p).off(env)}` : providerImpl(p).title }))],
    provider_gpus: OTHER_PROVIDERS.flatMap((p) => providerImpl(p).gpus(env).map((g) => ({ id: g, detail: providerImpl(p).title, provider: p }))),
    provider_regions: env.GMI_DEFAULT_IDC ? [{ id: env.GMI_DEFAULT_IDC, detail: "GMI default IDC" }] : [],
  };
}
