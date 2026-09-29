// Live values for the editors' autocomplete and dropdowns
// (GET /api/schemas/dynamic): GPU types with price and stock, regions and
// volumes, CPU flavors, release channels and their digests, image
// variants, pools, env keys in use and the controller's reserved keys.
import { RESERVED_KEYS } from "./cluster/payloads";
import { REGIONS, STANDARD_POOLS } from "./cluster/spec";
import { getCluster, listClusters } from "./cluster/store";
import type { Env } from "./env";
import { releaseHeads } from "./releases";
import { CPU_DPH_PER_VCPU, runpod } from "./runpod";

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

export const VARIANTS = ["h3-turbo", "h3-max", "ltx", "wan", "wan5b", "sfwan", "gateway"];
export const FAKE_MODELS = ["fake-h3-max", "fake-h3-turbo", "fake-sol-h3", "fake-ltx-pro", "fake-ltx-turbo", "fake-wan", "fake-sfwan"];

export async function dynamicEnums(env: Env, clusterId?: string) {
  const [gpus, heads, clusters, keys] = await Promise.all([
    gpuTypes(env).catch(() => [] as GpuType[]),
    releaseHeads(env).catch(() => ({ available: false, heads: [] as any[], history: [] })),
    listClusters(env),
    env.DB.prepare("SELECT key, scope, COUNT(*) AS n FROM env_vars GROUP BY key, scope ORDER BY key").all<{ key: string; scope: string; n: number }>(),
  ]);
  const channels = [...new Set(["stable", "latest", ...heads.heads.map((h: any) => h.channel)])];
  const pools = clusterId ? (await getCluster(env, clusterId).catch(() => null))?.spec.pools.map((p) => p.id) ?? [] : [...new Set(clusters.flatMap((c) => c.spec.pools.map((p) => p.id)))];
  return {
    gpu_types: gpus,
    regions: Object.entries(REGIONS).map(([id, r]) => ({ id, dc: r.dc, volume: r.volume, gpus: r.gpus })),
    cpu_flavors: Object.entries(CPU_DPH_PER_VCPU).map(([id, per]) => ({ id, dph_per_vcpu: per })),
    channels: channels.map((ch) => {
      const h = heads.heads.find((x: any) => x.channel === ch);
      return { id: ch, sha: h?.git_sha?.slice(0, 7) ?? null, promoted_at: h?.promoted_at ?? null, digests: h?.digests ?? {} };
    }),
    shas: [...new Set(heads.history.map((h: any) => String(h.git_sha).slice(0, 7)))].slice(0, 20),
    variants: VARIANTS,
    fake_models: FAKE_MODELS,
    models: STANDARD_POOLS.flatMap((p) => p.models || []),
    pools,
    clusters: clusters.map((c) => ({ id: c.id, name: c.name })),
    env_keys: [...new Set((keys.results || []).map((k) => k.key))],
    reserved_env_keys: [...RESERVED_KEYS, "FV_POOL_<ID>_URLS"],
  };
}
