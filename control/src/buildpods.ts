// Build pods managed by fv-control (docs/dev/build-pods-fv-control.md): CPU
// pods running scripts/dev/build-pod-server.py, one or more, in any Runpod
// datacenter. Only this module creates, starts, stops, replaces and deletes
// them; scripts/dev/build-pod.sh is a client of the pods' HTTP API and asks
// fv-control for a pod (`up`). Each pod's token is generated here, sealed in
// D1 and handed only to admin callers; the pod gets its sha256.
//
// What a pod runs comes from the repository at `server_ref` (main by
// default): the server source and the base image pin of build-pod.sh. A
// branch cannot put its server on a shared pod; a pod whose server or image
// is not the current one is replaced only once it is stopped (never while it
// runs jobs).
import { buildPodHealth, buildPodVerdict, stopBuildPod, type BuildPodHealth } from "./buildpod";
import { BUILD_POD_START_CMD } from "./buildpod-start";
import { b64, randomToken, seal, sha256Hex, unseal } from "./crypto";
import { defaults, type Env } from "./env";
import { deleteRunner, listRunners, queuedJobs, repoFile, runnerPatSet, runnerRegistrationToken, type GhRunner } from "./github";
import { runpod, type RunpodPod } from "./runpod";
import { audit, fetchWithTimeout, getSetting, HttpError, now, putSetting, scrub, utcDay } from "./util";
import type { AlertIn } from "./alerts";

// ---------------------------------------------------------------- policy

export interface BuildPodsPolicy {
  enabled: boolean;
  max_pods: number; // running (or being created) at once
  max_dph_per_pod: number;
  daily_usd_max: number; // build pods' spend today above this: `up` refuses, alert
  balance_margin: number; // `up` needs balance >= account floor + this
  flavors: string[]; // Runpod CPU flavors, preferred first
  vcpus: number[]; // sizes, preferred first
  disk_gb: number;
  disk_gb_fallback: number; // for sizes below the first
  regions: string[]; // preferred datacenter prefixes (EU, EUR-IS, US-CA-2 …)
  regions_only: boolean; // only the preferred ones
  volumes: Record<string, string>; // datacenter -> network volume id (optional local cache)
  idle_min: number;
  max_h: number;
  max_grace_min: number;
  evict_hours: number;
  backstop_margin_min: number; // the controller stops a pod this long after its own limits
  runner: boolean; // register each shared pod as a GitHub runner
  labels: string[]; // runner labels (+ fv-build-<region>)
  wake_on_queue: boolean; // queued GitHub jobs that need `labels[0]` wake a pod
  wake_workflows: string[]; // a queued/running run of these workflows wakes a pod too
  server_ref: string;
  image: string; // "" = the BASE_IMAGE_TAG pin of build-pod.sh at server_ref
  cache: { r2_bucket: string; r2_endpoint: string }; // "" endpoint: https://<CF_ACCOUNT_ID>.r2.cloudflarestorage.com
}

export const DEFAULT_BUILD_PODS_POLICY: BuildPodsPolicy = {
  enabled: false,
  max_pods: 2,
  max_dph_per_pod: 1.5,
  daily_usd_max: 20,
  balance_margin: 2,
  flavors: ["cpu5c", "cpu3c"],
  vcpus: [32, 16],
  disk_gb: 200,
  disk_gb_fallback: 80,
  regions: ["EU"],
  regions_only: false,
  volumes: {},
  idle_min: 20,
  max_h: 8,
  max_grace_min: 30,
  evict_hours: 6,
  backstop_margin_min: 30,
  runner: true,
  labels: ["fv-build"],
  wake_on_queue: true,
  wake_workflows: [],
  server_ref: "main",
  image: "",
  cache: { r2_bucket: "fv-build-cache", r2_endpoint: "" },
};

const FLAVOR_RE = /^cpu[0-9][a-z]$/;
const LABEL_RE = /^[A-Za-z0-9_.-]{1,40}$/;
const num = (v: unknown, d: number, lo: number, hi: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
};

/** Merges a stored or submitted policy over the defaults, clamped and validated. */
export function normalizePolicy(p: Partial<BuildPodsPolicy> | null | undefined): BuildPodsPolicy {
  const d = DEFAULT_BUILD_PODS_POLICY;
  const x = (p || {}) as any;
  const strs = (v: unknown, re: RegExp, dflt: string[]) => (Array.isArray(v) ? v.map(String).filter((s) => re.test(s)) : dflt);
  const flavors = strs(x.flavors, FLAVOR_RE, d.flavors);
  const vcpus = Array.isArray(x.vcpus) ? x.vcpus.map(Number).filter((n: number) => Number.isInteger(n) && n >= 2 && n <= 64) : d.vcpus;
  const labels = strs(x.labels, LABEL_RE, d.labels);
  const volumes: Record<string, string> = {};
  for (const [k, v] of Object.entries(x.volumes && typeof x.volumes === "object" ? x.volumes : {})) if (/^[A-Z]{2,4}(-[A-Z0-9]+)+$/.test(k) && /^[a-z0-9]{6,40}$/.test(String(v))) volumes[k] = String(v);
  const ref = typeof x.server_ref === "string" && /^[A-Za-z0-9._\/-]{1,200}$/.test(x.server_ref) ? x.server_ref : d.server_ref;
  const image = typeof x.image === "string" && /^([a-z0-9.-]+\/)?[a-z0-9._\/-]+(:[A-Za-z0-9._-]+|@sha256:[0-9a-f]{64})$/.test(x.image) ? x.image : "";
  const cache = x.cache && typeof x.cache === "object" ? x.cache : {};
  return {
    enabled: x.enabled === undefined ? d.enabled : !!x.enabled,
    max_pods: Math.round(num(x.max_pods, d.max_pods, 0, 8)),
    max_dph_per_pod: num(x.max_dph_per_pod, d.max_dph_per_pod, 0.05, 5),
    daily_usd_max: num(x.daily_usd_max, d.daily_usd_max, 0, 500),
    balance_margin: num(x.balance_margin, d.balance_margin, 0, 1000),
    flavors: flavors.length ? flavors : d.flavors,
    vcpus: vcpus.length ? vcpus : d.vcpus,
    disk_gb: Math.round(num(x.disk_gb, d.disk_gb, 20, 500)),
    disk_gb_fallback: Math.round(num(x.disk_gb_fallback, d.disk_gb_fallback, 20, 500)),
    regions: strs(x.regions, /^[A-Z]{2,4}(-[A-Z0-9]+)*$/, d.regions),
    regions_only: !!x.regions_only,
    volumes,
    idle_min: num(x.idle_min, d.idle_min, 5, 240),
    max_h: num(x.max_h, d.max_h, 0.5, 24),
    max_grace_min: num(x.max_grace_min, d.max_grace_min, 0, 120),
    evict_hours: num(x.evict_hours, d.evict_hours, 0, 72),
    backstop_margin_min: num(x.backstop_margin_min, d.backstop_margin_min, 5, 180),
    runner: x.runner === undefined ? d.runner : !!x.runner,
    labels: labels.length ? labels : d.labels,
    wake_on_queue: x.wake_on_queue === undefined ? d.wake_on_queue : !!x.wake_on_queue,
    wake_workflows: strs(x.wake_workflows, /^[A-Za-z0-9_.\/-]{1,100}$/, d.wake_workflows),
    server_ref: ref,
    image,
    cache: {
      r2_bucket: typeof cache.r2_bucket === "string" && /^[a-z0-9][a-z0-9-]{1,62}$/.test(cache.r2_bucket) ? cache.r2_bucket : d.cache.r2_bucket,
      r2_endpoint: typeof cache.r2_endpoint === "string" && /^https:\/\/[A-Za-z0-9.-]+$/.test(cache.r2_endpoint) ? cache.r2_endpoint : "",
    },
  };
}
export const buildPodsPolicy = async (env: Env) => normalizePolicy(await getSetting<Partial<BuildPodsPolicy>>(env, "build_pods", {}));

// ---------------------------------------------------------------- rows

export interface BuildPodRow {
  id: string;
  name: string;
  pod_id: string | null;
  provider: string;
  purpose: "shared" | "test";
  state: string;
  dc: string | null;
  region: string | null;
  flavor: string | null;
  vcpu: number | null;
  disk_gb: number | null;
  cost_per_hr: number | null;
  volume_id: string | null;
  image: string;
  server_ref: string;
  server_sha: string;
  token_sealed: string;
  token_sha: string;
  idle_min: number;
  max_h: number;
  max_grace_min: number;
  labels: string | null;
  runner_name: string | null;
  runner_id: number | null;
  runner_state: string | null;
  runner_error: string | null;
  runner_at: number | null;
  created_at: number;
  created_by: string;
  started_at: number | null;
  stopped_at: number | null;
  deleted_at: number | null;
  last_error: string | null;
}

export async function listBuildPods(env: Env, includeDeletedDays = 0): Promise<BuildPodRow[]> {
  const since = now() - includeDeletedDays * 86400_000;
  const r = await env.DB.prepare("SELECT * FROM build_pods WHERE state != 'deleted' OR deleted_at >= ? ORDER BY created_at DESC").bind(since).all<BuildPodRow>();
  return r.results || [];
}
export async function getBuildPod(env: Env, idOrPod: string): Promise<BuildPodRow> {
  const r = await env.DB.prepare("SELECT * FROM build_pods WHERE id = ? OR pod_id = ? OR name = ? ORDER BY created_at DESC LIMIT 1").bind(idOrPod, idOrPod, idOrPod).first<BuildPodRow>();
  if (!r) throw new HttpError(404, `no build pod ${idOrPod}`);
  return r;
}
async function update(env: Env, id: string, f: Record<string, unknown>): Promise<void> {
  const keys = Object.keys(f);
  if (!keys.length) return;
  await env.DB.prepare(`UPDATE build_pods SET ${keys.map((k) => `${k} = ?`).join(", ")} WHERE id = ?`).bind(...keys.map((k) => f[k] ?? null), id).run();
}
const aad = (id: string) => `build-pod:${id}`;
export async function buildPodToken(env: Env, row: BuildPodRow): Promise<string> {
  return unseal(env.CONTROL_KEK, row.token_sealed, aad(row.id));
}

// ---------------------------------------------------------------- locks

/** A lease on `key` for ttlMs (compare-and-set in D1); false when someone else holds it. */
export async function acquireLock(env: Env, key: string, holder: string, ttlMs: number): Promise<boolean> {
  const t = now();
  const r = await env.DB.prepare(
    "INSERT INTO locks (key, holder, expires_at) VALUES (?, ?, ?) ON CONFLICT (key) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at WHERE locks.expires_at < ?",
  )
    .bind(key, holder, t + ttlMs, t)
    .run();
  return Number((r as any)?.meta?.changes ?? 0) > 0;
}
export async function releaseLock(env: Env, key: string, holder: string): Promise<void> {
  await env.DB.prepare("DELETE FROM locks WHERE key = ? AND holder = ?").bind(key, holder).run();
}

// ---------------------------------------------------------------- placement

/** eu | us | ca | ap: the runner's region label, from a Runpod datacenter id. */
export function regionOf(dc: string): string {
  const p = dc.split("-")[0]!.toUpperCase();
  if (p === "EU" || p === "EUR") return "eu";
  if (p === "US") return "us";
  if (p === "CA") return "ca";
  return "ap";
}

export interface Candidate {
  dc: string;
  flavor: string;
  vcpu: number;
  ram: number;
  stock: string | null; // High | Medium | Low | null (none)
  price: number | null;
  volume_id?: string;
}
const STOCK_RANK: Record<string, number> = { High: 0, Medium: 1, Low: 2 };

/** `want`: a region (eu, us, ca, ap) or a datacenter id; empty: any. */
export function matchesRegion(dc: string, want?: string | null): boolean {
  if (!want) return true;
  return want.includes("-") ? dc === want : regionOf(dc) === want.toLowerCase();
}

/** Candidates with stock, cheap enough, in the wanted region, best first (docs §4). */
export function rankCandidates(cands: Candidate[], pol: BuildPodsPolicy, want?: string | null): Candidate[] {
  const pref = (dc: string) => {
    // "EU" (a region code) also takes EUR-*; "EUR-IS" or "EU-RO-1" are datacenter prefixes.
    const i = pol.regions.findIndex((r) => dc === r || dc.startsWith(r + "-") || (r.length === 2 && regionOf(dc) === r.toLowerCase()));
    return i < 0 ? pol.regions.length : i;
  };
  return cands
    .filter((c) => c.stock && c.stock in STOCK_RANK)
    .filter((c) => c.price === null || c.price <= pol.max_dph_per_pod)
    .filter((c) => matchesRegion(c.dc, want))
    .filter((c) => !pol.regions_only || pref(c.dc) < pol.regions.length)
    .sort(
      (a, b) =>
        Number(!a.volume_id) - Number(!b.volume_id) ||
        pref(a.dc) - pref(b.dc) ||
        pol.vcpus.indexOf(a.vcpu) - pol.vcpus.indexOf(b.vcpu) ||
        pol.flavors.indexOf(a.flavor) - pol.flavors.indexOf(b.flavor) ||
        STOCK_RANK[a.stock!]! - STOCK_RANK[b.stock!]! ||
        (a.price ?? 99) - (b.price ?? 99) ||
        a.dc.localeCompare(b.dc),
    );
}

/** CPU stock and price per (datacenter, flavor, size) from Runpod's GraphQL: two calls. */
export async function cpuCandidates(env: Env, pol: BuildPodsPolicy, want?: string | null, busyVolumes: Set<string> = new Set()): Promise<Candidate[]> {
  const d = await runpod.gql<any>(env, "{ dataCenters { id listed } cpuFlavors { id ramMultiplier } }");
  const mult = new Map<string, number>((d.cpuFlavors || []).map((f: any) => [String(f.id), Number(f.ramMultiplier || 2)]));
  const dcs: string[] = (d.dataCenters || []).filter((x: any) => x.listed !== false).map((x: any) => String(x.id)).filter((dc: string) => matchesRegion(dc, want));
  const combos: { dc: string; flavor: string; vcpu: number; ram: number; alias: string }[] = [];
  for (const dc of dcs)
    for (const flavor of pol.flavors)
      for (const vcpu of pol.vcpus) {
        if (!mult.has(flavor)) continue;
        combos.push({ dc, flavor, vcpu, ram: vcpu * mult.get(flavor)!, alias: `a${combos.length}` });
      }
  if (!combos.length) return [];
  const out: Candidate[] = [];
  // Chunks of 60 aliases per query keep each answer small.
  for (let i = 0; i < combos.length; i += 60) {
    const part = combos.slice(i, i + 60);
    const q = `{ ${part.map((c) => `${c.alias}: cpuFlavors { id specifics(input: {dataCenterId: ${JSON.stringify(c.dc)}, instanceId: ${JSON.stringify(`${c.flavor}-${c.vcpu}-${c.ram}`)}}) { stockStatus securePrice } }`).join(" ")} }`;
    const r = await runpod.gql<any>(env, q);
    for (const c of part) {
      const s = ((r?.[c.alias] || []) as any[]).find((f) => f.id === c.flavor)?.specifics;
      const vol = pol.volumes[c.dc];
      out.push({ dc: c.dc, flavor: c.flavor, vcpu: c.vcpu, ram: c.ram, stock: s?.stockStatus ?? null, price: s?.securePrice ?? null, ...(vol && !busyVolumes.has(vol) ? { volume_id: vol } : {}) });
    }
  }
  return out;
}

// ---------------------------------------------------------------- what a pod runs

export interface ServerBundle {
  ref: string;
  sha: string; // sha256[:12] of the server file, as the server reports it
  b64: string; // gzip + base64
  image: string;
}
const SERVER_PATH = "scripts/dev/build-pod-server.py";
const CLIENT_PATH = "scripts/dev/build-pod.sh";

export function imagePin(buildPodSh: string): string | null {
  const tag = /^BASE_IMAGE_TAG="([A-Za-z0-9._-]+)"/m.exec(buildPodSh)?.[1];
  const repo = /fastvideo-rs-build-base/.test(buildPodSh) ? "ghcr.io/zaitrarrio/fastvideo-rs-build-base" : null;
  return tag && repo ? `${repo}:${tag}` : null;
}

async function gzipB64(text: string): Promise<string> {
  const cs = new Blob([new TextEncoder().encode(text)]).stream().pipeThrough(new CompressionStream("gzip"));
  return b64(new Uint8Array(await new Response(cs).arrayBuffer()));
}

/** The server and image at `ref` (fetched from the repository). */
export async function serverBundle(env: Env, pol: BuildPodsPolicy, ref = pol.server_ref): Promise<ServerBundle> {
  const [src, client] = await Promise.all([repoFile(env, SERVER_PATH, ref), pol.image ? Promise.resolve("") : repoFile(env, CLIENT_PATH, ref)]);
  const image = pol.image || imagePin(client);
  if (!image) throw new HttpError(502, `no BASE_IMAGE_TAG pin in ${CLIENT_PATH}@${ref}`);
  const sha = (await sha256Hex(new TextEncoder().encode(src))).slice(0, 12);
  const bundle = { ref, sha, b64: await gzipB64(src), image };
  if (ref === pol.server_ref) await putSetting(env, "build_pods_current", { ref, sha, image, at: now() }, "build-pods");
  return bundle;
}
export interface CurrentRef {
  ref: string;
  sha: string;
  image: string;
  at: number;
}
export const currentRef = (env: Env) => getSetting<CurrentRef | { at: 0 }>(env, "build_pods_current", { at: 0 });
export function isOutdated(row: Pick<BuildPodRow, "server_ref" | "server_sha" | "image">, cur: CurrentRef | { at: 0 }): boolean {
  if (!("sha" in cur) || row.server_ref !== cur.ref) return false;
  return row.server_sha !== cur.sha || row.image !== cur.image;
}

// ---------------------------------------------------------------- payload

export interface PayloadIn {
  name: string;
  image: string;
  flavor: string;
  vcpu: number;
  disk_gb: number;
  dc: string;
  volume_id?: string | null;
  token_sha: string;
  server_b64: string;
  pol: BuildPodsPolicy;
  r2?: { endpoint: string; bucket: string; key_id: string; secret: string } | null;
}
/** Runpod's PodCreateInput for a build pod (the env is never returned by the API). */
export function buildPodPayload(p: PayloadIn): Record<string, unknown> {
  const env: Record<string, string> = {
    FV_BUILD_TOKEN_SHA256: p.token_sha,
    FV_BUILD_SERVER_B64: p.server_b64,
    FV_BUILD_IDLE_MIN: String(p.pol.idle_min),
    FV_BUILD_MAX_HOURS: String(p.pol.max_h),
    FV_BUILD_MAX_GRACE_MIN: String(p.pol.max_grace_min),
    FV_BUILD_EVICT_HOURS: String(p.pol.evict_hours),
    FV_BUILD_EVICT_FREE_GB: String(Math.min(Math.floor(p.disk_gb / 5), 40)),
    FV_BUILD_IMAGE: p.image,
    FV_BUILD_POD_NAME: p.name,
    FV_BUILD_MANAGED: "fv-control",
  };
  if (!p.volume_id) {
    // No network volume: the caches' root is on the container disk (R2 shares sccache and seeds).
    env.FV_BUILD_ROOT = "/root/fvb-cache";
    env.FV_BUILD_SCCACHE_SIZE = "10G";
  }
  if (p.r2) {
    env.FV_BUILD_R2_ENDPOINT = p.r2.endpoint;
    env.FV_BUILD_R2_BUCKET = p.r2.bucket;
    env.FV_BUILD_R2_ACCESS_KEY_ID = p.r2.key_id;
    env.FV_BUILD_R2_SECRET_ACCESS_KEY = p.r2.secret;
  }
  return {
    name: p.name,
    imageName: p.image,
    computeType: "CPU",
    cloudType: "SECURE",
    cpuFlavorIds: [p.flavor],
    cpuFlavorPriority: "custom",
    vcpuCount: p.vcpu,
    containerDiskInGb: p.disk_gb,
    volumeInGb: 0,
    ...(p.volume_id ? { networkVolumeId: p.volume_id, volumeMountPath: "/workspace" } : {}),
    dataCenterIds: [p.dc],
    ports: ["8000/http"],
    dockerStartCmd: ["/bin/bash", "-c", BUILD_POD_START_CMD],
    env,
  };
}
function r2Of(env: Env, pol: BuildPodsPolicy): PayloadIn["r2"] {
  if (!env.BUILD_CACHE_R2_ACCESS_KEY_ID || !env.BUILD_CACHE_R2_SECRET_ACCESS_KEY) return null;
  const endpoint = pol.cache.r2_endpoint || (env.CF_ACCOUNT_ID ? `https://${env.CF_ACCOUNT_ID}.r2.cloudflarestorage.com` : "");
  if (!endpoint) return null;
  return { endpoint, bucket: pol.cache.r2_bucket, key_id: env.BUILD_CACHE_R2_ACCESS_KEY_ID, secret: env.BUILD_CACHE_R2_SECRET_ACCESS_KEY };
}

// ---------------------------------------------------------------- guards

export async function buildPodSpendToday(env: Env): Promise<number> {
  const r = await env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS usd FROM cost_daily WHERE day = ? AND owner LIKE 'build-pod:%'").bind(utcDay(now())).first<{ usd: number }>();
  return Number(r?.usd || 0);
}
async function guard(env: Env, pol: BuildPodsPolicy): Promise<void> {
  if (!pol.enabled) throw new HttpError(403, "build pods are disabled (policy build_pods.enabled; Settings)");
  const { balance } = await runpod.account(env);
  const need = defaults.balanceFloor(env) + pol.balance_margin;
  if (balance < need) throw new HttpError(402, `Runpod balance $${balance.toFixed(2)} is below $${need.toFixed(2)} (floor + build_pods.balance_margin)`);
  const spent = await buildPodSpendToday(env);
  if (spent >= pol.daily_usd_max) throw new HttpError(402, `build pods spent $${spent.toFixed(2)} today (build_pods.daily_usd_max $${pol.daily_usd_max})`);
}

// ---------------------------------------------------------------- lifecycle

/** Reads Runpod's state of each live row (REST, one call per pod) and writes it back. */
async function refresh(env: Env, rows: BuildPodRow[]): Promise<BuildPodRow[]> {
  const out: BuildPodRow[] = [];
  for (const r of rows) {
    if (!r.pod_id || r.state === "deleted") {
      out.push(r);
      continue;
    }
    const p = await runpod.pod(env, r.pod_id);
    const state = !p ? "deleted" : p.desiredStatus === "RUNNING" ? "running" : p.desiredStatus === "EXITED" || p.desiredStatus === "TERMINATED" ? "stopped" : r.state;
    if (state !== r.state) {
      const f: Record<string, unknown> = { state };
      if (state === "deleted") f.deleted_at = now();
      if (state === "stopped") f.stopped_at = now();
      await update(env, r.id, f);
      Object.assign(r, f);
    }
    out.push(r);
  }
  return out;
}

export interface UpResult {
  action: "reused" | "started" | "created";
  pod: BuildPodRow;
  replaced?: string[];
}

/**
 * A ready-to-use shared pod (docs §4): the least busy running one, else
 * start a stopped current one, else create (deleting stopped outdated ones).
 * `needIdleRunner`: skip running pods whose runner is busy (scale-out for
 * queued CI jobs).
 */
export async function buildPodsUp(env: Env, actor: string, o: { region?: string | null; needIdleRunner?: boolean } = {}): Promise<UpResult> {
  const pol = await buildPodsPolicy(env);
  const holder = `${actor}:${randomToken("", 4)}`;
  if (!(await acquireLock(env, "build-pods:up", holder, 180_000))) throw new HttpError(409, "another build pod `up` is in progress; retry in a few seconds", { retry_after_s: 10 });
  try {
    if (!pol.enabled) throw new HttpError(403, "build pods are disabled (policy build_pods.enabled; Settings)");
    const rows = (await refresh(env, (await listBuildPods(env)).filter((r) => r.purpose === "shared"))).filter((r) => r.state !== "deleted" && r.state !== "failed");
    const inRegion = (r: BuildPodRow) => !o.region || (r.dc ? matchesRegion(r.dc, o.region) : false);
    // 1. a running pod
    let running = rows.filter((r) => (r.state === "running" || r.state === "creating") && inRegion(r));
    if (o.needIdleRunner && running.length) {
      const runners = runnerPatSet(env) ? await listRunners(env).catch(() => [] as GhRunner[]) : [];
      running = running.filter((r) => !runners.some((g) => g.name === r.runner_name && g.busy));
    }
    if (running.length) {
      const load: [BuildPodRow, number][] = [];
      for (const r of running) load.push([r, r.pod_id ? ((await buildPodHealth(env, r.pod_id))?.jobs_active ?? 0) : 0]);
      load.sort((a, b) => a[1] - b[1]);
      // Keeps the "outdated" mark current (the pod itself is left alone while it runs).
      await serverBundle(env, pol).catch(() => null);
      return { action: "reused", pod: load[0]![0] };
    }
    await guard(env, pol);
    const cur = await serverBundle(env, pol).catch(() => null);
    const curRef = cur ? { ref: cur.ref, sha: cur.sha, image: cur.image, at: now() } : await currentRef(env);
    // 2. a stopped current pod
    const stopped = rows.filter((r) => r.state === "stopped" && r.pod_id);
    const replaced: string[] = [];
    for (const r of stopped.filter((x) => inRegion(x) && !isOutdated(x, curRef))) {
      try {
        await runpod.start(env, r.pod_id!);
        await update(env, r.id, { state: "running", started_at: now(), runner_state: "none", runner_id: null, last_error: null });
        await audit(env, { actor, action: "build_pod.start", target: r.pod_id!, detail: r.name });
        return { action: "started", pod: await getBuildPod(env, r.id) };
      } catch (e) {
        // The host has no free CPU any more: this pod goes, a new one is placed.
        await deleteRow(env, r, actor, `start refused: ${(e as Error).message.slice(0, 160)}`);
        replaced.push(r.pod_id!);
      }
    }
    // 3. replace stopped outdated pods (a stopped pod runs nothing), then create
    for (const r of stopped.filter((x) => isOutdated(x, curRef))) {
      await deleteRow(env, r, actor, "outdated server or image (replaced while stopped)");
      replaced.push(r.pod_id!);
    }
    const live = rows.filter((r) => r.state === "running" || r.state === "creating").length;
    if (live >= pol.max_pods) throw new HttpError(409, `${live} build pod(s) already run (build_pods.max_pods ${pol.max_pods})`);
    if (!cur) await serverBundle(env, pol); // surfaces the GitHub error
    const row = await createBuildPod(env, actor, pol, { region: o.region, bundle: cur!, purpose: "shared" });
    return { action: "created", pod: row, replaced };
  } finally {
    await releaseLock(env, "build-pods:up", holder);
  }
}

/** Places and creates a new pod (a shared one through `up`, or a test pod with its own ref). */
export async function createBuildPod(
  env: Env,
  actor: string,
  pol: BuildPodsPolicy,
  o: { region?: string | null; bundle?: ServerBundle; purpose: "shared" | "test"; server_ref?: string },
): Promise<BuildPodRow> {
  await guard(env, pol);
  const bundle = o.bundle ?? (await serverBundle(env, pol, o.server_ref || pol.server_ref));
  const others = await listBuildPods(env);
  const busyVolumes = new Set(others.filter((r) => r.volume_id && r.state !== "deleted" && r.state !== "stopped").map((r) => r.volume_id!));
  const cands = rankCandidates(await cpuCandidates(env, pol, o.region, busyVolumes), pol, o.region);
  if (!cands.length) throw new HttpError(503, `no CPU stock for [${pol.flavors.join(" ")}] × [${pol.vcpus.join(" ")}] vCPU${o.region ? ` in ${o.region}` : ""} at most $${pol.max_dph_per_pod}/hr`);
  const id = `bp_${randomToken("", 6)}`;
  const token = randomToken("", 32);
  const tokenSha = await sha256Hex(token);
  const t = now();
  const labels = pol.labels.join(",");
  await env.DB.prepare(
    `INSERT INTO build_pods (id, name, purpose, state, image, server_ref, server_sha, token_sealed, token_sha, idle_min, max_h, max_grace_min, labels, runner_state, created_at, created_by)
     VALUES (?, ?, ?, 'creating', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'none', ?, ?)`,
  )
    .bind(id, `fv-build-pending-${id.slice(3)}`, o.purpose, bundle.image, bundle.ref, bundle.sha, await seal(env.CONTROL_KEK, token, aad(id)), tokenSha, pol.idle_min, pol.max_h, pol.max_grace_min, labels, t, actor)
    .run();
  const errors: string[] = [];
  const r2 = r2Of(env, pol);
  for (const c of cands.slice(0, 8)) {
    const region = regionOf(c.dc);
    const name = `fv-build-${region}-${id.slice(3, 9)}${o.purpose === "test" ? "-test" : ""}`;
    let disk = c.vcpu === pol.vcpus[0] ? pol.disk_gb : Math.min(pol.disk_gb_fallback, pol.disk_gb);
    for (let attempt = 0; attempt < 2; attempt++) {
      const payload = buildPodPayload({ name, image: bundle.image, flavor: c.flavor, vcpu: c.vcpu, disk_gb: disk, dc: c.dc, volume_id: c.volume_id, token_sha: tokenSha, server_b64: bundle.b64, pol, r2 });
      try {
        const res = await runpod.create(env, payload);
        const podId = String(res?.id || "");
        if (!podId) throw new HttpError(502, "runpod: create answered without an id");
        const dph = Number(res.costPerHr ?? c.price ?? 0);
        if (dph > pol.max_dph_per_pod) {
          await runpod.remove(env, podId).catch(() => {});
          errors.push(`${c.flavor}-${c.vcpu} in ${c.dc}: $${dph}/hr over the cap (deleted)`);
          break;
        }
        await update(env, id, {
          name,
          pod_id: podId,
          state: "running",
          dc: c.dc,
          region,
          flavor: c.flavor,
          vcpu: c.vcpu,
          disk_gb: disk,
          cost_per_hr: dph,
          volume_id: c.volume_id ?? null,
          labels: o.purpose === "shared" ? [...pol.labels, `${pol.labels[0]}-${region}`].join(",") : labels,
          started_at: now(),
          last_error: errors.length ? errors.join("; ").slice(0, 1000) : null,
        });
        await audit(env, { actor, action: "build_pod.create", target: podId, after: { name, dc: c.dc, flavor: c.flavor, vcpu: c.vcpu, disk_gb: disk, dph, image: bundle.image, server: `${bundle.ref}@${bundle.sha}`, volume: c.volume_id ?? null, r2: !!r2, purpose: o.purpose } });
        return getBuildPod(env, id);
      } catch (e) {
        const msg = scrub(env, (e as Error).message);
        const cap = /Container Disk must be less than or equal to (\d+)/.exec(msg);
        if (cap && Number(cap[1]) < disk && attempt === 0) {
          disk = Number(cap[1]);
          continue;
        }
        errors.push(`${c.flavor}-${c.vcpu} in ${c.dc}: ${msg.slice(0, 160)}`);
        break;
      }
    }
  }
  await update(env, id, { state: "failed", last_error: errors.join("; ").slice(0, 2000) });
  await audit(env, { actor, action: "build_pod.create", target: id, ok: false, detail: errors.join("; ") });
  throw new HttpError(503, `no build pod could be created: ${errors.join("; ").slice(0, 600)}`);
}

/** Whether a pod is busy: jobs on its server, or its GitHub runner running a workflow job. */
export async function busyReason(env: Env, row: BuildPodRow): Promise<string | null> {
  if (row.state !== "running" || !row.pod_id) return null;
  const h = await buildPodHealth(env, row.pod_id);
  if (h && Number(h.jobs_active || 0) > 0) return `${h.jobs_active} job(s) active`;
  if (row.runner_name && runnerPatSet(env)) {
    const g = (await listRunners(env).catch(() => [] as GhRunner[])).find((x) => x.name === row.runner_name);
    if (g?.busy) return `its runner ${g.name} runs a workflow job`;
  }
  return null;
}

/** Removes the pod's GitHub runner (never a busy one). */
export async function deregisterRunner(env: Env, row: BuildPodRow, runners?: GhRunner[]): Promise<string> {
  if (!row.runner_name || !runnerPatSet(env)) return "no runner";
  const list = runners ?? (await listRunners(env));
  const g = list.find((x) => x.name === row.runner_name);
  if (g?.busy) throw new HttpError(409, `runner ${g.name} is busy`);
  if (g) await deleteRunner(env, g.id);
  await update(env, row.id, { runner_state: "removed", runner_id: null });
  return g ? `runner ${g.name} removed` : "runner already gone";
}

export async function stopBuildPodRow(env: Env, row: BuildPodRow, actor: string, o: { force?: boolean; reason?: string } = {}): Promise<string> {
  if (!row.pod_id || row.state === "deleted") throw new HttpError(409, `${row.name} has no pod (${row.state})`);
  if (!o.force) {
    const busy = await busyReason(env, row);
    if (busy) throw new HttpError(409, `${row.name} is busy (${busy}); pass force to stop anyway`);
  }
  const rn = await deregisterRunner(env, row).catch((e) => `runner: ${(e as Error).message.slice(0, 120)}`);
  const result = await stopBuildPod(env, row.pod_id);
  const gone = result.startsWith("terminated");
  await update(env, row.id, gone ? { state: "deleted", deleted_at: now() } : { state: "stopped", stopped_at: now() });
  await audit(env, { actor, action: "build_pod.stop", target: row.pod_id, detail: `${row.name}: ${result}; ${rn}${o.reason ? ` (${o.reason})` : ""}${o.force ? " [force]" : ""}` });
  return result;
}

async function deleteRow(env: Env, row: BuildPodRow, actor: string, reason: string): Promise<void> {
  await deregisterRunner(env, row).catch(() => {});
  if (row.pod_id) await runpod.remove(env, row.pod_id);
  await update(env, row.id, { state: "deleted", deleted_at: now(), last_error: reason.slice(0, 500) });
  await audit(env, { actor, action: "build_pod.delete", target: row.pod_id || row.id, detail: `${row.name}: ${reason}` });
}
export async function deleteBuildPodRow(env: Env, row: BuildPodRow, actor: string, o: { force?: boolean } = {}): Promise<void> {
  if (row.state === "deleted") return;
  if (!o.force) {
    const busy = await busyReason(env, row);
    if (busy) throw new HttpError(409, `${row.name} is busy (${busy}); pass force to delete anyway`);
  }
  await deleteRow(env, row, actor, o.force ? "deleted (force)" : "deleted");
}

/** Starts one stopped pod (the console's Start); `up` is what clients use. */
export async function startBuildPodRow(env: Env, row: BuildPodRow, actor: string): Promise<void> {
  const pol = await buildPodsPolicy(env);
  await guard(env, pol);
  if (row.state !== "stopped" || !row.pod_id) throw new HttpError(409, `${row.name} is ${row.state}`);
  await runpod.start(env, row.pod_id);
  await update(env, row.id, { state: "running", started_at: now(), runner_state: "none", runner_id: null, last_error: null });
  await audit(env, { actor, action: "build_pod.start", target: row.pod_id, detail: row.name });
}

// ---------------------------------------------------------------- runner

async function podCall(env: Env, row: BuildPodRow, method: string, path: string, body?: unknown): Promise<Response> {
  const token = await buildPodToken(env, row);
  return fetchWithTimeout(`${defaults.podUrl(env, row.pod_id!)}${path}`, {
    method,
    headers: { authorization: `Bearer ${token}`, ...(body !== undefined ? { "content-type": "application/json" } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
    timeoutMs: 20000,
  });
}

/** Sends a fresh registration token to the pod's POST /v1/runner (PR #32's contract). */
export async function registerRunner(env: Env, row: BuildPodRow): Promise<string> {
  if (!row.pod_id || row.state !== "running") throw new HttpError(409, `${row.name} is not running`);
  const name = `fv-build-${row.pod_id}`;
  const regToken = await runnerRegistrationToken(env);
  const r = await podCall(env, row, "POST", "/v1/runner", { token: regToken, repo: defaults.githubRepo(env), labels: row.labels || "fv-build", name });
  const t = now();
  if (r.status === 404) {
    await update(env, row.id, { runner_state: "unsupported", runner_error: "the pod's server has no /v1/runner (before PR #32)", runner_at: t });
    return "unsupported";
  }
  if (r.status !== 202 && r.status !== 200) {
    const err = scrub(env, (await r.text()).slice(0, 300)).split(regToken).join("<token>");
    await update(env, row.id, { runner_state: "failed", runner_error: `POST /v1/runner: ${r.status} ${err}`, runner_at: t });
    return "failed";
  }
  await update(env, row.id, { runner_state: "registering", runner_name: name, runner_error: null, runner_at: t });
  return "registering";
}

/** The pod's GET /v1/runner phase → runner_state. */
async function pollRunner(env: Env, row: BuildPodRow): Promise<void> {
  const r = await podCall(env, row, "GET", "/v1/runner").catch(() => null);
  if (!r || !r.ok) return;
  const j = (await r.json().catch(() => ({}))) as { phase?: string; error?: string | null; name?: string };
  if (j.phase === "running") await update(env, row.id, { runner_state: "running", runner_error: null, runner_name: j.name || row.runner_name });
  else if (j.phase === "failed" || String(j.phase || "").startsWith("exited")) await update(env, row.id, { runner_state: "failed", runner_error: scrub(env, String(j.error || j.phase)).slice(0, 500) });
}

// ---------------------------------------------------------------- views

export interface BuildPodView {
  id: string;
  name: string;
  pod_id: string | null;
  purpose: string;
  state: string;
  dc: string | null;
  region: string | null;
  flavor: string | null;
  vcpu: number | null;
  disk_gb: number | null;
  cost_per_hr: number | null;
  volume_id: string | null;
  image: string;
  server: string;
  outdated: boolean;
  url: string | null;
  limits: { idle_min: number; max_h: number; max_grace_min: number };
  runner: { state: string | null; name: string | null; labels: string | null; error: string | null };
  spend_today: number;
  health: BuildPodHealth | null;
  phase: string | null; // the server's setup phase (ready when usable)
  created_at: number;
  created_by: string;
  started_at: number | null;
  last_error: string | null;
}
export async function buildPodView(env: Env, r: BuildPodRow, cur: CurrentRef | { at: 0 }, withHealth = true): Promise<BuildPodView> {
  const h = withHealth && r.state === "running" && r.pod_id ? await buildPodHealth(env, r.pod_id) : null;
  // Right after a start the proxy can answer from the previous container.
  const fresh = h && (!r.started_at || !h.boot || h.boot * 1000 >= r.started_at - 30_000);
  const spend = await env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS usd FROM cost_daily WHERE day = ? AND pod_id = ?").bind(utcDay(now()), r.pod_id || "-").first<{ usd: number }>();
  if (h?.self_stop?.error) h.self_stop.error = scrub(env, h.self_stop.error);
  return {
    id: r.id,
    name: r.name,
    pod_id: r.pod_id,
    purpose: r.purpose,
    state: r.state,
    dc: r.dc,
    region: r.region,
    flavor: r.flavor,
    vcpu: r.vcpu,
    disk_gb: r.disk_gb,
    cost_per_hr: r.cost_per_hr,
    volume_id: r.volume_id,
    image: r.image,
    server: `${r.server_ref}@${r.server_sha}`,
    outdated: isOutdated(r, cur),
    url: r.pod_id ? defaults.podUrl(env, r.pod_id) : null,
    limits: { idle_min: r.idle_min, max_h: r.max_h, max_grace_min: r.max_grace_min },
    runner: { state: r.runner_state, name: r.runner_name, labels: r.labels, error: r.runner_error },
    spend_today: Number(spend?.usd || 0),
    health: fresh ? h : null,
    phase: r.state === "running" ? (fresh ? (h?.phase ?? (h?.ready ? "ready" : "starting")) : "starting") : null,
    created_at: r.created_at,
    created_by: r.created_by,
    started_at: r.started_at,
    last_error: r.last_error,
  };
}
export async function buildPodsOverview(env: Env) {
  const [pol, rows] = await Promise.all([buildPodsPolicy(env), listBuildPods(env, 1)]);
  let cur = await currentRef(env);
  if (now() - cur.at > 10 * 60_000 && rows.some((r) => r.state !== "deleted")) {
    const b = await serverBundle(env, pol).catch(() => null);
    if (b) cur = { ref: b.ref, sha: b.sha, image: b.image, at: now() };
  }
  const pods: BuildPodView[] = [];
  for (const r of rows) pods.push(await buildPodView(env, r, cur));
  return {
    pods,
    policy: pol,
    current: cur,
    spend_today: await buildPodSpendToday(env),
    secrets: { runner_pat: runnerPatSet(env), r2_cache: !!r2Of(env, pol) },
  };
}

// ---------------------------------------------------------------- CI builder choice (docs §6)

export interface CiAnswer {
  builder: "pod" | "github" | "wait";
  reason: string;
  retry_after_s?: number;
  pod?: { id: string; region: string | null; runner: string | null };
}
/** Where a CI job that wants `label` should build: an idle runner now, a pod being woken, or GitHub-hosted. */
export async function ciBuildRunner(env: Env, actor: string, o: { label?: string; region?: string | null; wake?: boolean } = {}): Promise<CiAnswer> {
  const pol = await buildPodsPolicy(env);
  const label = o.label && LABEL_RE.test(o.label) ? o.label : pol.labels[0]!;
  if (!runnerPatSet(env)) return { builder: "github", reason: "fv-control has no GITHUB_RUNNER_PAT: it cannot see runners" };
  const runners = await listRunners(env);
  const idle = runners.filter((g) => g.status === "online" && !g.busy && g.labels.includes(label) && (!o.region || g.labels.includes(`${label}-${o.region}`)));
  if (idle.length) return { builder: "pod", reason: `${idle.length} idle ${label} runner(s) online` };
  if (!pol.enabled || !pol.runner) return { builder: "github", reason: "no idle runner, and fv-control's build pods are disabled" };
  if (o.wake === false) return { builder: "github", reason: "no idle runner (wake=false)" };
  try {
    const up = await buildPodsUp(env, actor, { region: o.region, needIdleRunner: true });
    const pod = { id: up.pod.id, region: up.pod.region, runner: up.pod.runner_name };
    return { builder: "wait", reason: `${up.action} build pod ${up.pod.name}; its runner is not online yet`, retry_after_s: 15, pod };
  } catch (e) {
    if (e instanceof HttpError && e.status === 409 && /in progress/.test(e.message)) return { builder: "wait", reason: e.message, retry_after_s: 10 };
    return { builder: "github", reason: `no idle runner and no pod: ${(e as Error).message.slice(0, 200)}` };
  }
}

// ---------------------------------------------------------------- cron (collector.ts)

export interface BuildPodsTick {
  alerts: AlertIn[];
  actions: string[];
}

/** The managed rows by Runpod pod id (the collector's owner attribution). */
export async function managedPodNames(env: Env): Promise<Map<string, string>> {
  const r = await env.DB.prepare("SELECT pod_id, name FROM build_pods WHERE pod_id IS NOT NULL").all<{ pod_id: string; name: string }>();
  return new Map((r.results || []).map((x) => [x.pod_id, x.name]));
}

/**
 * Every minute, with the pods of the collector's one GraphQL call: state
 * reconcile, the per-pod backstop, the balance floor, runner registration and
 * removal, and the wake on queued CI jobs.
 */
export async function buildPodsTick(env: Env, pods: RunpodPod[], balance: number, o: { stopOnFloor: boolean; floor: number }): Promise<BuildPodsTick> {
  const alerts: AlertIn[] = [];
  const actions: string[] = [];
  const pol = await buildPodsPolicy(env);
  const t = now();
  const byId = new Map(pods.map((p) => [p.id, p]));
  const rows = await listBuildPods(env);
  for (const r of rows) {
    if (!r.pod_id) {
      // A create that never answered (the Worker died mid-call): give up after 10 min.
      if (r.state === "creating" && t - r.created_at > 600_000) await update(env, r.id, { state: "failed", last_error: "create never completed" });
      continue;
    }
    const p = byId.get(r.pod_id);
    const state = !p ? "deleted" : p.desiredStatus === "RUNNING" ? "running" : "stopped";
    if (state !== r.state && !(r.state === "failed" && state === "deleted")) {
      const f: Record<string, unknown> = { state };
      if (state === "deleted") f.deleted_at = t;
      if (state === "stopped") f.stopped_at = t;
      if (state === "running" && r.state !== "creating") Object.assign(f, { started_at: t, runner_state: "none", runner_id: null });
      await update(env, r.id, f);
      Object.assign(r, f);
    }
    if (p && Number(p.costPerHr) !== r.cost_per_hr) await update(env, r.id, { cost_per_hr: Number(p.costPerHr) });
  }
  const live = rows.filter((r) => r.state === "running" && r.pod_id);

  // Balance floor: stop them like controller clusters.
  if (o.stopOnFloor && balance < o.floor)
    for (const r of live) {
      const res = await stopBuildPodRow(env, r, "policy:balance_floor", { force: true, reason: "balance floor" }).catch((e) => `stop failed: ${(e as Error).message.slice(0, 120)}`);
      actions.push(`build pod ${r.name}: ${res} (balance floor)`);
      r.state = "stopped";
    }

  // Per-pod backstop: its own limits plus the margin.
  for (const r of live.filter((x) => x.state === "running")) {
    const p = byId.get(r.pod_id!)!;
    const h = await buildPodHealth(env, r.pod_id!);
    const why = buildPodVerdict(p.runtime?.uptimeInSeconds, h, {
      build_pod_backstop: true,
      build_pod_max_h: r.max_h + (r.max_grace_min + pol.backstop_margin_min) / 60,
      build_pod_idle_grace_min: pol.backstop_margin_min,
    });
    if (!why) continue;
    const res = await stopBuildPodRow(env, r, "policy:build_pod_backstop", { force: true, reason: why }).catch((e) => `stop failed: ${(e as Error).message.slice(0, 120)}`);
    alerts.push({ key: `build_pod:${r.pod_id}`, kind: "build_pod", severity: "critical", target: r.pod_id!, message: `${r.name} ${r.pod_id}: ${why}: ${res}`, action: "stop" });
    actions.push(`build pod ${r.name}: ${res} (${why})`);
    r.state = "stopped";
  }

  const spent = await buildPodSpendToday(env);
  if (spent > pol.daily_usd_max) alerts.push({ key: "build_pod_spend", kind: "build_pod_spend", severity: "warn", message: `build pods spent $${spent.toFixed(2)} today (> $${pol.daily_usd_max}); new pods are refused` });

  // Runners: register ready pods, poll registering ones, remove stopped pods' runners and stale offline ones.
  if (runnerPatSet(env)) {
    let runners: GhRunner[] | null = null;
    const getRunners = async () => (runners ??= await listRunners(env));
    for (const r of rows) {
      try {
        if (r.state === "running" && r.purpose === "shared" && pol.runner) {
          if (r.runner_state === "registering") await pollRunner(env, r);
          const retry = r.runner_state === "failed" && t - Number(r.runner_at || 0) > 600_000;
          const stale = r.runner_state === "registering" && t - Number(r.runner_at || 0) > 900_000;
          if (!r.runner_state || r.runner_state === "none" || r.runner_state === "removed" || retry || stale) {
            const h = await buildPodHealth(env, r.pod_id!);
            const fresh = h && (!r.started_at || !h.boot || h.boot * 1000 >= r.started_at - 30_000);
            if (fresh && (h!.ready || h!.phase === "ready")) actions.push(`build pod ${r.name}: runner ${await registerRunner(env, r)}`);
          }
        } else if ((r.state === "stopped" || r.state === "deleted") && (r.runner_state === "running" || r.runner_state === "registering")) {
          actions.push(`build pod ${r.name}: ${await deregisterRunner(env, r, await getRunners())}`);
        }
      } catch (e) {
        alerts.push({ key: `build_pod_runner:${r.id}`, kind: "build_pod_runner", severity: "warn", target: r.pod_id || r.id, message: `${r.name}: runner: ${scrub(env, (e as Error).message).slice(0, 200)}` });
      }
    }
    // Offline runners of our pods that do not run (the container disk, and the runner's credentials, are gone).
    // Only pods this controller created: a runner of another pod is never touched.
    const runningIds = new Set(pods.filter((p) => p.desiredStatus === "RUNNING").map((p) => p.id));
    const ours = new Set((await env.DB.prepare("SELECT pod_id FROM build_pods WHERE pod_id IS NOT NULL AND created_at > ?").bind(t - 30 * 86400_000).all<{ pod_id: string }>()).results?.map((x) => x.pod_id) || []);
    if (ours.size && t % (15 * 60_000) < 60_000) {
      for (const g of await getRunners().catch(() => [] as GhRunner[])) {
        const m = /^fv-build-([a-z0-9]{8,20})$/.exec(g.name);
        if (m && ours.has(m[1]!) && g.status === "offline" && !g.busy && !runningIds.has(m[1]!)) {
          await deleteRunner(env, g.id).catch(() => {});
          actions.push(`runner ${g.name} removed (offline, pod not running)`);
        }
      }
    }

    // Wake on demand: queued jobs that need our label and no idle runner that fits.
    if (pol.enabled && pol.runner && (pol.wake_on_queue || pol.wake_workflows.length)) {
      try {
        const label = pol.labels[0]!;
        const q = await queuedJobs(env, label);
        const rs = await getRunners();
        const fits = (labels: string[]) => rs.some((g) => g.status === "online" && !g.busy && labels.every((l) => l === "self-hosted" || l === "linux" || l === "x64" || g.labels.includes(l)));
        const waiting = pol.wake_on_queue ? q.jobs.filter((j) => !fits(j.labels)) : [];
        const wf = q.active_runs.filter((r) => pol.wake_workflows.some((w) => r.path === w || r.path.endsWith(`/${w}`)));
        if (waiting.length || (wf.length && !rs.some((g) => g.status === "online" && g.labels.includes(label)))) {
          const rl = waiting.flatMap((j) => j.labels).find((l) => l.startsWith(`${label}-`));
          const region = rl ? rl.slice(label.length + 1) : null;
          const up = await buildPodsUp(env, "policy:wake_on_queue", { region, needIdleRunner: waiting.length > 0 });
          if (up.action !== "reused") {
            actions.push(`build pod ${up.pod.name}: ${up.action} for ${waiting.length} queued ${label} job(s)${wf.length ? ` / ${wf.length} watched run(s)` : ""}`);
            await audit(env, { actor: "policy:wake_on_queue", action: "build_pod.wake", target: up.pod.pod_id || up.pod.id, detail: `${up.action}; jobs ${waiting.map((j) => j.job_id).join(",")}` });
          }
        }
      } catch (e) {
        const m = (e as Error).message;
        if (!(e instanceof HttpError && e.status === 409))
          alerts.push({ key: "build_pod_wake", kind: "build_pod_runner", severity: "warn", message: `wake on queued jobs: ${scrub(env, m).slice(0, 200)}` });
      }
    }
  }
  return { alerts, actions };
}

