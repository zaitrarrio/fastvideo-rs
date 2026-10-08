// fv-control: the fastvideo-rs cluster controller (docs/control/README.md).
// A Hono app on Cloudflare Workers: JSON API under /api (what the dashboard
// in public/ uses, and scripts/serve/fv-control.sh), log ingest under
// /ingest, a per-minute cron (collector.ts) and one ClusterOps Durable
// Object per cluster (cluster/do.ts).
import { Hono, type Context } from "hono";
import { DEFAULT_POLICIES, policies } from "./alerts";
import { accessMode, clientIp, login, logout, mintApiToken, requireAuth, requireConsoleAuth, whoami } from "./auth";
import { cancelOp, currentOp, startOp } from "./cluster/control";
import { adminAll, adminOne, adminTargets, adminToken, desiredEnv, edgeCfg, edgeFamilies, edgePublic, edgeWorkers, envCtx, projectSpend, workerHealth } from "./cluster/ops";
import { isDirect, workerSystemEnv, type ClusterSecrets, type ClusterState, type PodRec } from "./cluster/payloads";
import { assertRegionsAvailable, defaultSpec, isEdge, normalizeSpec, POOL_PRESETS, STANDARD_POOLS, TEMPLATES } from "./cluster/spec";
import { allPods, deleteClusterRow, emptyState, getCluster, insertCluster, isStandalone, listClusters, livePods, newSecrets, STANDALONE, type Cluster } from "./cluster/store";
import { getDiagnosis } from "./podlogs";
import { bootPhase, bootRows, getBoot } from "./boottime";
import { standaloneSpec, standaloneView, STANDALONE_POOL, type LaunchRequest } from "./standalone";
import { buildPodStatus } from "./buildpod";
import {
  buildPodsOverview,
  buildPodsPolicy,
  buildPodsUp,
  buildPodToken,
  buildPodView,
  ciBuildRunner,
  cpuCandidates,
  createBuildPod,
  currentRef,
  deleteBuildPodRow,
  getBuildPod,
  normalizePolicy,
  rankCandidates,
  registerRunner,
  serverBundle,
  startBuildPodRow,
  stopBuildPodRow,
} from "./buildpods";
import { collect } from "./collector";
import { randomToken, sha256Hex } from "./crypto";
import { defaults, type Env, type Vars } from "./env";
import { deleteVar, listVars, maskRow, resolveView, setVar, type Scope } from "./envvars";
import { ciStatus, dispatchRelease } from "./github";
import { listTags } from "./ghcr";
import { canonicalId, docHistory, DOC_KINDS, docVersion, planSpec, poolSid, readDoc, restoreDoc, saveDoc, SCHEMA_OF, validateDoc, bumpDoc, type DocKind } from "./docs";
import { dynamicEnums } from "./dynamic";
import { downloadLogs, ingest, searchLogs } from "./logs";
import { decodeCursor, decodeTail, encodeTail, lineContext, logFacets, parseLogQuery, queryLogs, tailLogs, type XLine } from "./logquery";
import { availability, checkSpec, liveSpecIssues } from "./cluster/editor";
import { checkImages } from "./cluster/preflight";
import { resolveClusterImages } from "./ghcr";
import { assertNameFree, checkName } from "./names";
import { mapLaunchIssues } from "./standalone";
import { issuesText, jsonSchemas, parseOr400, validate, type Issue, type SchemaName } from "./schemas";
import { querySeries } from "./metrics";
import { clusterDrift, registry, releaseHeads } from "./releases";
import { runpod } from "./runpod";
import { cancelJobRow, cancelQueued, findJob, jobOf, listJobs } from "./jobs";
import { serverlessRoutes } from "./serverless/routes";
import { getRow, serverlessTick } from "./serverless/ops";
import { consoleRequest, sweepUploads, uploadGet } from "./serverless/console";
import { cloudrift, cloudriftEnabled, CLOUDRIFT_OWNER_TAG } from "./cloudrift";
import { OTHER_PROVIDERS, type OtherProviderId } from "./enums";
import { endpointReport, isOtherPod, lastReport, otherPodLogs, providerImpl, providerView, splitKey } from "./providers";
import { audit, fetchWithTimeout, getSetting, HttpError, newId, now, putSetting, scrub, utcDay } from "./util";

export { ClusterOps } from "./cluster/do";

type App = { Bindings: Env; Variables: Vars };
type C = Context<App>;
const app = new Hono<App>();

app.onError((err, c) => {
  const status = err instanceof HttpError ? err.status : 500;
  const msg = scrub(c.env, err.message || "error");
  if (status >= 500) console.error("fv-control error", msg);
  return c.json({ error: msg, ...(err instanceof HttpError && err.extra ? err.extra : {}) }, status as any);
});
app.use("*", async (c, next) => {
  await next();
  c.header("x-content-type-options", "nosniff");
  c.header("referrer-policy", "no-referrer");
  if (c.req.path.startsWith("/api/")) c.header("cache-control", "no-store");
});

const body = async <T = any>(c: C): Promise<T> => {
  try {
    return (await c.req.json()) as T;
  } catch {
    return {} as T;
  }
};
const actor = (c: C) => c.get("actor");
const auditC = (c: C, a: Omit<Parameters<typeof audit>[1], "actor">) => audit(c.env, { ...a, actor: actor(c), ip: clientIp(c) });

// ---------------- public
app.get("/healthz", (c) => c.json({ ok: true, service: "fv-control", environment: c.env.ENVIRONMENT || "dev", auth: accessMode(c.env) ? "access" : "passphrase" }));
app.post("/api/auth/login", async (c) => {
  const b = await body<{ passphrase?: string }>(c);
  return c.json({ ok: true, ...(await login(c, String(b.passphrase || ""))) });
});
app.post("/api/auth/logout", async (c) => {
  await logout(c);
  return c.json({ ok: true });
});
app.get("/api/auth/me", async (c) => c.json(await whoami(c)));
app.post("/ingest/v1/logs", async (c) => c.json(await ingest(c.env, c.req.raw, c.executionCtx as ExecutionContext)));
/** A GMI / Brev pod reports its phase and tunnel URL (providers.ts PROVIDER_BOOT) with its cluster's ingest token. */
app.post("/ingest/v1/endpoint", async (c) => c.json(await endpointReport(c.env, c.req.raw)));

// ---------------- everything else under /api needs auth
app.use("/api/*", requireAuth);

// Overview: the dashboard's key numbers.
app.get("/api/overview", async (c) => {
  const env = c.env;
  const t = now();
  const day = utcDay(t);
  const last = await env.DB.prepare("SELECT * FROM balance_samples ORDER BY at DESC LIMIT 1").first<{ at: number; balance: number; spend_per_hr: number }>();
  const hourAgo = await env.DB.prepare("SELECT * FROM balance_samples WHERE at <= ? ORDER BY at DESC LIMIT 1").bind(t - 3600_000).first<{ at: number; balance: number }>();
  const floor = Number(env.BALANCE_FLOOR || "8");
  const pol = await policies(env);
  const pods = await env.DB.prepare("SELECT * FROM pods WHERE gone_at IS NULL ORDER BY desired_status DESC, cost_per_hr DESC").all<any>();
  const running = (pods.results || []).filter((p) => p.desired_status === "RUNNING");
  const costToday = await env.DB.prepare("SELECT owner, cluster_id, SUM(usd) AS usd, SUM(minutes) AS minutes, SUM(idle_minutes) AS idle_minutes FROM cost_daily WHERE day = ? GROUP BY owner, cluster_id ORDER BY usd DESC").bind(day).all<any>();
  const alerts = await env.DB.prepare("SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END, opened_at DESC").all<any>();
  const clusters = await listClusters(env);
  const burn = last?.spend_per_hr ?? 0;
  const idle = running.filter((p) => p.idle_since && t - p.idle_since >= pol.idle_min * 60_000);
  return c.json({
    at: last?.at ?? null,
    balance: last?.balance ?? null,
    burn_per_hr: burn,
    balance_change_1h: last && hourAgo ? last.balance - hourAgo.balance : null,
    floor,
    hours_to_floor: last && burn > 0 ? Math.max(0, (last.balance - floor) / burn) : null,
    running_pods: running.length,
    running_gpu_pods: running.filter((p) => (p.gpu_count || 0) > 0).length,
    idle_pods: idle.map((p) => ({ pod_id: p.pod_id, name: p.name, owner: p.owner, idle_min: Math.round((t - p.idle_since) / 60000), cost_per_hr: p.cost_per_hr })),
    idle_burn_per_hr: idle.reduce((s, p) => s + (p.cost_per_hr || 0), 0),
    cost_today: costToday.results || [],
    cost_today_total: (costToday.results || []).reduce((s: number, r: any) => s + (r.usd || 0), 0),
    alerts: alerts.results || [],
    clusters: clusters.map((cl) => ({
      id: cl.id,
      name: cl.name,
      kind: isStandalone(cl) ? "standalone" : "cluster",
      status: cl.status,
      deadline: cl.deadline,
      dph: running.filter((p) => p.cluster_id === cl.id).reduce((s, p) => s + (p.cost_per_hr || 0), 0),
      pods: running.filter((p) => p.cluster_id === cl.id).length,
      cost_today: (costToday.results || []).filter((r: any) => r.cluster_id === cl.id).reduce((s: number, r: any) => s + r.usd, 0),
    })),
    policies: pol,
    cloudrift: cloudriftEnabled(env) ? await cloudriftView(env) : null,
  });
});

// ---------------- providers (Runpod is the primary; CloudRift, docs/ops/cloudrift.md)
async function cloudriftView(env: Env) {
  const acct = await getSetting<{ at: number | null; balance: number | null; spend_per_hr: number; running: number }>(env, "cloudrift_account", { at: null, balance: null, spend_per_hr: 0, running: 0 });
  const floor = defaults.cloudriftFloor(env);
  return { ...acct, floor, hours_to_floor: acct.balance !== null && acct.spend_per_hr > 0 ? Math.max(0, (acct.balance - floor) / acct.spend_per_hr) : null };
}
app.get("/api/providers", async (c) =>
  c.json({
    providers: [
      { id: "runpod", enabled: true, launch: true },
      { id: "cloudrift", enabled: cloudriftEnabled(c.env), launch: false, ...(cloudriftEnabled(c.env) ? await cloudriftView(c.env) : {}) },
      // GMI Cloud and NVIDIA Brev (docs/serve/deploy-gmi-brev.md): launch targets for standalone pods and pools.
      ...(await Promise.all(OTHER_PROVIDERS.map(async (p) => ({ ...(await providerView(c.env, p)), launch: true })))),
    ],
  }),
);
/** Price and stock of a GMI / Brev GPU product (the planner's view). */
app.get("/api/providers/:p/offers", async (c) => {
  const p = c.req.param("p") as OtherProviderId;
  if (!(OTHER_PROVIDERS as readonly string[]).includes(p)) throw new HttpError(404, `no launch provider ${p} (${OTHER_PROVIDERS.join(", ")})`);
  const impl = providerImpl(p);
  const off = impl.off(c.env);
  if (off) throw new HttpError(400, `${impl.title} is off: ${off} (docs/serve/deploy-gmi-brev.md §8)`);
  const gpus = c.req.query("gpu") ? [c.req.query("gpu")!] : impl.gpus(c.env);
  const region = c.req.query("region") || undefined;
  return c.json({ provider: p, offers: await Promise.all(gpus.map((g) => impl.offer(c.env, g, region))) });
});
/** A GMI / Brev pod's own log as the provider keeps it (GMI: GET /v1/containers/{id}/logs; Brev: none). */
app.get("/api/pods/:id/provider-logs", async (c) => {
  const id = c.req.param("id");
  if (!isOtherPod(id)) throw new HttpError(400, "a GMI / Brev pod id (gmi:<name>, brev:<name>); Runpod pods: /api/pods/:id/runpod-logs");
  const lines = await otherPodLogs(c.env, id);
  return c.json({ source: splitKey(id)!.provider, lines: lines.slice(-1000).map((l) => scrub(c.env, l)), report: await lastReport(c.env, id) });
});
app.get("/api/providers/cloudrift/price", async (c) => {
  const gpu = c.req.query("gpu") || "";
  if (!gpu) throw new HttpError(400, "gpu: a brand (RTX PRO 6000) or a variant name");
  return c.json({ gpu, offers: await cloudrift.price(c.env, gpu) });
});
/** Terminates one of our CloudRift rentals (tag fv-owner:fastvideo-rs); others are refused. */
app.post("/api/providers/cloudrift/instances/:id/terminate", async (c) => {
  const id = c.req.param("id");
  if (!cloudriftEnabled(c.env)) throw new HttpError(400, "CloudRift is not configured (CLOUDRIFT_API_KEY)");
  const inst = (await cloudrift.instances(c.env)).find((i) => i.id === id);
  if (!inst) throw new HttpError(404, "no live CloudRift rental with that id");
  if (!inst.ours) throw new HttpError(403, `not ours: only rentals tagged ${CLOUDRIFT_OWNER_TAG} are touched`);
  const ok = await cloudrift.terminate(c.env, id);
  await auditC(c, { action: "cloudrift.terminate", target: id, after: { name: inst.name, ok } });
  return c.json({ id, terminated: ok });
});

app.get("/api/pods", async (c) => {
  const all = c.req.query("all") === "1";
  const r = await c.env.DB.prepare(`SELECT p.*, (SELECT SUM(usd) FROM cost_daily d WHERE d.pod_id = p.pod_id AND d.day = ?) AS cost_today, (SELECT SUM(usd) FROM cost_daily d WHERE d.pod_id = p.pod_id) AS cost_total FROM pods p ${all ? "" : "WHERE gone_at IS NULL"} ORDER BY desired_status DESC, cost_per_hr DESC`)
    .bind(utcDay(now()))
    .all<any>();
  return c.json({ pods: r.results || [] });
});
app.get("/api/pods/:id", async (c) => {
  const id = c.req.param("id");
  const p = await c.env.DB.prepare("SELECT * FROM pods WHERE pod_id = ?").bind(id).first<any>();
  if (!p) throw new HttpError(404, "unknown pod");
  const costs = await c.env.DB.prepare("SELECT day, usd, minutes, idle_minutes FROM cost_daily WHERE pod_id = ? ORDER BY day DESC LIMIT 30").bind(id).all<any>();
  const ctl = await c.env.DB.prepare("SELECT * FROM cluster_pods WHERE pod_id = ?").bind(id).first<any>();
  return c.json({ pod: p, costs: costs.results || [], controller: ctl || null });
});
app.get("/api/pods/:id/runpod-logs", async (c) => {
  const l = await runpod.logs(c.env, c.req.param("id"));
  const s = (x: string[]) => x.slice(-1000).map((line) => scrub(c.env, line));
  return c.json({ source: "runpod", container: s(l.container), system: s(l.system) });
});
/** A pod's logs as JSON, oldest first: what fv-serve shipped and the Runpod container / system log the controller captured from boot (podlogs.ts). ?after_id= follows; ?source=runpod|serve|control, ?level=, ?q=, ?since=, ?until=, ?limit= (≤ 2000). */
app.get("/api/pods/:id/logs", async (c) => {
  const q = c.req.query();
  const id = c.req.param("id");
  const after = q.after_id ? Number(q.after_id) : undefined;
  const lines = await searchLogs(c.env, {
    pod: id,
    text: q.q,
    level: q.level,
    source: q.source,
    since: q.since ? Number(q.since) : undefined,
    until: q.until ? Number(q.until) : undefined,
    limit: q.limit ? Number(q.limit) : undefined,
    after_id: after,
  });
  return c.json({ pod: id, lines, next_after_id: lines.length ? lines[lines.length - 1].id : after ?? 0 });
});
/** A controller pod's boot timeline (boottime.ts): each phase's time since create and duration, the weight components, the phase it is in. */
app.get("/api/pods/:id/boot", async (c) => {
  const bt = await getBoot(c.env, c.req.param("id"));
  if (!bt) throw new HttpError(404, "not a controller pod");
  return c.json({ pod_id: c.req.param("id"), phase: bt.t.ready ? null : bootPhase(bt), timeline: bootRows(bt), boot: bt });
});
/** A pod's status as JSON: the account view (Runpod state, $/hr, health, utilisation), the controller's record, its boot diagnosis and its cost. */
app.get("/api/pods/:id/status", async (c) => {
  const id = c.req.param("id");
  const [p, ctl, cost, boot] = await Promise.all([
    c.env.DB.prepare("SELECT * FROM pods WHERE pod_id = ?").bind(id).first<any>(),
    c.env.DB.prepare("SELECT cp.*, cl.name AS cluster_name, cl.source AS cluster_source FROM cluster_pods cp LEFT JOIN clusters cl ON cl.id = cp.cluster_id WHERE cp.pod_id = ?").bind(id).first<any>(),
    c.env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS total, COALESCE(SUM(CASE WHEN day = ? THEN usd ELSE 0 END), 0) AS today, COALESCE(SUM(minutes), 0) AS minutes FROM cost_daily WHERE pod_id = ?").bind(utcDay(now()), id).first<any>(),
    getDiagnosis(c.env, id),
  ]);
  if (!p && !ctl) throw new HttpError(404, "unknown pod");
  const bt = await getBoot(c.env, id);
  const lines = await c.env.DB.prepare("SELECT COUNT(*) AS n, MAX(ts) AS last FROM log_lines WHERE pod_id = ?").bind(id).first<{ n: number; last: number | null }>();
  return c.json({
    pod_id: id,
    kind: ctl ? (ctl.cluster_source === STANDALONE ? "standalone" : "cluster") : "external",
    owner: p?.owner ?? null,
    runpod: p ? { desired_status: p.desired_status, cost_per_hr: p.cost_per_hr, gpu: p.gpu, dc: p.dc, uptime_s: p.uptime_s, gpu_util: p.gpu_util, cpu: p.cpu, mem: p.mem, health: p.health, build_sha: p.build_sha, last_seen: p.last_seen, gone_at: p.gone_at } : null,
    controller: ctl ? { cluster_id: ctl.cluster_id, cluster: ctl.cluster_name, pool: ctl.pool, status: ctl.status, image: ctl.image, url: ctl.url, created_at: ctl.created_at, ready_at: ctl.ready_at, deleted_at: ctl.deleted_at } : null,
    boot,
    boot_phase: bt && !bt.t.ready ? bootPhase(bt) : null,
    boot_timeline: bootRows(bt),
    cost,
    logs: { lines: lines?.n ?? 0, last_ts: lines?.last ?? null },
  });
});
app.get("/api/metrics/series", async (c) => {
  const hours = Math.min(Math.max(Number(c.req.query("hours") || 6), 1), 24 * 7);
  return c.json(await querySeries(c.env, hours, c.req.query("pod") || undefined));
});
app.get("/api/balance", async (c) => {
  const hours = Math.min(Math.max(Number(c.req.query("hours") || 24), 1), 24 * 30);
  const r = await c.env.DB.prepare("SELECT at, balance, spend_per_hr FROM balance_samples WHERE at > ? ORDER BY at").bind(now() - hours * 3600_000).all<any>();
  const pts = r.results || [];
  const step = Math.max(1, Math.floor(pts.length / 500));
  return c.json({ points: pts.filter((_, i) => i % step === 0 || i === pts.length - 1) });
});
app.get("/api/costs", async (c) => {
  const days = Math.min(Math.max(Number(c.req.query("days") || 7), 1), 90);
  const since = utcDay(now() - (days - 1) * 86400_000);
  const q = (sql: string) => c.env.DB.prepare(sql).bind(since).all<any>().then((r) => r.results || []);
  const [byDay, byOwner, byCluster, byPod] = await Promise.all([
    q("SELECT day, SUM(usd) AS usd, SUM(idle_minutes) AS idle_minutes, SUM(minutes) AS minutes FROM cost_daily WHERE day >= ? GROUP BY day ORDER BY day"),
    q("SELECT owner, SUM(usd) AS usd, SUM(minutes) AS minutes, SUM(idle_minutes) AS idle_minutes FROM cost_daily WHERE day >= ? GROUP BY owner ORDER BY usd DESC"),
    q("SELECT d.cluster_id, c.name, SUM(d.usd) AS usd, SUM(d.minutes) AS minutes FROM cost_daily d LEFT JOIN clusters c ON c.id = d.cluster_id WHERE day >= ? AND d.cluster_id IS NOT NULL GROUP BY d.cluster_id ORDER BY usd DESC"),
    q("SELECT d.pod_id, p.name, d.owner, SUM(d.usd) AS usd, SUM(d.minutes) AS minutes, SUM(d.idle_minutes) AS idle_minutes FROM cost_daily d LEFT JOIN pods p ON p.pod_id = d.pod_id WHERE day >= ? GROUP BY d.pod_id ORDER BY usd DESC LIMIT 100"),
  ]);
  const byDayOwner = await q("SELECT day, owner, SUM(usd) AS usd FROM cost_daily WHERE day >= ? GROUP BY day, owner ORDER BY day");
  return c.json({ since, by_day: byDay, by_owner: byOwner, by_cluster: byCluster, by_pod: byPod, by_day_owner: byDayOwner });
});

// ---------------- clusters
function clusterView(cl: Cluster) {
  return { id: cl.id, name: cl.name, kind: isStandalone(cl) ? "standalone" : "cluster", status: cl.status, deadline: cl.deadline, source: cl.source, created_at: cl.created_at, updated_at: cl.updated_at, created_by: cl.created_by, spec: cl.spec, state: cl.state };
}
app.get("/api/templates", (c) =>
  c.json({
    // Every template as a spec (the keys standard / tiny-cpu as before), their titles, and the pool presets the dashboard can add.
    ...Object.fromEntries(Object.keys(TEMPLATES).map((k) => [k, defaultSpec("example", k)])),
    templates: Object.entries(TEMPLATES).map(([id, t]) => ({ id, title: t.title, pools: t.pools })),
    pool_presets: POOL_PRESETS,
    standard_pools: STANDARD_POOLS,
  }),
);
app.get("/api/clusters", async (c) => {
  // Standalone pods are listed under /api/standalone (?all=1: here as well).
  const cls = (await listClusters(c.env)).filter((cl) => c.req.query("all") === "1" || !isStandalone(cl));
  const out = [];
  for (const cl of cls) out.push({ ...clusterView(cl), op: await currentOp(c.env, cl.id).catch(() => null) });
  return c.json({ clusters: out });
});
app.post("/api/clusters", async (c) => {
  const b = await body(c);
  const spec = normalizeSpec(b.spec || b);
  await assertNameFree(c.env, "cluster", spec.name);
  const live = await liveSpecIssues(c.env, spec);
  if (live.issues.length) throw new HttpError(400, issuesText(live.issues, "spec"), { issues: live.issues });
  const { s, ingestHash } = await newSecrets();
  const cl = await insertCluster(c.env, spec, emptyState(), s, ingestHash, actor(c), "controller", "defined", null);
  await auditC(c, { action: "cluster.define", target: cl.name, after: spec });
  return c.json({ cluster: clusterView(cl) }, 201);
});
app.get("/api/clusters/:id", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const pods = await livePods(c.env, cl.id);
  const ops = await c.env.DB.prepare("SELECT id, kind, status, params, error, actor, created_at, updated_at FROM operations WHERE cluster_id = ? ORDER BY created_at DESC LIMIT 20").bind(cl.id).all<any>();
  const heads = await releaseHeads(c.env);
  const live = await c.env.DB.prepare("SELECT * FROM pods WHERE cluster_id = ? AND gone_at IS NULL").bind(cl.id).all<any>();
  const edge_url = isEdge(cl.spec) ? edgeCfg(c.env)?.url || null : undefined;
  return c.json({ cluster: clusterView(cl), ...(edge_url !== undefined ? { edge_url } : {}), pods, live: live.results || [], ops: ops.results || [], op: await currentOp(c.env, cl.id).catch(() => null), drift: clusterDrift(cl, heads.heads) });
});
app.put("/api/clusters/:id/spec", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const b = await body(c);
  const doc = normalizeSpec({ ...(b.spec || b), name: cl.name });
  await saveDoc(c.env, "cluster-spec", cl.id, doc, { version: b.version, actor: actor(c), ip: clientIp(c) });
  return c.json({ cluster: clusterView(await getCluster(c.env, cl.id)) });
});
app.delete("/api/clusters/:id", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  if (allPods(cl.state).length) throw new HttpError(409, "the cluster has pods: stop it first");
  await deleteClusterRow(c.env, cl.id);
  await auditC(c, { action: "cluster.delete", target: cl.name, before: cl.spec });
  return c.json({ deleted: cl.id });
});
// Every problem of a draft spec with its field path (the configuration editor; nothing is saved).
// `id`: an existing cluster (its name is fixed); none: a new one (its name must be free).
app.post("/api/clusters/validate", async (c) => {
  const b = await body(c);
  const cl = b.id ? await getCluster(c.env, String(b.id)) : null;
  return c.json(await checkSpec(c.env, b.spec, cl ? { id: cl.id, name: cl.name } : undefined));
});
// The price check of a cluster's saved spec, or of a draft (`spec`; id `new` for one not defined yet),
// with Runpod's stock per pool (`availability`, null when Runpod did not answer).
app.post("/api/clusters/:id/price", async (c) => {
  const id = c.req.param("id");
  const b = await body(c);
  const cl = id === "new" ? null : await getCluster(c.env, id);
  if (!cl && !b.spec) throw new HttpError(400, "spec: the draft to price");
  const spec = b.spec ? normalizeSpec(cl ? { ...b.spec, name: cl.name } : b.spec) : cl!.spec;
  const [price, avail] = await Promise.all([
    projectSpend(c.env, spec, { hours: Number(b.hours) > 0 ? Number(b.hours) : spec.cap_s / 3600 }),
    b.availability === false ? Promise.resolve(null) : availability(c.env, spec).catch(() => null),
  ]);
  return c.json({ ...price, availability: avail });
});

const OPS: Record<string, { kind: Parameters<typeof startOp>[2]; params: (b: any) => any }> = {
  start: { kind: "up", params: (b) => ({ skip_price_check: false, ...(b.confirm_over_floor ? {} : {}) }) },
  stop: { kind: "down", params: () => ({ reason: "stop" }) },
  extend: { kind: "extend", params: (b) => parseOr400("extend", { minutes: b.minutes }) },
  scale: { kind: "scale", params: (b) => parseOr400("scale", { pool: b.pool, count: b.count }) },
  roll: { kind: "roll", params: (b) => parseOr400("roll", { target: b.target ?? "stable", ...(Array.isArray(b.pools) ? { pools: b.pools } : {}) }) },
  // pods / pools: restart just those, whether or not their env changed; neither: every pod whose env changed.
  restart: { kind: "restart", params: (b) => ({ pods: Array.isArray(b.pods) && b.pods.length ? b.pods.map(String) : undefined, pools: Array.isArray(b.pools) && b.pools.length ? b.pools.map(String) : undefined }) },
};
for (const [path, def] of Object.entries(OPS)) {
  app.post(`/api/clusters/:id/${path}`, async (c) => {
    const cl = await getCluster(c.env, c.req.param("id"));
    const params = def.params(await body(c));
    // Ops that create pods refuse a stored spec that still names an unavailable region (us: no weights volume).
    if (["up", "scale", "roll", "restart"].includes(def.kind)) assertRegionsAvailable(cl.spec);
    for (const p of def.kind === "scale" ? [params.pool] : def.kind === "roll" ? params.pools || [] : [])
      if (!cl.spec.pools.some((x) => x.id === p)) throw new HttpError(400, `pool: ${cl.name} has no pool ${p} (${cl.spec.pools.map((x) => x.id).join(", ")})`, { issues: [{ path: ["pool"], message: `no pool ${p}` }] });
    if (def.kind === "scale" && isStandalone(cl) && params.count > 1) throw new HttpError(400, `${cl.name} is a standalone pod: one pod (launch another for more)`);
    const r = await startOp(c.env, cl.id, def.kind, params, actor(c));
    await auditC(c, { action: `cluster.${def.kind}`, target: cl.name, before: { status: cl.status, deadline: cl.deadline }, after: params });
    return c.json({ operation: r.id, kind: def.kind }, 202);
  });
}
app.post("/api/clusters/:id/cancel", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const r = await cancelOp(c.env, cl.id, actor(c));
  await auditC(c, { action: "cluster.cancel", target: cl.name, after: r });
  return c.json(r);
});
app.get("/api/clusters/:id/ops", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const r = await c.env.DB.prepare("SELECT * FROM operations WHERE cluster_id = ? ORDER BY created_at DESC LIMIT 50").bind(cl.id).all<any>();
  return c.json({ operations: (r.results || []).map((o) => ({ ...o, log: JSON.parse(o.log || "[]") })) });
});
app.get("/api/ops/:id", async (c) => {
  const o = await c.env.DB.prepare("SELECT * FROM operations WHERE id = ?").bind(c.req.param("id")).first<any>();
  if (!o) throw new HttpError(404, "no such operation");
  return c.json({ operation: { ...o, log: JSON.parse(o.log || "[]"), params: JSON.parse(o.params || "{}") } });
});

/** The effective env of every pod of a cluster (masked), and which pods need a restart to get it. */
app.get("/api/clusters/:id/env", async (c) => {
  const env = c.env;
  const cl = await getCluster(env, c.req.param("id"));
  const ctx = await envCtx(env, cl);
  const rows = await livePods(env, cl.id);
  const pods = [];
  for (const r of rows.filter((x: any) => x.slot !== "retired" && x.role === "worker")) {
    const des = await desiredEnv(env, cl, ctx, "worker", { pod: r.pod_id, pool: r.pool, image: r.image });
    pods.push({ pod_id: r.pod_id, role: r.role, pool: r.pool, needs_restart: des.hash !== r.env_hash, env_applied_at: r.env_applied_at, env: await resolveView(env, cl.id, r.pod_id, des.system, r.role === "worker" ? r.pool : null) });
  }
  // What a new pod of each kind would get (also when the cluster is stopped).
  const preview: Record<string, unknown> = {};
  const img = (k: string) => cl.state.images[k] || cl.state.image || `(${k} image)`;
  for (const p of cl.spec.pools) preview[p.id] = await resolveView(env, cl.id, null, workerSystemEnv(ctx, p, img(p.id)), p.id);
  return c.json({ pods, preview, needs_restart: pods.filter((p) => p.needs_restart).map((p) => p.pod_id) });
});
/** Where clients go: the edge, or each worker of a direct cluster (docs/control/gateway-less-auth.md). */
function clientUrls(cl: Cluster, env?: Env) {
  const workers = Object.entries(cl.state.workers || {}).flatMap(([pool, recs]) => recs.map((r) => ({ pod: r.pod, pool, url: r.url || null })));
  const edge = isEdge(cl.spec) && env ? edgeCfg(env)?.url || null : null;
  return { direct: isDirect(cl.spec, cl.state), ...(isEdge(cl.spec) ? { edge_url: edge } : {}), workers };
}
/** The cluster's front: the edge's status and families view, or each direct worker's /health. (`/gateway` is the path older dashboards call.) */
async function frontView(c: any) {
  const cl = await getCluster(c.env, c.req.param("id"));
  if (isEdge(cl.spec)) {
    // The edge is the front: its public status and its families view, with each pod's place in it.
    const url = edgeCfg(c.env)?.url || null;
    const status = await edgePublic(c.env, cl, "/fv/v1/status").catch((e: Error) => ({ status: 0, body: { error: e.message } }));
    const families = await edgeFamilies(c.env).catch((e: Error) => ({ error: e.message }));
    const fronts = edgeWorkers(families);
    const workers = clientUrls(cl, c.env).workers.map((w) => ({ ...w, front: fronts.get(w.pod) || null }));
    return c.json({ url, edge: true, status: status.body, families, workers });
  }
  // Direct: each worker's public /health stands in for the status view.
  const u = clientUrls(cl);
  const workers = await Promise.all(u.workers.map(async (w) => ({ ...w, health: await workerHealth(c.env, w.pod) })));
  return c.json({ url: null, direct: true, workers });
}
// ---------------- jobs of clusters and standalone pods (src/jobs.ts; docs/control/README.md "Jobs")
app.get("/api/clusters/:id/jobs", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const q = parseOr400("jobs-query", Object.fromEntries(Object.entries(c.req.query()).filter(([, v]) => v !== "")));
  return c.json(await listJobs(c.env, cl, q));
});
app.post("/api/clusters/:id/jobs/cancel-queued", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const x = parseOr400("jobs-cancel-queued", await body(c));
  return c.json(await cancelQueued(c.env, { actor: actor(c), ip: clientIp(c) }, cl, x));
});
/** One job by any of its ids (the API's own, or fv-serve's internal uuid); the full job record never leaves fv-control. */
app.get("/api/jobs/:job", async (c) => {
  const x = parseOr400("job-cancel", { job: c.req.param("job"), ...(c.req.query("cluster") ? { cluster: c.req.query("cluster") } : {}) });
  const { c: cl, row } = await findJob(c.env, x.job, x.cluster);
  return c.json({ cluster: { id: cl.id, name: cl.name }, job: await jobOf(c.env, cl, row) });
});
app.post("/api/jobs/:job/cancel", async (c) => {
  const x = parseOr400("job-cancel", { ...(await body(c)), job: c.req.param("job") });
  const { c: cl, row } = await findJob(c.env, x.job, x.cluster);
  const r = await cancelJobRow(c.env, { actor: actor(c), ip: clientIp(c) }, cl, row);
  if (!r.ok) throw new HttpError(502, `cancel ${r.external_id} (${r.api}, ${cl.name}): ${r.note}`, { result: r });
  return c.json(r);
});
app.get("/api/clusters/:id/front", frontView);
app.get("/api/clusters/:id/gateway", frontView);
app.post("/api/clusters/:id/admin-token", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const tok = await adminToken(c.env, cl);
  await auditC(c, { action: "cluster.admin-token.reveal", target: cl.name });
  const u = clientUrls(cl, c.env);
  const base = u.edge_url ?? u.workers.find((w) => w.url)?.url;
  return c.json({ admin_token: tok, console: base ? `${base}/console/admin` : null, ...u });
});
app.post("/api/clusters/:id/mint-key", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const name = parseOr400("mint-key", { name: (await body(c)).name ?? "fv-control" }).name;
  const r = await adminOne(c.env, cl, "POST", "/fv/v1/admin/keys", { name });
  if (r.status !== 201 || !r.body?.api_key) throw new HttpError(502, `mint refused (${r.status})`);
  await auditC(c, { action: "cluster.mint-key", target: cl.name, after: { key: r.body.key?.id, name } });
  // Gateway-less: the other workers load the key from D1 within 30 s.
  return c.json({ api_key: r.body.api_key, key: r.body.key, ...(isDirect(cl.spec, cl.state) ? { minted_on: r.pod, propagation_s: 30 } : {}) }, 201);
});
app.get("/api/clusters/:id/keys", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const r = await adminOne(c.env, cl, "GET", "/fv/v1/admin/keys");
  if (r.status !== 200) throw new HttpError(502, `key list refused (${r.status})`);
  return c.json({ keys: r.body?.keys || [], backend: r.body?.backend });
});
app.delete("/api/clusters/:id/keys/:kid", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const kid = c.req.param("kid");
  if (!/^key_[0-9a-f]{12}$/.test(kid)) throw new HttpError(400, "key id: key_<12 hex>");
  adminTargets(cl, c.env);
  // Every worker of a gateway-less cluster at once; one that misses it reads the revocation from D1 within 30 s.
  const rs = await adminAll(c.env, cl, "DELETE", `/fv/v1/admin/keys/${kid}`);
  const ok = rs.filter((r) => r.status === 200);
  await auditC(c, { action: "cluster.revoke-key", target: cl.name, after: { key: kid, applied: ok.map((r) => r.pod), failed: rs.filter((r) => r.status !== 200).map((r) => `${r.pod}:${r.status}`) } });
  if (!ok.length) throw new HttpError(rs.every((r) => r.status === 404) ? 404 : 502, `revoke refused (${rs.map((r) => r.status).join(",")})`);
  return c.json({ key: ok[0]!.body?.key, applied: ok.map((r) => r.pod), failed: rs.filter((r) => r.status !== 200).map((r) => ({ pod: r.pod, status: r.status })) });
});


// ---------------- standalone pods (standalone.ts, docs/control/standalone-pods.md)
async function standaloneOf(c: C): Promise<Cluster> {
  const cl = await getCluster(c.env, c.req.param("id")!);
  if (!isStandalone(cl)) throw new HttpError(404, `${cl.name} is a cluster, not a standalone pod (/api/clusters)`);
  return cl;
}
const standaloneOut = async (c: C, cl: Cluster) => standaloneView(c.env, cl, await currentOp(c.env, cl.id).catch(() => null));
app.get("/api/standalone", async (c) => {
  const pods = [];
  for (const cl of (await listClusters(c.env)).filter(isStandalone)) pods.push(await standaloneOut(c, cl));
  return c.json({ pods });
});
/** Every problem of a draft launch request, path-anchored in the request's own fields (the launch form; nothing is saved). */
app.post("/api/standalone/validate", async (c) => {
  const b = await body<any>(c);
  const issues: Issue[] = [];
  let spec = null;
  try {
    spec = standaloneSpec(b).spec;
  } catch (e) {
    const ex = e as HttpError;
    issues.push(...mapLaunchIssues((ex.extra?.issues as Issue[]) || [{ path: [], message: ex.message }]));
  }
  if (typeof b?.name === "string" && !issues.some((i) => i.path[0] === "name")) {
    const r = await checkName(c.env, "cluster", b.name);
    if (!r.ok && r.problem) issues.push({ path: ["name"], message: r.problem });
  }
  const warnings: Issue[] = [];
  if (spec) {
    const live = await liveSpecIssues(c.env, spec);
    issues.push(...mapLaunchIssues(live.issues));
    warnings.push(...mapLaunchIssues(live.warnings));
  }
  return c.json({ ok: !issues.length, issues, warnings, spec: issues.length ? null : spec });
});
/** Launch: define the pod and start it (start: false only defines it). The same `up` as a cluster: price check, image preflight, placement, wait until ready. */
app.post("/api/standalone", async (c) => {
  requireAdmin(c);
  const b = await body<LaunchRequest & { start?: boolean; skip_image_check?: boolean }>(c);
  const { spec, env: vars } = standaloneSpec(b);
  await assertNameFree(c.env, "cluster", spec.name);
  const live = await liveSpecIssues(c.env, spec);
  if (live.issues.length) {
    const issues = mapLaunchIssues(live.issues);
    throw new HttpError(400, issuesText(issues), { issues });
  }
  const { s, ingestHash } = await newSecrets();
  const cl = await insertCluster(c.env, spec, emptyState(), s, ingestHash, actor(c), STANDALONE, "defined", null);
  for (const v of vars) await setVar(c.env, "cluster", cl.id, v.key, v.value, v.secret, actor(c));
  await auditC(c, { action: "standalone.launch", target: cl.name, after: { spec, env: vars.map((v) => ({ key: v.key, secret: v.secret, value: v.secret ? "••••••••" : v.value })) } });
  let operation: string | null = null;
  if (b.start !== false) {
    operation = (await startOp(c.env, cl.id, "up", { skip_price_check: false, skip_image_check: !!b.skip_image_check }, actor(c))).id;
    await auditC(c, { action: "standalone.start", target: cl.name });
  }
  return c.json({ pod: await standaloneOut(c, await getCluster(c.env, cl.id)), operation }, 201);
});
app.get("/api/standalone/:id", async (c) => c.json({ pod: await standaloneOut(c, await standaloneOf(c)) }));
app.post("/api/standalone/:id/start", async (c) => {
  requireAdmin(c);
  const cl = await standaloneOf(c);
  if ((cl.state.workers[STANDALONE_POOL] || []).length) throw new HttpError(409, `${cl.name} has a pod already`);
  assertRegionsAvailable(cl.spec);
  const b = await body<{ skip_image_check?: boolean }>(c);
  const r = await startOp(c.env, cl.id, "up", { skip_price_check: false, skip_image_check: !!b.skip_image_check }, actor(c));
  await auditC(c, { action: "standalone.start", target: cl.name });
  return c.json({ operation: r.id, kind: "up" }, 202);
});
/** Stop: the pod is deleted (the GPU and its cost released; logs and costs kept); the definition stays, and start makes a new pod. */
app.post("/api/standalone/:id/stop", async (c) => {
  requireAdmin(c);
  const cl = await standaloneOf(c);
  const cur = (await currentOp(c.env, cl.id).catch(() => null)) as { kind?: string } | null;
  if (cur && cur.kind !== "down") await cancelOp(c.env, cl.id, actor(c));
  const r = await startOp(c.env, cl.id, "down", { reason: "stop" }, actor(c));
  await auditC(c, { action: "standalone.stop", target: cl.name, before: { status: cl.status, deadline: cl.deadline } });
  return c.json({ operation: r.id, kind: "down" }, 202);
});
app.post("/api/standalone/:id/extend", async (c) => {
  requireAdmin(c);
  const cl = await standaloneOf(c);
  const { minutes } = parseOr400("extend", { minutes: (await body(c)).minutes });
  const r = await startOp(c.env, cl.id, "extend", { minutes }, actor(c));
  await auditC(c, { action: "standalone.extend", target: cl.name, after: { minutes } });
  return c.json({ operation: r.id, kind: "extend" }, 202);
});
/** Delete: stop the pod if there is one (the definition goes when it is gone: 202), else delete the definition now. */
app.delete("/api/standalone/:id", async (c) => {
  requireAdmin(c);
  const cl = await standaloneOf(c);
  if (allPods(cl.state).length || ["starting", "stopping"].includes(cl.status)) {
    const cur = (await currentOp(c.env, cl.id).catch(() => null)) as { kind?: string } | null;
    if (cur) await cancelOp(c.env, cl.id, actor(c));
    const r = await startOp(c.env, cl.id, "down", { reason: "delete", delete_definition: true }, actor(c));
    await auditC(c, { action: "standalone.delete", target: cl.name, before: cl.spec, detail: "stopping first" });
    return c.json({ operation: r.id, kind: "down", deleting: cl.id }, 202);
  }
  await deleteClusterRow(c.env, cl.id);
  await auditC(c, { action: "standalone.delete", target: cl.name, before: cl.spec });
  return c.json({ deleted: cl.id });
});

// ---------------- env vars
const scopeOf = (s: string): Scope => {
  if (s !== "account" && s !== "cluster" && s !== "pool" && s !== "pod") throw new HttpError(400, "scope: account | cluster | pool | pod");
  return s;
};
app.get("/api/env/account", async (c) => c.json({ vars: (await listVars(c.env, "account", "")).map(maskRow) }));
app.get("/api/env/:scope/:sid", async (c) => {
  const scope = scopeOf(c.req.param("scope"));
  const sid = scope === "account" ? "" : scope === "pool" ? await poolSid(c.env, c.req.param("sid")) : c.req.param("sid");
  return c.json({ vars: (await listVars(c.env, scope, sid)).map(maskRow) });
});
async function putVar(c: C, scope: Scope, sid: string, key: string) {
  const b = await body<{ value?: string; secret?: boolean }>(c);
  if (scope === "cluster") await getCluster(c.env, sid);
  if (scope === "pool") sid = await poolSid(c.env, sid);
  const r = await setVar(c.env, scope, sid, key, String(b.value ?? ""), !!b.secret, actor(c));
  await bumpDoc(c.env, "env", scope === "account" ? "account" : `${scope}:${sid}`, actor(c));
  await auditC(c, { action: "env.set", target: `${scope}:${sid || "-"}:${key}`, before: r.before, after: r.after });
  return c.json({ ok: true, ...r.after, key });
}
async function delVar(c: C, scope: Scope, sid: string, key: string) {
  if (scope === "pool") sid = await poolSid(c.env, sid);
  const r = await deleteVar(c.env, scope, sid, key);
  await bumpDoc(c.env, "env", scope === "account" ? "account" : `${scope}:${sid}`, actor(c));
  await auditC(c, { action: "env.delete", target: `${scope}:${sid || "-"}:${key}`, before: r.before });
  return c.json({ ok: true });
}
app.put("/api/env/account/:key", (c) => putVar(c, "account", "", c.req.param("key")));
app.delete("/api/env/account/:key", (c) => delVar(c, "account", "", c.req.param("key")));
app.put("/api/env/:scope/:sid/:key", (c) => putVar(c, scopeOf(c.req.param("scope")), c.req.param("sid"), c.req.param("key")));
app.delete("/api/env/:scope/:sid/:key", (c) => delVar(c, scopeOf(c.req.param("scope")), c.req.param("sid"), c.req.param("key")));

// ---------------- the shared build pod (read only: buildpod.ts, its /healthz timers)
app.get("/api/buildpod", async (c) => {
  const pol = await policies(c.env);
  const pods = await buildPodStatus(c.env, pol);
  // Its self-stop error carries a Runpod reply: scrubbed like every other upstream text.
  for (const p of pods) if (p.health?.self_stop?.error) p.health.self_stop.error = scrub(c.env, p.health.self_stop.error);
  return c.json({ pods, policy: { backstop: pol.build_pod_backstop, max_h: pol.build_pod_max_h, idle_grace_min: pol.build_pod_idle_grace_min } });
});

// ---------------- build pods managed by fv-control (buildpods.ts, docs/dev/build-pods-fv-control.md)
const requireAdmin = (c: C) => {
  if (c.get("scope") !== "admin") throw new HttpError(403, "needs an admin token");
};
const podOut = async (c: C, id: string, withToken: boolean) => {
  const row = await getBuildPod(c.env, id);
  const view = await buildPodView(c.env, row, await currentRef(c.env));
  if (!withToken) return { pod: view };
  requireAdmin(c);
  await auditC(c, { action: "build_pod.token", target: row.pod_id || row.id, detail: row.name });
  return { pod: view, token: await buildPodToken(c.env, row) };
};
app.get("/api/build-pods", async (c) => c.json(await buildPodsOverview(c.env)));
app.get("/api/build-pods/policy", async (c) => c.json({ policy: await buildPodsPolicy(c.env), defaults: normalizePolicy({}) }));
app.put("/api/build-pods/policy", async (c) => {
  const before = await buildPodsPolicy(c.env);
  const b = await body<{ policy?: Record<string, unknown> }>(c);
  const merged = { ...before, ...(b.policy || {}) };
  parseOr400("build-pods-policy", merged);
  const next = normalizePolicy(merged as any);
  await putSetting(c.env, "build_pods", next, actor(c));
  await auditC(c, { action: "build_pods.policy", before, after: next });
  return c.json({ policy: next });
});
app.get("/api/build-pods/plan", async (c) => {
  const pol = await buildPodsPolicy(c.env);
  const region = c.req.query("region") || null;
  const ranked = rankCandidates(await cpuCandidates(c.env, pol, region), pol, region);
  const bundle = await serverBundle(c.env, pol).catch((e) => ({ error: (e as Error).message }));
  return c.json({ candidates: ranked.slice(0, 15), server: "error" in bundle ? bundle : { ref: bundle.ref, sha: bundle.sha, image: bundle.image, bytes_b64: bundle.b64.length } });
});
app.post("/api/build-pods/up", async (c) => {
  requireAdmin(c);
  const b = await body<{ region?: string }>(c);
  const region = typeof b.region === "string" && /^[A-Za-z0-9-]{2,20}$/.test(b.region) ? b.region : null;
  const up = await buildPodsUp(c.env, actor(c), { region });
  return c.json({ action: up.action, replaced: up.replaced || [], ...(await podOut(c, up.pod.id, true)) }, up.action === "created" ? 201 : 200);
});
app.post("/api/build-pods", async (c) => {
  requireAdmin(c);
  const b = await body<{ region?: string; purpose?: string; server_ref?: string }>(c);
  const pol = await buildPodsPolicy(c.env);
  const purpose = b.purpose === "shared" ? "shared" : "test";
  const row = await createBuildPod(c.env, actor(c), pol, { region: b.region || null, purpose, server_ref: b.server_ref });
  return c.json(await podOut(c, row.id, true), 201);
});
app.get("/api/build-pods/:id", async (c) => c.json(await podOut(c, c.req.param("id"), false)));
app.get("/api/build-pods/:id/token", async (c) => c.json(await podOut(c, c.req.param("id"), true)));
app.post("/api/build-pods/:id/start", async (c) => {
  await startBuildPodRow(c.env, await getBuildPod(c.env, c.req.param("id")), actor(c));
  return c.json(await podOut(c, c.req.param("id"), false));
});
app.post("/api/build-pods/:id/stop", async (c) => {
  const b = await body<{ force?: boolean }>(c);
  const result = await stopBuildPodRow(c.env, await getBuildPod(c.env, c.req.param("id")), actor(c), { force: !!b.force, reason: "by hand" });
  return c.json({ result, ...(await podOut(c, c.req.param("id"), false)) });
});
app.delete("/api/build-pods/:id", async (c) => {
  await deleteBuildPodRow(c.env, await getBuildPod(c.env, c.req.param("id")), actor(c), { force: c.req.query("force") === "1" });
  return c.json(await podOut(c, c.req.param("id"), false));
});
app.post("/api/build-pods/:id/runner", async (c) => {
  const row = await getBuildPod(c.env, c.req.param("id"));
  const state = await registerRunner(c.env, row);
  await auditC(c, { action: "build_pod.runner", target: row.pod_id || row.id, detail: state });
  return c.json({ runner: state });
});
// CI (docs §6): an idle runner now, a pod being woken ("wait": poll again), or GitHub-hosted. Scope `ci` tokens reach only /api/ci/*.
app.post("/api/ci/build-runner", async (c) => {
  const b = await body<{ label?: string; region?: string; wake?: boolean; workflow?: string; run_id?: number | string }>(c);
  const region = typeof b.region === "string" && /^[a-z]{2}$/.test(b.region) ? b.region : null;
  const ans = await ciBuildRunner(c.env, actor(c), { label: b.label, region, wake: b.wake });
  if (ans.builder === "wait" && ans.pod) await auditC(c, { action: "build_pod.ci_wake", target: ans.pod.id, detail: `${String(b.workflow || "").slice(0, 80)} run ${String(b.run_id || "").slice(0, 20)}: ${ans.reason}` });
  return c.json(ans);
});

// ---------------- alerts, policies
app.get("/api/alerts", async (c) => {
  const open = c.req.query("open") !== "0";
  const r = await c.env.DB.prepare(`SELECT * FROM alerts ${open ? "WHERE resolved_at IS NULL" : ""} ORDER BY opened_at DESC LIMIT 200`).all<any>();
  return c.json({ alerts: r.results || [] });
});
app.post("/api/alerts/:id/resolve", async (c) => {
  await c.env.DB.prepare("UPDATE alerts SET resolved_at = ? WHERE id = ? AND resolved_at IS NULL").bind(now(), Number(c.req.param("id"))).run();
  await auditC(c, { action: "alert.resolve", target: c.req.param("id") });
  return c.json({ ok: true });
});
app.get("/api/policies", async (c) => c.json({ policies: await policies(c.env), defaults: DEFAULT_POLICIES }));
app.put("/api/policies", async (c) => {
  const b = await body<any>(c);
  const next = { ...(await policies(c.env)), ...b };
  delete next.version;
  const r = await saveDoc(c.env, "policies", "default", next, { version: b.version, actor: actor(c), ip: clientIp(c) });
  return c.json({ policies: r.doc, version: r.version });
});

// ---------------- logs
app.get("/api/logs", async (c) => {
  const q = c.req.query();
  return c.json({
    lines: await searchLogs(c.env, {
      pod: q.pod,
      cluster: q.cluster,
      text: q.q,
      level: q.level,
      since: q.since ? Number(q.since) : undefined,
      until: q.until ? Number(q.until) : undefined,
      limit: q.limit ? Number(q.limit) : undefined,
      after_id: q.after_id ? Number(q.after_id) : undefined,
      source: q.source,
    }),
  });
});
// The log explorer (src/logquery.ts): every source in one line shape, filtered server-side, keyset pages.
app.get("/api/logs/query", async (c) => c.json(await queryLogs(c.env, parseLogQuery(c.req.query()))));
// New lines since `tail` (the state the previous call returned; none: start now).
app.get("/api/logs/live", async (c) => {
  const r = await tailLogs(c.env, parseLogQuery(c.req.query()), decodeTail(c.req.query("tail")));
  return c.json({ lines: r.lines, tail: encodeTail(r.state) });
});
app.get("/api/logs/context", async (c) => {
  const uid = c.req.query("uid") || "";
  if (!/^[a-z]:[A-Za-z0-9_:.-]{1,120}$/.test(uid)) throw new HttpError(400, "uid: a line's uid");
  return c.json(await lineContext(c.env, uid, Number(c.req.query("before") ?? 10), Number(c.req.query("after") ?? 10)));
});
app.get("/api/logs/facets", async (c) => c.json(await logFacets(c.env, parseLogQuery(c.req.query()), Number(c.req.query("buckets") || 60))));
// The filtered result as NDJSON or text, paged server-side (at most 50 000 lines).
app.get("/api/logs/export", async (c) => {
  const q = parseLogQuery({ ...c.req.query(), limit: "1000" });
  const fmt = c.req.query("format") === "txt" ? "txt" : "ndjson";
  const max = Math.min(Number(c.req.query("max") || 50_000), 50_000);
  const env = c.env;
  const enc = new TextEncoder();
  const line = (l: XLine) =>
    fmt === "txt"
      ? `${new Date(l.ts).toISOString()} ${l.level.toUpperCase().padEnd(5)} [${l.source}${l.pod_id ? ` ${l.pod_id}` : ""}${l.pool ? ` ${l.pool}` : ""}] ${l.target ? `${l.target}: ` : ""}${l.msg}${l.fields ? ` ${JSON.stringify(l.fields)}` : ""}\n`
      : JSON.stringify(l) + "\n";
  const stream = new ReadableStream({
    async start(ctl) {
      let n = 0;
      let cursor = q.cursor;
      try {
        while (n < max) {
          const r = await queryLogs(env, { ...q, cursor, limit: Math.min(1000, max - n) });
          for (const l of r.lines) ctl.enqueue(enc.encode(line(l)));
          n += r.lines.length;
          if (!r.next) break;
          cursor = decodeCursor(r.next);
        }
      } catch (e) {
        ctl.enqueue(enc.encode(fmt === "txt" ? `# export stopped: ${(e as Error).message}\n` : JSON.stringify({ error: (e as Error).message }) + "\n"));
      }
      ctl.close();
    },
  });
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "");
  return new Response(stream, {
    headers: { "content-type": fmt === "txt" ? "text/plain; charset=utf-8" : "application/x-ndjson", "content-disposition": `attachment; filename="fv-logs-${stamp}.${fmt === "txt" ? "log" : "ndjson"}"`, "cache-control": "no-store" },
  });
});
app.get("/api/logs/download", async (c) => {
  const pod = c.req.query("pod") || "";
  const row = await c.env.DB.prepare("SELECT cluster_id FROM cluster_pods WHERE pod_id = ?").bind(pod).first<{ cluster_id: string }>();
  if (!row) throw new HttpError(404, "not a controller pod");
  const day = c.req.query("day") || utcDay(now());
  return new Response(await downloadLogs(c.env, row.cluster_id, pod, day), {
    headers: { "content-type": "application/x-ndjson", "content-disposition": `attachment; filename="${pod}-${day}.ndjson"`, "cache-control": "no-store" },
  });
});
app.get("/api/logs/tail", async (c) => {
  const pod = c.req.query("pod") || "*";
  let cluster = c.req.query("cluster");
  if (!cluster && pod !== "*") cluster = (await c.env.DB.prepare("SELECT cluster_id FROM cluster_pods WHERE pod_id = ?").bind(pod).first<{ cluster_id: string }>())?.cluster_id;
  if (!cluster) throw new HttpError(400, "cluster or a controller pod");
  const url = new URL("https://ops/tail");
  url.searchParams.set("pod", pod);
  return c.env.CLUSTER_OPS.get(c.env.CLUSTER_OPS.idFromName(cluster)).fetch(url.toString(), c.req.raw);
});

// ---------------- releases, images, GitHub
app.get("/api/releases", async (c) => {
  const heads = await releaseHeads(c.env);
  const cls = await listClusters(c.env);
  return c.json({ ...heads, registry: await registry(c.env), drift: cls.map((cl) => ({ cluster: cl.name, ...clusterDrift(cl, heads.heads) })) });
});
app.get("/api/images/tags", async (c) => {
  const f = c.req.query("filter") || "";
  const tags = (await listTags(c.env)).filter((t) => t.includes(f));
  return c.json({ tags: tags.slice(-500) });
});
app.get("/api/github/ci", async (c) => c.json(await ciStatus(c.env)));
app.post("/api/github/release", async (c) => {
  const b = await body(c);
  parseOr400("release-dispatch", b);
  const known = (await dynamicEnums(c.env)).channels.map((x) => x.id);
  if (b.channel && !known.includes(b.channel)) throw new HttpError(400, `channel: one of ${known.join(", ")}`, { issues: [{ path: ["channel"], message: `one of ${known.join(", ")}` }] });
  const r = await dispatchRelease(c.env, b);
  await auditC(c, { action: `release.${b.action}`, target: b.channel || "stable", after: r.inputs });
  return c.json(r, 202);
});

// ---------------- tokens, audit, collect
app.get("/api/tokens", async (c) => {
  const r = await c.env.DB.prepare("SELECT id, name, scope, created_at, created_by, expires_at, last_used_at, revoked_at FROM api_tokens ORDER BY created_at DESC").all<any>();
  return c.json({ tokens: r.results || [] });
});
app.post("/api/tokens", async (c) => {
  if (c.get("authKind") === "token") throw new HttpError(403, "API tokens cannot mint tokens");
  const b = await body<{ name?: string; scope?: string; ttl_days?: number }>(c);
  const v = validate("token-create", { scope: "admin", ...b });
  if (!v.ok) throw new HttpError(400, v.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; "), { issues: v.issues });
  const name = String(b.name);
  await assertNameFree(c.env, "token", name);
  const scope = b.scope === "read" ? "read" : b.scope === "ci" ? "ci" : "admin";
  const t = await mintApiToken(c.env, name, scope, actor(c), b.ttl_days ? Number(b.ttl_days) : 90);
  await auditC(c, { action: "token.mint", target: t.id, after: { name, scope, expires_at: t.expires_at } });
  return c.json(t, 201);
});
app.delete("/api/tokens/:id", async (c) => {
  await c.env.DB.prepare("UPDATE api_tokens SET revoked_at = ? WHERE id = ?").bind(now(), c.req.param("id")).run();
  await auditC(c, { action: "token.revoke", target: c.req.param("id") });
  return c.json({ ok: true });
});
app.get("/api/audit", async (c) => {
  const limit = Math.min(Number(c.req.query("limit") || 100), 1000);
  const r = await c.env.DB.prepare("SELECT * FROM audit ORDER BY id DESC LIMIT ?").bind(limit).all<any>();
  return c.json({ audit: r.results || [] });
});
app.post("/api/collect", async (c) => c.json(await collect(c.env)));

// ---------------- schemas and editable documents (src/schemas.ts, src/docs.ts)
app.get("/api/schemas", (c) => c.json({ schemas: jsonSchemas(), documents: SCHEMA_OF }));
/** A name's pattern, reserved names and uniqueness (the forms check as you type; every create path enforces the same). */
app.get("/api/names/:kind", async (c) => c.json(await checkName(c.env, c.req.param("kind"), c.req.query("name") || "")));
/**
 * The image preflight before a save or launch (preflight.ts, the check `up` runs before any pod is paid for):
 * each pool's image resolves to a digest, and that build has what the spec asks of it. Body: {spec} (a cluster spec),
 * {launch} (a standalone launch request) or {endpoint} (a serverless spec: its variant and image).
 */
app.post("/api/preflight", async (c) => {
  const b = await body<any>(c);
  let spec;
  if (b.launch) spec = standaloneSpec(b.launch).spec;
  else if (b.endpoint) {
    const e = b.endpoint;
    spec = normalizeSpec({ name: "preflight", image: e.image, control_plane: "direct", pools: [{ id: "endpoint", variant: e.variant, compute: e.variant === "cpu" ? "CPU" : "GPU", count: 1, config: "/etc/fv/runpod.toml", ...(e.variant === "cpu" ? { fake_models: ["fake-wan"] } : { models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] }) }] });
  } else spec = normalizeSpec(b.spec || {});
  let images: Record<string, string>;
  try {
    images = await resolveClusterImages(c.env, spec);
  } catch (e) {
    return c.json({ ok: false, errors: [`image: ${scrub(c.env, (e as Error).message)}`], warnings: [], images: {}, revisions: {} });
  }
  const pf = await checkImages(c.env, spec, images);
  return c.json({ ok: pf.errors.length === 0, ...pf, images });
});
app.get("/api/schemas/dynamic", async (c) => c.json(await dynamicEnums(c.env, c.req.query("cluster") || undefined)));
/** Any schema's own check (refinements included) on a draft, without acting: the forms' cross-field rules. */
app.post("/api/schemas/:name/validate", async (c) => {
  const name = c.req.param("name") as SchemaName;
  if (!jsonSchemas()[name]) throw new HttpError(404, "no such schema");
  const v = validate(name, (await body<any>(c)).doc);
  return c.json({ ok: v.ok, issues: v.ok ? [] : v.issues });
});
app.get("/api/schemas/:name", (c) => {
  const s = jsonSchemas()[c.req.param("name")];
  if (!s) throw new HttpError(404, "no such schema");
  return c.json(s);
});
const kindOf = (k: string): DocKind => {
  if (!DOC_KINDS.includes(k as DocKind)) throw new HttpError(404, `document kinds: ${DOC_KINDS.join(", ")}`);
  return k as DocKind;
};
app.get("/api/docs/:kind/:id", async (c) => {
  const kind = kindOf(c.req.param("kind"));
  const id = await canonicalId(c.env, kind, c.req.param("id"));
  return c.json({ kind, id, schema: SCHEMA_OF[kind], version: await docVersion(c.env, kind, id), doc: await readDoc(c.env, kind, id) });
});
app.post("/api/docs/:kind/:id/validate", async (c) => {
  const kind = kindOf(c.req.param("kind"));
  const id = await canonicalId(c.env, kind, c.req.param("id"));
  const r = await validateDoc(c.env, kind, id, (await body(c)).doc);
  return c.json({ ok: r.ok, issues: r.issues });
});
app.post("/api/docs/:kind/:id/plan", async (c) => {
  if (c.req.param("kind") !== "cluster-spec") throw new HttpError(400, "plans are for cluster-spec documents");
  return c.json(await planSpec(c.env, c.req.param("id"), (await body(c)).doc));
});
app.put("/api/docs/:kind/:id", async (c) => {
  const kind = kindOf(c.req.param("kind"));
  const b = await body<{ doc?: unknown; version?: number }>(c);
  if (typeof b.version !== "number") throw new HttpError(400, "version: the version you loaded (optimistic concurrency)");
  return c.json(await saveDoc(c.env, kind, c.req.param("id"), b.doc, { version: b.version, actor: actor(c), ip: clientIp(c) }));
});
app.get("/api/docs/:kind/:id/history", async (c) => c.json({ history: await docHistory(c.env, kindOf(c.req.param("kind")), c.req.param("id")) }));
app.post("/api/docs/:kind/:id/restore", async (c) => {
  const kind = kindOf(c.req.param("kind"));
  const b = await body<{ audit_id?: number; which?: "after" | "before"; version?: number }>(c);
  if (typeof b.version !== "number") throw new HttpError(400, "version: the version you loaded");
  return c.json(await restoreDoc(c.env, kind, c.req.param("id"), Number(b.audit_id), b.which === "before" ? "before" : "after", { version: b.version, actor: actor(c), ip: clientIp(c) }));
});

// ---------------- Runpod serverless endpoints (src/serverless/, docs/control/serverless.md)
app.route("/api/serverless", serverlessRoutes);

app.all("/api/*", () => {
  throw new HttpError(404, "no such route");
});

// ---------------- the serverless console (src/serverless/console.ts, docs/control/serverless.md "Console"):
// fv-serve's console for one endpoint, its API calls turned into Runpod jobs. Uploads are read back by the
// workers through a signed URL (public: the token is the capability).
app.get("/serverless-uploads/:token/:name", (c) => uploadGet(c.env, c.req.param("token")));
app.use("/serverless/*", requireConsoleAuth);
const consoleRoute = async (c: C) => {
  const ep = c.req.param("ep") || "";
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(ep)) throw new HttpError(404, "no such endpoint");
  const row = await getRow(c.env, ep);
  return consoleRequest(c.env, c.req.raw, row, { ep, who: { actor: actor(c), ip: clientIp(c) }, readOnly: c.get("scope") === "read", waitUntil: (p) => c.executionCtx.waitUntil(p) });
};
app.all("/serverless/:ep", consoleRoute);
app.all("/serverless/:ep/*", consoleRoute);
app.all("*", async (c) => {
  if (c.env.ASSETS) return c.env.ASSETS.fetch(c.req.raw);
  return c.text("not found", 404);
});

export default {
  fetch: app.fetch,
  async scheduled(_ev: ScheduledController, env: Env, ctx: ExecutionContext) {
    if (env.CRON_DISABLED === "1") return;
    ctx.waitUntil(
      collect(env).catch((e) => {
        console.error("collector failed", scrub(env, (e as Error).message));
      }),
    );
    ctx.waitUntil(serverlessTick(env).catch((e) => console.error("serverless tick failed", scrub(env, (e as Error).message))));
    // The serverless console's uploads (R2 console-uploads/): gone after a day, checked hourly.
    if (new Date().getUTCMinutes() === 7) ctx.waitUntil(sweepUploads(env).catch(() => 0));
  },
} satisfies ExportedHandler<Env>;

export const _app = app;
