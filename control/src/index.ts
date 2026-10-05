// fv-control: the fastvideo-rs cluster controller (docs/control/README.md).
// A Hono app on Cloudflare Workers: JSON API under /api (what the dashboard
// in public/ uses, and scripts/serve/fv-control.sh), log ingest under
// /ingest, a per-minute cron (collector.ts) and one ClusterOps Durable
// Object per cluster (cluster/do.ts).
import { Hono, type Context } from "hono";
import { DEFAULT_POLICIES, policies } from "./alerts";
import { accessMode, clientIp, login, logout, mintApiToken, requireAuth, whoami } from "./auth";
import { cancelOp, currentOp, startOp } from "./cluster/control";
import { adminAll, adminGet, adminOne, adminTargets, adminToken, desiredEnv, envCtx, gatewayPublic, projectSpend, workerHealth } from "./cluster/ops";
import { gatewaySystemEnv, isDirect, workerSystemEnv, type ClusterSecrets, type ClusterState, type PodRec } from "./cluster/payloads";
import { defaultSpec, normalizeSpec, STANDARD_POOLS } from "./cluster/spec";
import { allPods, emptyState, getCluster, listClusters, livePods, saveSecrets, saveSpec, type Cluster } from "./cluster/store";
import { collect } from "./collector";
import { randomToken, sha256Hex, unb64, x25519Generate, x25519PublicOf, b64 } from "./crypto";
import type { Env, Vars } from "./env";
import { deleteVar, listVars, maskRow, resolveView, setVar, type Scope } from "./envvars";
import { ciStatus, dispatchRelease } from "./github";
import { listTags } from "./ghcr";
import { canonicalId, docHistory, DOC_KINDS, docVersion, planSpec, readDoc, restoreDoc, saveDoc, SCHEMA_OF, validateDoc, bumpDoc, type DocKind } from "./docs";
import { dynamicEnums } from "./dynamic";
import { downloadLogs, ingest, searchLogs } from "./logs";
import { jsonSchemas, validate } from "./schemas";
import { querySeries } from "./metrics";
import { clusterDrift, registry, releaseHeads } from "./releases";
import { runpod } from "./runpod";
import { audit, HttpError, newId, now, putSetting, scrub, utcDay } from "./util";

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
      status: cl.status,
      deadline: cl.deadline,
      dph: running.filter((p) => p.cluster_id === cl.id).reduce((s, p) => s + (p.cost_per_hr || 0), 0),
      pods: running.filter((p) => p.cluster_id === cl.id).length,
      cost_today: (costToday.results || []).filter((r: any) => r.cluster_id === cl.id).reduce((s: number, r: any) => s + r.usd, 0),
    })),
    policies: pol,
  });
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
  return { id: cl.id, name: cl.name, status: cl.status, deadline: cl.deadline, source: cl.source, created_at: cl.created_at, updated_at: cl.updated_at, created_by: cl.created_by, spec: cl.spec, state: cl.state };
}
async function newSecrets(): Promise<{ s: ClusterSecrets; ingestHash: string }> {
  const kp = await x25519Generate();
  const ingestToken = randomToken("fvi_", 32);
  return {
    s: { internal_token: randomToken(), url_signing_key: randomToken(), admin_recipient: kp.publicRaw, admin_private: kp.privatePkcs8, ingest_token: ingestToken },
    ingestHash: await sha256Hex(ingestToken),
  };
}
async function insertCluster(env: Env, spec: ReturnType<typeof normalizeSpec>, state: ClusterState, secrets: ClusterSecrets, ingestHash: string, by: string, source: string, status: string, deadline: number | null): Promise<Cluster> {
  const id = newId("c");
  const exists = await env.DB.prepare("SELECT 1 AS x FROM clusters WHERE name = ?").bind(spec.name).first();
  if (exists) throw new HttpError(409, `a cluster named ${spec.name} exists`);
  await env.DB.prepare("INSERT INTO clusters (id, name, spec, state, status, deadline, ingest_hash, source, created_at, updated_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    .bind(id, spec.name, JSON.stringify(spec), JSON.stringify(state), status, deadline, ingestHash, source, now(), now(), by)
    .run();
  const cl = await getCluster(env, id);
  await saveSecrets(env, cl, secrets);
  return cl;
}

app.get("/api/templates", (c) => c.json({ standard: defaultSpec("example"), "tiny-cpu": defaultSpec("example", "tiny-cpu"), standard_pools: STANDARD_POOLS }));
app.get("/api/clusters", async (c) => {
  const cls = await listClusters(c.env);
  const out = [];
  for (const cl of cls) out.push({ ...clusterView(cl), op: await currentOp(c.env, cl.id).catch(() => null) });
  return c.json({ clusters: out });
});
app.post("/api/clusters", async (c) => {
  const b = await body(c);
  const spec = normalizeSpec(b.spec || b);
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
  return c.json({ cluster: clusterView(cl), pods, live: live.results || [], ops: ops.results || [], op: await currentOp(c.env, cl.id).catch(() => null), drift: clusterDrift(cl, heads.heads) });
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
  await c.env.DB.prepare("DELETE FROM clusters WHERE id = ?").bind(cl.id).run();
  await c.env.DB.prepare("DELETE FROM env_vars WHERE scope = 'cluster' AND scope_id = ?").bind(cl.id).run();
  await auditC(c, { action: "cluster.delete", target: cl.name, before: cl.spec });
  return c.json({ deleted: cl.id });
});
app.post("/api/clusters/:id/price", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const b = await body(c);
  return c.json(await projectSpend(c.env, cl.spec, { hours: Number(b.hours) > 0 ? Number(b.hours) : cl.spec.cap_s / 3600 }));
});

const OPS: Record<string, { kind: Parameters<typeof startOp>[2]; params: (b: any) => any }> = {
  start: { kind: "up", params: (b) => ({ skip_price_check: false, ...(b.confirm_over_floor ? {} : {}) }) },
  stop: { kind: "down", params: () => ({ reason: "stop" }) },
  extend: { kind: "extend", params: (b) => ({ minutes: Number(b.minutes) }) },
  scale: { kind: "scale", params: (b) => ({ pool: String(b.pool || ""), count: Number(b.count) }) },
  roll: { kind: "roll", params: (b) => ({ target: String(b.target || "stable"), pools: Array.isArray(b.pools) ? b.pools.map(String) : undefined, gateway: !!b.gateway }) },
  restart: { kind: "restart", params: (b) => ({ pods: Array.isArray(b.pods) ? b.pods.map(String) : undefined }) },
  "gateway/start": { kind: "gateway-start", params: () => ({}) },
  "gateway/stop": { kind: "gateway-stop", params: () => ({}) },
};
for (const [path, def] of Object.entries(OPS)) {
  app.post(`/api/clusters/:id/${path}`, async (c) => {
    const cl = await getCluster(c.env, c.req.param("id"));
    const params = def.params(await body(c));
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
  for (const r of rows.filter((x: any) => x.slot !== "retired")) {
    const des = await desiredEnv(env, cl, ctx, r.role, { pod: r.pod_id, pool: r.pool, image: r.image });
    pods.push({ pod_id: r.pod_id, role: r.role, pool: r.pool, needs_restart: des.hash !== r.env_hash, env_applied_at: r.env_applied_at, env: await resolveView(env, cl.id, r.pod_id, des.system) });
  }
  // What a new pod of each kind would get (also when the cluster is stopped).
  const preview: Record<string, unknown> = {};
  const img = (k: string) => cl.state.images[k] || cl.state.image || `(${k} image)`;
  if (cl.spec.gateway.enabled) preview.gateway = await resolveView(env, cl.id, null, gatewaySystemEnv(ctx, img("gateway")));
  for (const p of cl.spec.pools) preview[p.id] = await resolveView(env, cl.id, null, workerSystemEnv(ctx, p, img(p.id)));
  return c.json({ pods, preview, needs_restart: pods.filter((p) => p.needs_restart).map((p) => p.pod_id) });
});
/** Where clients go: the gateway, or each worker of a gateway-less cluster (docs/control/gateway-less-auth.md). */
function clientUrls(cl: Cluster) {
  const workers = Object.entries(cl.state.workers || {}).flatMap(([pool, recs]) => recs.map((r) => ({ pod: r.pod, pool, url: r.url || null })));
  return { direct: isDirect(cl.spec, cl.state), gateway_url: cl.state.gateway_url || null, workers };
}
app.get("/api/clusters/:id/gateway", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  if (isDirect(cl.spec, cl.state)) {
    // No gateway: each worker's public /health stands in for the status and pools views.
    const u = clientUrls(cl);
    const workers = await Promise.all(u.workers.map(async (w) => ({ ...w, health: await workerHealth(c.env, w.pod) })));
    return c.json({ url: null, direct: true, workers });
  }
  const status = await gatewayPublic(c.env, cl, "/fv/v1/status").catch((e) => ({ status: 0, body: { error: (e as Error).message } }));
  const pools = await adminGet(c.env, cl, "/fv/v1/gateway/pools").catch((e) => ({ error: (e as Error).message }));
  return c.json({ url: cl.state.gateway_url || null, status: status.body, pools });
});
app.post("/api/clusters/:id/admin-token", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const tok = await adminToken(c.env, cl);
  await auditC(c, { action: "cluster.admin-token.reveal", target: cl.name });
  const u = clientUrls(cl);
  const base = u.direct ? u.workers.find((w) => w.url)?.url : cl.state.gateway_url;
  return c.json({ admin_token: tok, console: base ? `${base}/console/admin` : null, ...u });
});
app.post("/api/clusters/:id/mint-key", async (c) => {
  const cl = await getCluster(c.env, c.req.param("id"));
  const name = String((await body(c)).name || "fv-control");
  if (!/^[A-Za-z0-9 ._-]{1,60}$/.test(name)) throw new HttpError(400, "name: 1-60 characters");
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
  adminTargets(cl);
  // Every worker of a gateway-less cluster at once; one that misses it reads the revocation from D1 within 30 s.
  const rs = await adminAll(c.env, cl, "DELETE", `/fv/v1/admin/keys/${kid}`);
  const ok = rs.filter((r) => r.status === 200);
  await auditC(c, { action: "cluster.revoke-key", target: cl.name, after: { key: kid, applied: ok.map((r) => r.pod), failed: rs.filter((r) => r.status !== 200).map((r) => `${r.pod}:${r.status}`) } });
  if (!ok.length) throw new HttpError(rs.every((r) => r.status === 404) ? 404 : 502, `revoke refused (${rs.map((r) => r.status).join(",")})`);
  return c.json({ key: ok[0]!.body?.key, applied: ok.map((r) => r.pod), failed: rs.filter((r) => r.status !== 200).map((r) => ({ pod: r.pod, status: r.status })) });
});

/** Import a runpod-cluster.sh state file (artifacts/runpod/serve/cluster.json) and, optionally, its .admin-key.pem. */
app.post("/api/clusters/import", async (c) => {
  const b = await body<{ name?: string; state?: any; admin_key_pem?: string }>(c);
  const st = b.state;
  if (!st || typeof st !== "object" || !st.internal_token || !st.gateway?.pod) throw new HttpError(400, "state: a runpod-cluster.sh cluster.json (internal_token, gateway.pod, workers)");
  const pools = Object.keys(st.workers || {});
  const spec = normalizeSpec({
    name: b.name || `imported-${String(st.gateway.pod).slice(0, 6)}`,
    image: st.images && Object.keys(st.images).length ? { ref: st.images.gateway || st.image } : { ref: st.image },
    pools: pools.map((p) => ({ ...(STANDARD_POOLS.find((x) => x.id === p) || { id: p, variant: p, compute: "GPU", config: `/etc/fv/runpod-${p}.toml`, models: [] }), count: 1 })),
    gateway: { auth: st.auth || "keys" },
    min_balance: Number(st.min_balance || 8.25),
    log_shipping: true,
  });
  if (st.images && Object.keys(st.images).length) spec.image = { channel: "stable" };
  const recOf = (r: any, pool?: string): PodRec => ({ pod: r.pod, pool, gpu: r.gpu, cpu: r.cpu, dc: r.dc, dph: Number(r.dph || 0), created: Number(r.created || Math.floor(now() / 1000)), image: r.image || st.images?.[pool || "gateway"] || st.image, url: r.url });
  const state: ClusterState = {
    image: st.image,
    images: st.images && Object.keys(st.images).length ? st.images : Object.fromEntries(["gateway", ...pools].map((k) => [k, st.image])),
    gateway: { ...recOf(st.gateway), url: st.gateway_url },
    gateway_url: st.gateway_url,
    workers: Object.fromEntries(pools.map((p) => [p, [recOf(st.workers[p], p)]])),
    ...(st.rolling ? { rolling: Object.fromEntries(Object.entries(st.rolling).map(([p, r]) => [p, [recOf(r, p)]])) } : {}),
    ...(st.retired?.length ? { retired: st.retired.map((r: any) => recOf(r)) } : {}),
  };
  const ingestToken = randomToken("fvi_", 32);
  const s: ClusterSecrets = { internal_token: st.internal_token, url_signing_key: st.url_signing_key, admin_recipient: st.admin_recipient || undefined, admin_token: st.admin_token || undefined, smoke_api_key: st.smoke_api_key || undefined, ingest_token: ingestToken };
  if (!st.admin_recipient && st.admin_token) s.legacy_admin_token = true;
  if (b.admin_key_pem) {
    const der = b.admin_key_pem.replace(/-----[^-]+-----/g, "").replace(/\s+/g, "");
    s.admin_private = b64(unb64(der));
    const pub = await x25519PublicOf(s.admin_private);
    if (s.admin_recipient && pub !== s.admin_recipient) throw new HttpError(400, "admin_key_pem is not the key pair of this state (public keys differ)");
    s.admin_recipient = pub;
  }
  const deadline = Number(st.deadline) ? Number(st.deadline) * 1000 : null;
  const cl = await insertCluster(c.env, spec, state, s, await sha256Hex(ingestToken), actor(c), "import", "running", deadline);
  for (const x of allPods(state))
    await c.env.DB.prepare("INSERT OR IGNORE INTO cluster_pods (pod_id, cluster_id, role, pool, slot, image, gpu, dc, cost_per_hr, url, created_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready')")
      .bind(x.rec.pod, cl.id, x.role, x.rec.pool ?? null, x.slot, x.rec.image, x.rec.gpu ?? (x.rec.cpu ? `cpu:${x.rec.cpu}` : null), x.rec.dc ?? null, x.rec.dph, x.rec.url ?? null, x.rec.created * 1000)
      .run();
  await auditC(c, { action: "cluster.import", target: cl.name, after: { pods: allPods(state).map((x) => x.rec.pod), deadline } });
  return c.json({ cluster: clusterView(cl), note: "env changes (log shipping) apply on the next restart; the script's local backstop still holds its own copy of the state" }, 201);
});

// ---------------- env vars
const scopeOf = (s: string): Scope => {
  if (s !== "account" && s !== "cluster" && s !== "pod") throw new HttpError(400, "scope: account | cluster | pod");
  return s;
};
app.get("/api/env/account", async (c) => c.json({ vars: (await listVars(c.env, "account", "")).map(maskRow) }));
app.get("/api/env/:scope/:sid", async (c) => {
  const scope = scopeOf(c.req.param("scope"));
  return c.json({ vars: (await listVars(c.env, scope, scope === "account" ? "" : c.req.param("sid"))).map(maskRow) });
});
async function putVar(c: C, scope: Scope, sid: string, key: string) {
  const b = await body<{ value?: string; secret?: boolean }>(c);
  if (scope === "cluster") await getCluster(c.env, sid);
  const r = await setVar(c.env, scope, sid, key, String(b.value ?? ""), !!b.secret, actor(c));
  await bumpDoc(c.env, "env", scope === "account" ? "account" : `${scope}:${sid}`, actor(c));
  await auditC(c, { action: "env.set", target: `${scope}:${sid || "-"}:${key}`, before: r.before, after: r.after });
  return c.json({ ok: true, ...r.after, key });
}
async function delVar(c: C, scope: Scope, sid: string, key: string) {
  const r = await deleteVar(c.env, scope, sid, key);
  await bumpDoc(c.env, "env", scope === "account" ? "account" : `${scope}:${sid}`, actor(c));
  await auditC(c, { action: "env.delete", target: `${scope}:${sid || "-"}:${key}`, before: r.before });
  return c.json({ ok: true });
}
app.put("/api/env/account/:key", (c) => putVar(c, "account", "", c.req.param("key")));
app.delete("/api/env/account/:key", (c) => delVar(c, "account", "", c.req.param("key")));
app.put("/api/env/:scope/:sid/:key", (c) => putVar(c, scopeOf(c.req.param("scope")), c.req.param("sid"), c.req.param("key")));
app.delete("/api/env/:scope/:sid/:key", (c) => delVar(c, scopeOf(c.req.param("scope")), c.req.param("sid"), c.req.param("key")));

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
    }),
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
  const v = validate("release-dispatch", b);
  if (!v.ok) throw new HttpError(400, v.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; "), { issues: v.issues });
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
  const scope = b.scope === "read" ? "read" : "admin";
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
app.get("/api/schemas/dynamic", async (c) => c.json(await dynamicEnums(c.env, c.req.query("cluster") || undefined)));
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

app.all("/api/*", () => {
  throw new HttpError(404, "no such route");
});
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
  },
} satisfies ExportedHandler<Env>;

export const _app = app;
