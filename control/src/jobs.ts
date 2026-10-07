// fv-serve jobs of clusters and standalone pods (docs/control/README.md
// "Jobs"): the list, and cancel.
//
// Where the jobs are: every fv-serve worker writes its jobs to a D1 `jobs`
// table (serve-kit d1/schema.rs: id = the internal uuid, protocol =
// the API that owns external_id, worker = the pod that holds it). Edge
// clusters write to the edge's D1 (EDGE_DB, the database EDGE_D1_DATABASE_ID
// names); direct clusters and standalone pods to fv-jobs (JOBS_DB). A job
// still queued at the edge (in its family object) has no worker yet.
//
// How a cancel is routed: to the worker's internal route
// `DELETE /fv/v1/internal/jobs/{id}` with the internal token (the edge's
// for edge fronts, the cluster's for direct workers), straight to the pod.
// It is keyed by the internal id, so it serves every API, and it runs
// serve-kit's cancel_job: a queued job is cancelled at once, a running one
// gets cancel_requested and stops at its next denoise step. On an edge
// front the cancel goes through the engine seam (FrontGate::cancel): the
// job held there stops locally, any other one through the family object's
// cancel, so for a job queued at the edge (no worker) any live front of
// the cluster will do. The public routes of the API that owns the id
// (jobapi.ts) are the fallback, through the edge or on the pod: they need
// the job owner's API key (the edge does not take its admin token as a
// key), so they only work where auth is none.
import { cancelRoute, noCancelReason } from "./jobapi";
import { defaults, type Env } from "./env";
import { edgeFetch, requireEdge } from "./cluster/ops";
import { isEdge } from "./cluster/spec";
import { getCluster, secretsOf, type Cluster } from "./cluster/store";
import { audit, fetchWithTimeout, HttpError } from "./util";

export type Actor = { actor: string; ip?: string };
export const TERMINAL_JOB = new Set(["succeeded", "failed", "cancelled"]);

/** The jobs D1 a cluster's workers write to. */
export function jobsDb(env: Env, c: Cluster): { db: D1Database; source: "edge" | "fv-jobs" } {
  if (isEdge(c.spec)) {
    if (!env.EDGE_DB) throw new HttpError(409, `${c.name} is an edge cluster: its jobs are in the edge's D1, and fv-control has no EDGE_DB binding (wrangler.toml, the database EDGE_D1_DATABASE_ID names)`);
    return { db: env.EDGE_DB, source: "edge" };
  }
  if (!env.JOBS_DB) throw new HttpError(409, `${c.name} is a direct ${c.source === "standalone" ? "pod" : "cluster"}: its jobs are in fv-jobs, and fv-control has no JOBS_DB binding`);
  return { db: env.JOBS_DB, source: "fv-jobs" };
}

interface PodRow {
  pod_id: string;
  pool: string | null;
  created_at: number;
  deleted_at: number | null;
}
/** The cluster's pods, newest first (history too: a stopped cluster's jobs still list). D1 binds at most 100 values. */
async function podsOf(env: Env, c: Cluster): Promise<PodRow[]> {
  const r = await env.DB.prepare("SELECT pod_id, pool, created_at, deleted_at FROM cluster_pods WHERE cluster_id = ? AND role = 'worker' ORDER BY created_at DESC LIMIT 90").bind(c.id).all<PodRow>();
  return r.results || [];
}

export interface JobView {
  id: string; // fv-serve's internal id
  external_id: string; // the id its API gave the client
  api: string; // the API that owns external_id (protocol)
  status: string;
  model: string;
  resolved_model: string | null;
  task: string | null;
  progress: number;
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
  worker: string | null;
  pool: string | null;
  owner: string | null; // the API key id (never the key)
  cancel_requested: boolean;
  /** The owning API's own cancel route (what a client calls); null when it has none. */
  cancel_route: string | null;
}
export interface JobsQuery {
  status?: string; // csv of JOB_STATUSES
  pool?: string;
  pod?: string;
  limit?: string;
}

/** A jobs D1 no worker has written to yet has no table (serve-kit creates it on first use). */
const noTable = (e: unknown) => /no such table: jobs/i.test(String((e as Error)?.message || e));
const COLS =
  "id, external_id, protocol, status, model, resolved_model, task, progress, created_at, completed_at, worker, owner, json_extract(job, '$.started_at') AS started_at, json_extract(job, '$.cancel_requested') AS cancel_requested";
function view(r: any, pools: Map<string, string | null>): JobView {
  const started = r.started_at ? Date.parse(String(r.started_at)) : NaN;
  const route = cancelRoute(r.protocol, r.external_id, r.model);
  return {
    id: String(r.id),
    external_id: String(r.external_id),
    api: String(r.protocol),
    status: String(r.status),
    model: String(r.model),
    resolved_model: r.resolved_model ?? null,
    task: r.task ?? null,
    progress: Number(r.progress || 0),
    created_at: Number(r.created_at),
    started_at: Number.isFinite(started) ? started : null,
    completed_at: r.completed_at ?? null,
    worker: r.worker ?? null,
    pool: r.worker ? (pools.get(r.worker) ?? null) : null,
    owner: r.owner ?? null,
    cancel_requested: r.cancel_requested === 1 || r.cancel_requested === true,
    cancel_route: route ? `${route.method} ${route.path}` : null,
  };
}

/** When an edge cluster ran: from its first `up` (clients can enqueue while the workers boot) or its first pod, to now while
 * it has live pods, else to its last pod's deletion. */
async function edgeWindow(env: Env, c: Cluster, pods: PodRow[]): Promise<{ since: number; until: number } | null> {
  if (!pods.length) return null;
  const up = await env.DB.prepare("SELECT MIN(created_at) AS at FROM operations WHERE cluster_id = ? AND kind = 'up'").bind(c.id).first<{ at: number | null }>();
  const since = Math.min(up?.at ?? Number.MAX_SAFE_INTEGER, ...pods.map((p) => p.created_at));
  const until = pods.some((p) => !p.deleted_at) ? Number.MAX_SAFE_INTEGER : Math.max(...pods.map((p) => p.deleted_at || 0));
  return { since, until };
}
/** Which rows are the cluster's: its pods' jobs; for an edge cluster also the jobs still queued at the edge
 * (no worker) while it ran (one running edge cluster per edge, docs/serve/edge-control-plane.md §5.1). */
function scope(c: Cluster, pods: PodRow[], q: { pool?: string; pod?: string }, win: { since: number; until: number } | null): { sql: string; args: unknown[] } {
  let ps = pods;
  if (q.pool) ps = ps.filter((p) => p.pool === q.pool);
  if (q.pod) ps = ps.filter((p) => p.pod_id === q.pod);
  const parts: string[] = [];
  const args: unknown[] = [];
  if (ps.length) {
    parts.push(`worker IN (${ps.map(() => "?").join(", ")})`);
    args.push(...ps.map((p) => p.pod_id));
  }
  if (isEdge(c.spec) && !q.pool && !q.pod && win) {
    parts.push("(worker IS NULL AND created_at >= ? AND created_at <= ?)");
    args.push(win.since, win.until);
  }
  return { sql: parts.length ? `(${parts.join(" OR ")})` : "0", args };
}

/** GET /api/clusters/:id/jobs: recent and running jobs, newest first, with the count per status. */
export async function listJobs(env: Env, c: Cluster, q: JobsQuery) {
  const { db, source } = jobsDb(env, c);
  const pods = await podsOf(env, c);
  if (q.pool && !c.spec.pools.some((p) => p.id === q.pool) && !pods.some((p) => p.pool === q.pool)) throw new HttpError(404, `no pool ${q.pool} in ${c.name}`);
  if (q.pod && !pods.some((p) => p.pod_id === q.pod)) throw new HttpError(404, `pod ${q.pod} is not one of ${c.name}'s`);
  const pools = new Map(pods.map((p) => [p.pod_id, p.pool]));
  const where = scope(c, pods, q, isEdge(c.spec) ? await edgeWindow(env, c, pods) : null);
  const statuses = q.status ? q.status.split(",").filter(Boolean) : [];
  const limit = Math.min(Math.max(Number(q.limit || 100) || 100, 1), 500);
  const st = statuses.length ? ` AND status IN (${statuses.map(() => "?").join(", ")})` : "";
  let rows: { results?: any[] };
  let counts: { results?: { status: string; n: number }[] };
  let note: string | undefined;
  try {
    rows = await db.prepare(`SELECT ${COLS} FROM jobs WHERE ${where.sql}${st} ORDER BY created_at DESC LIMIT ?`).bind(...where.args, ...statuses, limit).all<any>();
    counts = await db.prepare(`SELECT status, COUNT(*) AS n FROM jobs WHERE ${where.sql} GROUP BY status`).bind(...where.args).all<{ status: string; n: number }>();
  } catch (e) {
    if (!noTable(e)) throw e;
    rows = counts = { results: [] };
    note = `${source === "edge" ? "the edge's D1" : "fv-jobs"} has no jobs table yet: no worker has written a job to it`;
  }
  return {
    ...(note ? { note } : {}),
    cluster: { id: c.id, name: c.name, kind: c.source === "standalone" ? "standalone" : "cluster", control_plane: isEdge(c.spec) ? "edge" : "direct" },
    source,
    counts: Object.fromEntries((counts.results || []).map((r) => [r.status, Number(r.n)])),
    pods: pods.map((p) => ({ pod_id: p.pod_id, pool: p.pool, live: !p.deleted_at })),
    jobs: (rows.results || []).map((r) => view(r, pools)),
  };
}

// ---------------------------------------------------------------- cancel
export interface Attempt {
  via: "internal" | "api";
  target: string;
  status: number;
  message: string;
}
export interface CancelResult {
  job: string;
  external_id: string;
  api: string;
  cluster: string;
  ok: boolean;
  /** internal: the worker's internal route on the pod; api: the owning API's public route. */
  via: "internal" | "api" | null;
  pod: string | null;
  status: string;
  cancel_requested: boolean;
  note: string;
  attempts: Attempt[];
}

const internalToken = async (env: Env, c: Cluster) => (isEdge(c.spec) ? requireEdge(env).internal_token : (await secretsOf(env, c)).internal_token);
const short = (b: any) => (typeof b === "string" ? b : b?.error?.message || b?.error || b?.message || JSON.stringify(b ?? "")).toString().slice(0, 200);
async function call(url: string, init: RequestInit & { timeoutMs?: number }, viaEdge?: Env): Promise<{ status: number; body: any }> {
  try {
    const r = viaEdge ? await edgeFetch(viaEdge, url, init) : await fetchWithTimeout(url, init);
    const t = await r.text();
    let body: any = t;
    try {
      body = JSON.parse(t);
    } catch {
      /* text */
    }
    return { status: r.status, body };
  } catch (e) {
    return { status: 0, body: (e as Error).message.slice(0, 160) };
  }
}

/** Cancels one job row of cluster `c`; see the module comment for the route. */
export async function cancelJobRow(env: Env, who: Actor, c: Cluster, row: any, o: { audit?: boolean } = {}): Promise<CancelResult> {
  const base = { job: String(row.id), external_id: String(row.external_id), api: String(row.protocol), cluster: c.name };
  if (TERMINAL_JOB.has(row.status)) return { ...base, ok: true, via: null, pod: null, status: row.status, cancel_requested: false, note: `already ${row.status}: nothing to cancel`, attempts: [] };
  const pods = await podsOf(env, c);
  const livePods = pods.filter((p) => !p.deleted_at).map((p) => p.pod_id);
  const holder = row.worker ? String(row.worker) : null;
  const targets: string[] = [];
  if (holder && livePods.includes(holder)) targets.push(holder);
  // An edge front cancels any job through the family object (a job queued at the edge, or a holder that is gone).
  if (isEdge(c.spec)) for (const p of livePods) if (!targets.includes(p)) targets.push(p);
  const attempts: Attempt[] = [];
  const tok = await internalToken(env, c);
  for (const pod of targets.slice(0, 4)) {
    const r = await call(`${defaults.podUrl(env, pod)}/fv/v1/internal/jobs/${encodeURIComponent(row.id)}`, { method: "DELETE", headers: { "x-fv-internal-token": tok }, timeoutMs: 15000 });
    attempts.push({ via: "internal", target: pod, status: r.status, message: r.status === 200 ? String(r.body?.status || "") : short(r.body) });
    if (r.status === 200) return finish(env, who, c, o, { ...base, ok: true, via: "internal", pod, status: String(r.body?.status || row.status), cancel_requested: !!r.body?.cancel_requested, note: noteFor(r.body?.status, !!r.body?.cancel_requested, pod === holder), attempts });
    if (r.status === 401 || r.status === 403) break; // the token is wrong for every pod alike
  }
  // The owning API's public route: through the edge (edge clusters) or on the holder (direct).
  const route = cancelRoute(row.protocol, row.external_id, row.model);
  if (route) {
    let url: string | null = null;
    let viaEdge: Env | undefined;
    let auth = "";
    if (isEdge(c.spec)) {
      const e = requireEdge(env);
      url = `${e.url}${route.path}`;
      viaEdge = env;
      auth = e.admin_token;
    } else if (holder && livePods.includes(holder)) {
      url = `${defaults.podUrl(env, holder)}${route.path}`;
      auth = (await secretsOf(env, c)).admin_token || "";
    }
    if (url) {
      const r = await call(url, { method: route.method, headers: auth ? { authorization: `Bearer ${auth}` } : {}, timeoutMs: 15000 }, viaEdge);
      attempts.push({ via: "api", target: `${route.method} ${route.path}`, status: r.status, message: short(r.body) });
      if (r.status >= 200 && r.status < 300) return finish(env, who, c, o, { ...base, ok: true, via: "api", pod: isEdge(c.spec) ? null : holder, status: String(r.body?.status || "cancel requested"), cancel_requested: true, note: `cancelled through ${base.api}'s own route`, attempts });
    }
  }
  const why = !targets.length
    ? holder
      ? `the pod that held it (${holder}) is gone: the job ended with it (its D1 row can still say ${row.status})`
      : `${c.name} has no live worker to cancel it through`
    : attempts.map((a) => `${a.via} ${a.target}: ${a.status || "unreachable"} ${a.message}`).join("; ");
  const res: CancelResult = { ...base, ok: false, via: null, pod: holder, status: row.status, cancel_requested: false, note: `${why}${route ? "" : `; ${noCancelReason(base.api)}`}`, attempts };
  await finish(env, who, c, o, res);
  return res;
}
function noteFor(status: unknown, requested: boolean, onHolder: boolean): string {
  if (status === "cancelled") return "cancelled";
  if (requested) return `cancel requested${onHolder ? "" : " (through the edge's family object)"}: it stops at its next denoise step`;
  return `the worker answered ${String(status || "?")}`;
}
async function finish(env: Env, who: Actor, c: Cluster, o: { audit?: boolean }, r: CancelResult): Promise<CancelResult> {
  if (o.audit !== false)
    await audit(env, { actor: who.actor, ip: who.ip, action: "job.cancel", target: c.name, ok: r.ok, detail: r.ok ? undefined : r.note, after: { job: r.external_id, id: r.job, api: r.api, via: r.via, pod: r.pod, status: r.status } });
  return r;
}

/** One row (findJob's) as the Jobs view shows it. */
export async function jobOf(env: Env, c: Cluster, row: any): Promise<JobView> {
  return view(row, new Map((await podsOf(env, c)).map((p) => [p.pod_id, p.pool])));
}
/** A job by any of its ids, in the jobs D1s fv-control reads; with the cluster or standalone pod that ran it. */
export async function findJob(env: Env, jobId: string, clusterHint?: string): Promise<{ c: Cluster; row: any }> {
  const hint = clusterHint ? await getCluster(env, clusterHint) : null;
  const dbs: { db: D1Database; edge: boolean }[] = [];
  if (env.EDGE_DB) dbs.push({ db: env.EDGE_DB, edge: true });
  if (env.JOBS_DB) dbs.push({ db: env.JOBS_DB, edge: false });
  if (!dbs.length) throw new HttpError(409, "fv-control reads no jobs D1 (EDGE_DB, JOBS_DB): bind them in wrangler.toml");
  const found: { c: Cluster; row: any }[] = [];
  let foreign: string | null = null;
  for (const { db, edge } of dbs) {
    const rows = await db
      .prepare("SELECT *, json_extract(job, '$.started_at') AS started_at, json_extract(job, '$.cancel_requested') AS cancel_requested FROM jobs WHERE external_id = ? OR id = ? LIMIT 5")
      .bind(jobId, jobId)
      .all<any>()
      .catch((e) => {
        if (noTable(e)) return { results: [] as any[] };
        throw e;
      });
    for (const row of rows.results || []) {
      const { job: _drop, ...r } = row; // the full job (signed URLs, request) never leaves here
      if (r.worker) {
        const owner = await env.DB.prepare("SELECT cluster_id FROM cluster_pods WHERE pod_id = ?").bind(r.worker).first<{ cluster_id: string }>();
        if (!owner) {
          foreign = r.worker;
          continue;
        }
        const c = await getCluster(env, owner.cluster_id).catch(() => null);
        if (c && (!hint || hint.id === c.id)) found.push({ c, row: r });
      } else if (edge) {
        // Queued at the edge: the edge cluster that ran when it was submitted.
        const cands = hint ? [hint] : await edgeClustersAt(env, Number(r.created_at));
        for (const c of cands) if (isEdge(c.spec)) found.push({ c, row: r });
      }
    }
  }
  if (!found.length) {
    if (foreign) throw new HttpError(403, `job ${jobId} ran on pod ${foreign}, which fv-control did not create: refusing to touch it`);
    throw new HttpError(404, `no job ${jobId} in ${dbs.map((d) => (d.edge ? "the edge's D1" : "fv-jobs")).join(" or ")}${hint ? ` for ${hint.name}` : ""} (ids: fvjob_…, a fal request id, a MiniMax task id, video_gen_…, or the internal uuid)`);
  }
  const uniq = [...new Map(found.map((f) => [`${f.c.id}|${f.row.id}`, f])).values()];
  if (uniq.length > 1) throw new HttpError(409, `job ${jobId} matches ${uniq.length} jobs (${uniq.map((f) => `${f.c.name}: ${f.row.protocol} ${f.row.external_id}`).join("; ")}): give the cluster`);
  return uniq[0]!;
}
async function edgeClustersAt(env: Env, at: number): Promise<Cluster[]> {
  const r = await env.DB.prepare("SELECT DISTINCT cluster_id FROM cluster_pods WHERE role = 'worker'").all<{ cluster_id: string }>();
  const out: Cluster[] = [];
  for (const x of r.results || []) {
    const c = await getCluster(env, x.cluster_id).catch(() => null);
    if (!c || !isEdge(c.spec)) continue;
    const win = await edgeWindow(env, c, await podsOf(env, c));
    if (win && at >= win.since && at <= win.until) out.push(c);
  }
  return out;
}

/** POST /api/clusters/:id/jobs/cancel-queued: every queued job (of one pool), one cancel each. */
export async function cancelQueued(env: Env, who: Actor, c: Cluster, o: { pool?: string; max?: number }) {
  const { db } = jobsDb(env, c);
  const pods = await podsOf(env, c);
  const where = scope(c, pods, { pool: o.pool }, isEdge(c.spec) ? await edgeWindow(env, c, pods) : null);
  const max = o.max ?? 100;
  const rows = await db
    .prepare(`SELECT id, external_id, protocol, status, model, worker FROM jobs WHERE ${where.sql} AND status = 'queued' ORDER BY created_at ASC LIMIT ?`)
    .bind(...where.args, max)
    .all<any>()
    .catch((e) => {
      if (noTable(e)) return { results: [] as any[] };
      throw e;
    });
  const list = rows.results || [];
  const results: CancelResult[] = [];
  // A few at a time: each is one internal call to a pod.
  for (let i = 0; i < list.length; i += 4) results.push(...(await Promise.all(list.slice(i, i + 4).map((r) => cancelJobRow(env, who, c, r, { audit: false })))));
  const failed = results.filter((r) => !r.ok);
  await audit(env, { actor: who.actor, ip: who.ip, action: "jobs.cancel-queued", target: c.name, ok: !failed.length, detail: failed.length ? `${failed.length} not cancelled: ${failed.slice(0, 3).map((f) => `${f.external_id}: ${f.note}`).join("; ")}` : undefined, after: { pool: o.pool ?? null, queued: list.length, cancelled: results.length - failed.length, failed: failed.length } });
  return { queued: list.length, cancelled: results.length - failed.length, failed: failed.map((f) => ({ job: f.external_id, api: f.api, note: f.note })), results };
}
