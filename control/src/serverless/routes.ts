// /api/serverless/*: Runpod serverless endpoints fv-control manages
// (docs/control/serverless.md). Mounted by index.ts under the /api auth
// middleware (read tokens: GET only).
import { Hono, type Context } from "hono";
import { clientIp } from "../auth";
import type { Env, Vars } from "../env";
import { audit, HttpError, now, putSetting, scrub, utcDay } from "../util";
import { endpointView } from "./payloads";
import {
  captureLogs,
  createEndpoint,
  deleteEndpoint,
  extendEndpoint,
  getRow,
  invoke,
  jobStats,
  listRows,
  logSource,
  normalizeSlsPolicy,
  pollJob,
  recentJobs,
  rowView,
  scaleEndpoint,
  serverlessTick,
  slsPolicy,
  specOf,
  updateEndpoint,
  DEFAULT_SLS_POLICY,
} from "./ops";
import { sls } from "./runpod-sls";
import { parseOr400 } from "../schemas";
import { checkName } from "../names";
import { checkEndpointSpec, defaultEndpointSpec, normalizeEndpointSpec } from "./spec";

type App = { Bindings: Env; Variables: Vars };
type C = Context<App>;
export const serverlessRoutes = new Hono<App>();

const body = async <T = any>(c: C): Promise<T> => {
  try {
    return (await c.req.json()) as T;
  } catch {
    return {} as T;
  }
};
const who = (c: C) => ({ actor: c.get("actor"), ip: clientIp(c) });

serverlessRoutes.get("/", async (c) => {
  const rows = await listRows(c.env, c.req.query("all") === "1");
  // Account endpoints fv-control did not create: id and name only, never touched.
  let external: { id: string; name: string; workers: number }[] | null = null;
  if (c.req.query("external") === "1") {
    const ours = new Set(rows.map((r) => r.endpoint_id));
    external = (await sls.live(c.env)).endpoints.filter((e) => !ours.has(e.id)).map((e) => ({ id: e.id, name: e.name, workers: e.pods.length }));
  }
  const day = utcDay(now());
  const today = await c.env.DB.prepare("SELECT pod_id, usd FROM cost_daily WHERE day = ? AND pod_id LIKE 'sls:%'").bind(day).all<{ pod_id: string; usd: number }>();
  const todayBy = new Map((today.results || []).map((r) => [r.pod_id, r.usd]));
  return c.json({
    endpoints: rows.map((r) => ({ ...rowView(r), cost_today: todayBy.get(`sls:${r.endpoint_id}`) ?? 0 })),
    ...(external ? { external } : {}),
    policy: await slsPolicy(c.env),
  });
});
serverlessRoutes.get("/defaults", (c) => c.json({ spec: defaultEndpointSpec(c.req.query("name") || "example", c.req.query("variant") || "cpu") }));
serverlessRoutes.get("/policy", async (c) => c.json({ policy: await slsPolicy(c.env), defaults: DEFAULT_SLS_POLICY }));
serverlessRoutes.put("/policy", async (c) => {
  const before = await slsPolicy(c.env);
  const b = await body<{ policy?: object }>(c);
  const merged = { ...before, ...(b.policy || b) };
  parseOr400("serverless-policy", merged);
  const next = normalizeSlsPolicy(merged);
  await putSetting(c.env, "serverless", next, c.get("actor"));
  await audit(c.env, { ...who(c), action: "serverless.policy", before, after: next });
  return c.json({ policy: next });
});
/** Validates a spec as given (issues) and with the defaults filled (normalized). */
serverlessRoutes.post("/validate", async (c) => {
  const b = await body(c);
  const doc = b.spec ?? b;
  const raw = checkEndpointSpec(doc);
  // A new endpoint's name must be free (`id`: an existing endpoint being edited keeps its own).
  const nameIssue = typeof doc?.name === "string" && !b.id ? await checkName(c.env, "endpoint", doc.name).then((r) => (r.taken ? [{ path: ["name"], message: r.problem! }] : [])) : [];
  try {
    const spec = normalizeEndpointSpec(doc);
    if (nameIssue.length) return c.json({ ok: false, error: nameIssue[0]!.message, issues: nameIssue });
    return c.json({ ok: true, spec, raw_issues: raw.ok ? [] : raw.issues });
  } catch (e) {
    return c.json({ ok: false, error: (e as Error).message, issues: [...((e as HttpError).extra?.issues as any[] ?? []), ...nameIssue] });
  }
});
serverlessRoutes.post("/", async (c) => {
  const b = await body(c);
  const row = await createEndpoint(c.env, who(c), b.spec ?? b);
  return c.json({ endpoint: rowView(row) }, 201);
});
/** The tick by hand (also runs every minute from the cron): health, backstops, billing. */
serverlessRoutes.post("/tick", async (c) => c.json(await serverlessTick(c.env, { force_billing: true })));

serverlessRoutes.get("/:id", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  const live = row.endpoint_id && !row.deleted_at ? await sls.getEndpoint(c.env, row.endpoint_id).catch((e) => ({ error: scrub(c.env, (e as Error).message) })) : null;
  const health = row.endpoint_id && !row.deleted_at ? await sls.health(c.env, row.endpoint_id).catch((e) => ({ error: scrub(c.env, (e as Error).message) })) : null;
  const jobs = await recentJobs(c.env, row.id, 20);
  const costs = row.endpoint_id ? await c.env.DB.prepare("SELECT day, usd, minutes FROM cost_daily WHERE pod_id = ? ORDER BY day DESC LIMIT 30").bind(`sls:${row.endpoint_id}`).all<any>() : { results: [] };
  const audits = await c.env.DB.prepare("SELECT at, actor, action, ok, detail FROM audit WHERE target = ? AND action LIKE 'serverless.%' ORDER BY id DESC LIMIT 30").bind(row.name).all<any>();
  return c.json({
    endpoint: rowView(row),
    runpod: live && !(live as any).error ? endpointView(live) : live,
    health,
    jobs,
    stats: jobStats(jobs),
    costs: costs.results || [],
    audit: audits.results || [],
    log_source: logSource(row),
  });
});
serverlessRoutes.put("/:id", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  const b = await body(c);
  const spec = b.spec ?? b;
  // A partial document is merged over the current spec.
  return c.json({ endpoint: rowView(await updateEndpoint(c.env, who(c), row, { ...specOf(row), ...spec })) });
});
serverlessRoutes.post("/:id/scale", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  const raw = await body<{ workers_min?: number; workers_max?: number }>(c);
  const b = parseOr400("serverless-scale", { ...(raw.workers_min !== undefined ? { workers_min: raw.workers_min } : {}), ...(raw.workers_max !== undefined ? { workers_max: raw.workers_max } : {}) });
  return c.json({ endpoint: rowView(await scaleEndpoint(c.env, who(c), row, b)) });
});
serverlessRoutes.post("/:id/extend", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  const { minutes } = parseOr400("extend", { minutes: (await body(c)).minutes });
  return c.json({ endpoint: rowView(await extendEndpoint(c.env, who(c), row, minutes)) });
});
serverlessRoutes.delete("/:id", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  const r = await deleteEndpoint(c.env, who(c), row, "by hand");
  return c.json({ endpoint: rowView(r), deleted: r.status === "deleted" }, r.status === "deleted" ? 200 : 202);
});
serverlessRoutes.post("/:id/invoke", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  return c.json(await invoke(c.env, who(c), row, await body(c)));
});
serverlessRoutes.get("/:id/jobs", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  const jobs = await recentJobs(c.env, row.id, Math.min(Number(c.req.query("limit") || 50), 200));
  return c.json({ jobs, stats: jobStats(jobs) });
});
serverlessRoutes.get("/:id/jobs/:job", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  return c.json({ job: await pollJob(c.env, row, Number(c.req.param("job"))) });
});
/** A worker's Runpod log tail; also stored in the log store (log_lines, cluster_id serverless:<id>, pod_id = worker). */
serverlessRoutes.get("/:id/logs", async (c) => {
  const row = await getRow(c.env, c.req.param("id"));
  if (!row.endpoint_id || row.deleted_at) throw new HttpError(409, `endpoint is ${row.status}`);
  const live = await sls.getEndpoint(c.env, row.endpoint_id);
  const workers: string[] = Array.isArray(live?.workers) ? live.workers.map((w: any) => String(w.id)) : [];
  const worker = c.req.query("worker") || workers[0];
  if (!worker) return c.json({ workers, worker: null, container: [], system: [], note: "no worker is up: logs exist only while a worker runs (stored ones: /api/logs?pod=<worker>)" });
  if (!workers.includes(worker)) {
    // A worker that already stopped: what the log store kept.
    const kept = await c.env.DB.prepare("SELECT ts, level, target, msg FROM log_lines WHERE cluster_id = ? AND pod_id = ? ORDER BY ts LIMIT 2000").bind(logSource(row), worker).all<any>();
    if (!(kept.results || []).length) throw new HttpError(404, "not a worker of this endpoint");
    return c.json({ workers, worker, stored: kept.results });
  }
  const r = await captureLogs(c.env, row, worker);
  return c.json({ workers, worker, ...r });
});
