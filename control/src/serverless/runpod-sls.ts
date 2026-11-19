// Runpod's serverless APIs (docs/control/serverless.md "Runpod API"):
//   REST v2 api.runpod.io/v2   /templates, /serverless (+ /<id>/workers), /billing/serverless, /catalog/gpus
//                              (REST v1, rest.runpod.io/v1, is retired: /endpoints, /billing/endpoints are gone)
//   queue   api.runpod.ai/v2    /<id>/health, /run, /runsync, /status/<job>, /cancel/<job>, /purge-queue
//   LB      https://<id>.api.runpod.ai/<path>
//   GraphQL myself { clientBalance endpoints { … pods } } (live workers and their $/hr)
// The API key only goes in an Authorization header; every upstream text is
// scrubbed. Nothing here returns an endpoint's or a template's env (Runpod
// returns it in clear): views pick named fields (payloads.ts endpointView).
import { runpodErrorText } from "../runpoderr";
import { defaults, type Env } from "../env";
import { runpod } from "../runpod";
import { fetchWithTimeout, HttpError, scrub } from "../util";

/** Upstream bases; tests point them at a mock. The REST base is `RUNPOD_REST` (env.ts defaults.runpodRest; its older
 * name `RUNPOD_REST2` still works). */
export type SlsEnv = Env & { RUNPOD_QUEUE?: string; RUNPOD_LB_URL?: string };
export const slsBases = (e: SlsEnv) => ({
  rest: defaults.runpodRest(e),
  queue: e.RUNPOD_QUEUE || "https://api.runpod.ai/v2",
  lb: (id: string) => (e.RUNPOD_LB_URL || "https://{id}.api.runpod.ai").replaceAll("{id}", id),
});

const enc = encodeURIComponent;
/** Runpod answers a template or endpoint that is already gone with 404, or with 400/500 and "… not found" (live
 * 2026-10-06 on v1: a template Runpod had deleted with its endpoint). */
export const isGone = (e: unknown) => e instanceof HttpError && (e.status === 404 || /not found/i.test(e.message));

async function raw(env: Env, url: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<{ status: number; body: any; ms: number }> {
  const t0 = Date.now();
  const r = await fetchWithTimeout(url, {
    ...init,
    headers: { authorization: `Bearer ${env.RUNPOD_API_KEY}`, "content-type": "application/json", ...(init.headers || {}) },
  });
  const text = await r.text();
  let body: any = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { raw: scrub(env, text.slice(0, 2000)) };
  }
  return { status: r.status, body, ms: Date.now() - t0 };
}
async function ok(env: Env, url: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<any> {
  const r = await raw(env, url, init);
  if (r.status < 200 || r.status >= 300) {
    const b = r.body;
    const msg = runpodErrorText(b, typeof init.body === "string" ? init.body : null);
    throw new HttpError(r.status >= 500 ? 502 : r.status === 401 || r.status === 403 ? 502 : r.status, `runpod: ${scrub(env, String(msg))}`, { upstream: r.status });
  }
  return r.body;
}
const json = (b: unknown) => (b === undefined ? undefined : JSON.stringify(b));

export interface LiveEndpoint {
  id: string;
  name: string;
  type?: string; // QB | LB
  workersMin?: number;
  workersMax?: number;
  pods: { id: string; desiredStatus?: string; costPerHr?: number }[];
}

/** A worker of REST v2 `GET /serverless/<id>/workers` (its id is the pod's). */
export interface SlsWorker {
  id: string;
  status: string; // RUNNING | IDLE | INITIALIZING | THROTTLED | UNHEALTHY
  isStale?: boolean;
  version?: number | null;
  gpuCount?: number;
  image?: string | null;
  uptimeSeconds?: number | null;
  gpuTypeId?: string | null;
  dataCenterId?: string | null;
  startedAt?: string | null;
}

export const sls = {
  // ---- templates and endpoints (REST v2)
  createTemplate: (env: Env, payload: unknown) => runpod.rest(env, "POST", "/templates", payload),
  updateTemplate: (env: Env, id: string, payload: unknown) => runpod.rest(env, "PATCH", `/templates/${enc(id)}`, payload),
  /** A template, or null when Runpod has none with that id. */
  async getTemplate(env: Env, id: string): Promise<any | null> {
    try {
      const t = await runpod.rest(env, "GET", `/templates/${enc(id)}`);
      return t && t.id ? t : null;
    } catch (e) {
      if (isGone(e) || (e instanceof HttpError && e.status === 400)) return null;
      throw e;
    }
  },
  async deleteTemplate(env: Env, id: string): Promise<boolean> {
    try {
      await runpod.rest(env, "DELETE", `/templates/${enc(id)}`);
      return true;
    } catch (e) {
      if (isGone(e)) return true;
      throw e;
    }
  },
  /** `POST /serverless`: a queue or load-balancer endpoint (payloads.ts endpointCreatePayload). */
  createEndpoint: (env: Env, payload: unknown) => runpod.rest(env, "POST", "/serverless", payload),
  /** `PATCH /serverless/<id>`: scaling, placement, the GPU block, or `templateId` to re-apply a template's container
   * settings (v2 keeps no link to the template). */
  patchEndpoint: (env: Env, id: string, body: unknown) => runpod.rest(env, "PATCH", `/serverless/${enc(id)}`, body),
  /** The live workers of an endpoint (`GET /serverless/<id>/workers`; v1's `?includeWorkers=true`). */
  async workers(env: Env, id: string): Promise<SlsWorker[]> {
    const r = await runpod.rest(env, "GET", `/serverless/${enc(id)}/workers`);
    return Array.isArray(r?.workers) ? r.workers : [];
  },
  /** An endpoint with its workers (`workerList`: v2 names the min / max block `workers`), or null when Runpod has
   * none with that id. */
  async getEndpoint(env: Env, id: string): Promise<any | null> {
    let e: any;
    try {
      e = await runpod.rest(env, "GET", `/serverless/${enc(id)}`);
    } catch (e) {
      if (e instanceof HttpError && (e.status === 404 || e.status === 400)) return null;
      throw e;
    }
    if (!e || !e.id) return null;
    const workerList = await sls.workers(env, id).catch((err) => {
      if (isGone(err)) return [] as SlsWorker[];
      throw err;
    });
    return { ...e, workerList };
  },
  async deleteEndpoint(env: Env, id: string): Promise<boolean> {
    try {
      await runpod.rest(env, "DELETE", `/serverless/${enc(id)}`);
      return true;
    } catch (e) {
      if (isGone(e)) return true;
      throw e;
    }
  },
  /** Serverless spend per endpoint and UTC day since `startIso` (REST v2 `GET /billing/serverless`: one record per
   * endpoint per bucket, `serverlessId`, `startTime`, `totalAmount`; v1's `/billing/endpoints` is a different product
   * in v2, Runpod's public endpoints). v2 reports no billed time (v1's `timeBilledMs`): `timeBilledMs` is null. */
  async billing(env: Env, startIso: string): Promise<{ endpointId: string; time: string; amount: number; timeBilledMs: number | null }[]> {
    const r = await runpod.rest(env, "GET", `/billing/serverless?bucketSize=day&startTime=${enc(startIso)}`);
    const recs: any[] = Array.isArray(r?.records) ? r.records : [];
    return recs.filter((x) => x && x.serverlessId).map((x) => ({ endpointId: String(x.serverlessId), time: String(x.startTime || ""), amount: Number(x.totalAmount || 0), timeBilledMs: null }));
  },
  /** The GPU catalog: every type's serverless pool (`GET /catalog/gpus`; null when the type is in no pool). */
  async gpuCatalog(env: SlsEnv): Promise<{ id: string; pool: string | null }[]> {
    const j = await ok(env, `${slsBases(env).rest}/catalog/gpus`, { timeoutMs: 20000 });
    return Array.isArray(j?.gpus) ? j.gpus.map((g: any) => ({ id: String(g.id), pool: g.pool ?? null })) : [];
  },

  // ---- the queue API (api.runpod.ai/v2/<id>)
  health: (env: SlsEnv, id: string) => ok(env, `${slsBases(env).queue}/${enc(id)}/health`, { timeoutMs: 15000 }),
  run: (env: SlsEnv, id: string, body: unknown) => ok(env, `${slsBases(env).queue}/${enc(id)}/run`, { method: "POST", body: json(body), timeoutMs: 30000 }),
  /** Runpod holds /runsync up to ~90 s, then answers with the job still IN_QUEUE / IN_PROGRESS. */
  runsync: (env: SlsEnv, id: string, body: unknown) => ok(env, `${slsBases(env).queue}/${enc(id)}/runsync`, { method: "POST", body: json(body), timeoutMs: 100000 }),
  status: (env: SlsEnv, id: string, job: string) => ok(env, `${slsBases(env).queue}/${enc(id)}/status/${enc(job)}`, { timeoutMs: 20000 }),
  /** Cancels a queued job, or tells the worker running it to stop (its job-stop long poll): {id, status}. */
  cancel: (env: SlsEnv, id: string, job: string) => ok(env, `${slsBases(env).queue}/${enc(id)}/cancel/${enc(job)}`, { method: "POST", timeoutMs: 20000 }),
  /** Drops every queued job (running ones are not touched): {removed, status}. */
  purge: (env: SlsEnv, id: string) => ok(env, `${slsBases(env).queue}/${enc(id)}/purge-queue`, { method: "POST", timeoutMs: 30000 }),

  // ---- the load balancer (https://<id>.api.runpod.ai)
  lb: (env: SlsEnv, id: string, method: string, path: string, body?: unknown) =>
    raw(env, `${slsBases(env).lb(id)}${path}`, { method, body: body === undefined || method === "GET" ? undefined : json(body), timeoutMs: 150000 }),

  // ---- GraphQL: the balance and every endpoint's live workers in one call
  async live(env: Env): Promise<{ balance: number; endpoints: LiveEndpoint[] }> {
    const d = await runpod.gql<any>(env, "{ myself { clientBalance endpoints { id name type workersMin workersMax pods { id desiredStatus costPerHr } } } }");
    return { balance: Number(d.myself.clientBalance), endpoints: (d.myself.endpoints || []).map((e: any) => ({ ...e, pods: e.pods || [] })) };
  },
  /** The container / system log tail Runpod keeps for a worker (the pods' log endpoint, hapi). */
  logs: (env: Env, workerId: string) => runpod.logs(env, workerId),
};

export const balanceFloor = (env: Env) => defaults.balanceFloor(env);
