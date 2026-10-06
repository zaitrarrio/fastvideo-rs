// Runpod's serverless APIs (docs/control/serverless.md "Runpod API"):
//   REST v1 rest.runpod.io/v1   /templates, /endpoints, /billing/endpoints
//   REST v2 api.runpod.io/v2    /serverless (load-balancer create), /catalog/gpus
//   queue   api.runpod.ai/v2    /<id>/health, /run, /runsync, /status/<job>, /cancel/<job>
//   LB      https://<id>.api.runpod.ai/<path>
//   GraphQL myself { clientBalance endpoints { … pods } } (live workers and their $/hr)
// The API key only goes in an Authorization header; every upstream text is
// scrubbed. Nothing here returns a template's env (Runpod returns it in
// clear): views pick named fields (payloads.ts endpointView).
import { defaults, type Env } from "../env";
import { runpod } from "../runpod";
import { fetchWithTimeout, HttpError, scrub } from "../util";

/** Upstream bases; tests point them at a mock. */
export type SlsEnv = Env & { RUNPOD_REST2?: string; RUNPOD_QUEUE?: string; RUNPOD_LB_URL?: string };
export const slsBases = (e: SlsEnv) => ({
  rest2: e.RUNPOD_REST2 || "https://api.runpod.io/v2",
  queue: e.RUNPOD_QUEUE || "https://api.runpod.ai/v2",
  lb: (id: string) => (e.RUNPOD_LB_URL || "https://{id}.api.runpod.ai").replaceAll("{id}", id),
});

const enc = encodeURIComponent;
/** Runpod answers a template or endpoint that is already gone with 404, or with 400/500 and "… not found"
 * (a v2 endpoint's template, which Runpod deletes with the endpoint: live 2026-10-06). */
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
    const msg = typeof b === "object" && b ? b.error || b.message || JSON.stringify(b).slice(0, 300) : String(b).slice(0, 300);
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

export const sls = {
  // ---- templates and endpoints (REST v1)
  createTemplate: (env: Env, payload: unknown) => runpod.rest(env, "POST", "/templates", payload),
  updateTemplate: (env: Env, id: string, payload: unknown) => runpod.rest(env, "PATCH", `/templates/${enc(id)}`, payload),
  async deleteTemplate(env: Env, id: string): Promise<boolean> {
    try {
      await runpod.rest(env, "DELETE", `/templates/${enc(id)}`);
      return true;
    } catch (e) {
      if (isGone(e)) return true;
      throw e;
    }
  },
  createEndpoint: (env: Env, payload: unknown) => runpod.rest(env, "POST", "/endpoints", payload),
  patchEndpoint: (env: Env, id: string, body: unknown) => runpod.rest(env, "PATCH", `/endpoints/${enc(id)}`, body),
  /** An endpoint with its workers, or null when Runpod has none with that id. */
  async getEndpoint(env: Env, id: string): Promise<any | null> {
    try {
      const e = await runpod.rest(env, "GET", `/endpoints/${enc(id)}?includeWorkers=true&includeTemplate=true`);
      return e && e.id ? e : null;
    } catch (e) {
      if (e instanceof HttpError && (e.status === 404 || e.status === 400)) return null;
      throw e;
    }
  },
  async deleteEndpoint(env: Env, id: string): Promise<boolean> {
    try {
      await runpod.rest(env, "DELETE", `/endpoints/${enc(id)}`);
      return true;
    } catch (e) {
      if (isGone(e)) return true;
      throw e;
    }
  },
  /** Serverless spend per endpoint and UTC day since `startIso` (REST v1 /billing/endpoints). */
  async billing(env: Env, startIso: string): Promise<{ endpointId: string; time: string; amount: number; timeBilledMs: number }[]> {
    const r = await runpod.rest(env, "GET", `/billing/endpoints?bucketSize=day&grouping=endpointId&startTime=${enc(startIso)}`);
    return (Array.isArray(r) ? r : []).filter((x: any) => x && x.endpointId).map((x: any) => ({ endpointId: String(x.endpointId), time: String(x.time || ""), amount: Number(x.amount || 0), timeBilledMs: Number(x.timeBilledMs || 0) }));
  },

  // ---- REST v2: load-balancer endpoints, the GPU catalog
  createLb: (env: SlsEnv, payload: unknown) => ok(env, `${slsBases(env).rest2}/serverless`, { method: "POST", body: json(payload), timeoutMs: 60000 }),
  async gpuCatalog(env: SlsEnv): Promise<{ id: string; pool: string | null }[]> {
    const j = await ok(env, `${slsBases(env).rest2}/catalog/gpus`, { timeoutMs: 20000 });
    return Array.isArray(j?.gpus) ? j.gpus.map((g: any) => ({ id: String(g.id), pool: g.pool ?? null })) : [];
  },

  // ---- the queue API (api.runpod.ai/v2/<id>)
  health: (env: SlsEnv, id: string) => ok(env, `${slsBases(env).queue}/${enc(id)}/health`, { timeoutMs: 15000 }),
  run: (env: SlsEnv, id: string, body: unknown) => ok(env, `${slsBases(env).queue}/${enc(id)}/run`, { method: "POST", body: json(body), timeoutMs: 30000 }),
  /** Runpod holds /runsync up to ~90 s, then answers with the job still IN_QUEUE / IN_PROGRESS. */
  runsync: (env: SlsEnv, id: string, body: unknown) => ok(env, `${slsBases(env).queue}/${enc(id)}/runsync`, { method: "POST", body: json(body), timeoutMs: 100000 }),
  status: (env: SlsEnv, id: string, job: string) => ok(env, `${slsBases(env).queue}/${enc(id)}/status/${enc(job)}`, { timeoutMs: 20000 }),
  cancel: (env: SlsEnv, id: string, job: string) => ok(env, `${slsBases(env).queue}/${enc(id)}/cancel/${enc(job)}`, { method: "POST", timeoutMs: 20000 }),

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
