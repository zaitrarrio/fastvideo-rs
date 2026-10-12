// POST /ingest/v1/weights-plan: a pod or serverless worker with weights_source r2 asks for its weights (r2weights.ts,
// docs/ops/r2-weights.md §4 and §8).
// Auth, one of:
//  - a cluster's ingest token (FV_ENDPOINT_REPORT_TOKEN, as /ingest/v1/endpoint): the pod must be a live pod of that
//    cluster; the trees are its pool's (weightsPlan);
//  - a serverless endpoint's weights token (FV_WEIGHTS_TOKEN, serverless/weights.ts): the endpoint must be live and on
//    weights_source r2; the trees are its spec's (slsTrees). A request naming another endpoint is refused.
// Never what the request names. The answer is the trees' manifests with a presigned GET URL per file (short-lived):
// the pod gets no R2 key, and the URLs are never logged.
import { getCluster } from "./cluster/store";
import { sha256Hex } from "./crypto";
import type { Env } from "./env";
import { weightsPlan } from "./providers";
import { r2Cfg, r2Plan, r2Trees } from "./r2weights";
import { slsTokenRow } from "./serverless/ingest";
import { specOf } from "./serverless/ops";
import { slsFetchTrees } from "./serverless/weights";
import { HttpError, parseJson, rateLimit } from "./util";

async function planFor(env: Env, trees: string[]): Promise<Response> {
  const { cfg, off } = r2Cfg(env);
  if (!cfg) throw new HttpError(503, off!);
  let out;
  try {
    out = await r2Plan(env, cfg, trees);
  } catch (e) {
    throw new HttpError(502, (e as Error).message.slice(0, 300));
  }
  return new Response(JSON.stringify(out), { headers: { "content-type": "application/json", "cache-control": "no-store" } });
}

export async function weightsPlanRequest(env: Env, req: Request): Promise<Response> {
  const auth = req.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) throw new HttpError(401, "ingest token required");
  const token = auth.slice(7).trim();
  const text = await req.text();
  if (text.length > 4096) throw new HttpError(413, "request too large");
  const body = parseJson<{ pod?: unknown; endpoint?: unknown }>(text, {}) || {};
  if (token.startsWith("fvw_")) return slsPlan(env, token, body);
  const cl = await env.DB.prepare("SELECT id FROM clusters WHERE ingest_hash = ?").bind(await sha256Hex(token)).first<{ id: string }>();
  if (!cl) throw new HttpError(401, "invalid ingest token");
  if (!(await rateLimit(env, `weights-plan:${cl.id}`, 60, 60))) throw new HttpError(429, "weights plan rate limit (60/min per cluster)");
  const pod = body.pod;
  if (typeof pod !== "string" || !pod) throw new HttpError(400, "pod: the pod's id (FV_POD_ID)");
  const row = await env.DB.prepare("SELECT cluster_id, pool FROM cluster_pods WHERE pod_id = ? AND deleted_at IS NULL").bind(pod).first<{ cluster_id: string; pool: string | null }>();
  if (!row || row.cluster_id !== cl.id) throw new HttpError(404, "no live pod of this cluster with that id");
  const c = await getCluster(env, cl.id);
  const pool = c.spec.pools.find((p) => p.id === row.pool);
  if (!pool) throw new HttpError(409, `pool ${row.pool} is no longer in the cluster's spec`);
  const plan = weightsPlan(pool, env);
  if (plan.source !== "r2") throw new HttpError(409, `pool ${pool.id} does not use weights_source r2`);
  if (plan.unsupported.length) throw new HttpError(409, `not in the R2 weights store: ${plan.unsupported.join(", ")}`);
  return planFor(env, plan.trees);
}

/** A serverless worker's plan: its endpoint's trees, by the endpoint's weights token. */
async function slsPlan(env: Env, token: string, body: { endpoint?: unknown }): Promise<Response> {
  const row = await slsTokenRow(env, token);
  if (!row) throw new HttpError(401, "invalid weights token (unknown, rotated out, or its endpoint was deleted)");
  if (body.endpoint !== undefined && ![row.id, row.name, row.endpoint_id].includes(String(body.endpoint))) throw new HttpError(403, "this weights token is not for that endpoint");
  if (!(await rateLimit(env, `weights-plan:${row.id}`, 120, 60))) throw new HttpError(429, "weights plan rate limit (120/min per endpoint)");
  const spec = specOf(row);
  if (spec.weights_source !== "r2") throw new HttpError(409, `endpoint ${row.name} does not use weights_source r2`);
  const trees = slsFetchTrees(env, spec);
  const pub = r2Trees(env);
  const missing = trees.filter((t) => !pub[t]);
  if (missing.length) throw new HttpError(409, `not in the R2 weights store: ${missing.join(", ")}`);
  if (!trees.length) throw new HttpError(409, `endpoint ${row.name} loads no weight tree`);
  return planFor(env, trees);
}
