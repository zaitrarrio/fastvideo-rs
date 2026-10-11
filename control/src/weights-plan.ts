// POST /ingest/v1/weights-plan: a pod with weights_source r2 asks for its weights (r2weights.ts, docs/ops/r2-weights.md §4).
// Auth: its cluster's ingest token (FV_ENDPOINT_REPORT_TOKEN, as /ingest/v1/endpoint). The pod must be a live pod of
// that cluster; the trees are its pool's (weightsPlan), never what the request names. The answer is the trees'
// manifests with a presigned GET URL per file (short-lived): the pod gets no R2 key, and the URLs are never logged.
import { getCluster } from "./cluster/store";
import { sha256Hex } from "./crypto";
import type { Env } from "./env";
import { weightsPlan } from "./providers";
import { r2Cfg, r2Plan } from "./r2weights";
import { HttpError, parseJson, rateLimit } from "./util";

export async function weightsPlanRequest(env: Env, req: Request): Promise<Response> {
  const auth = req.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) throw new HttpError(401, "ingest token required");
  const cl = await env.DB.prepare("SELECT id FROM clusters WHERE ingest_hash = ?").bind(await sha256Hex(auth.slice(7).trim())).first<{ id: string }>();
  if (!cl) throw new HttpError(401, "invalid ingest token");
  if (!(await rateLimit(env, `weights-plan:${cl.id}`, 60, 60))) throw new HttpError(429, "weights plan rate limit (60/min per cluster)");
  const text = await req.text();
  if (text.length > 4096) throw new HttpError(413, "request too large");
  const pod = parseJson<{ pod?: unknown }>(text, {})?.pod;
  if (typeof pod !== "string" || !pod) throw new HttpError(400, "pod: the pod's id (FV_POD_ID)");
  const row = await env.DB.prepare("SELECT cluster_id, pool FROM cluster_pods WHERE pod_id = ? AND deleted_at IS NULL").bind(pod).first<{ cluster_id: string; pool: string | null }>();
  if (!row || row.cluster_id !== cl.id) throw new HttpError(404, "no live pod of this cluster with that id");
  const c = await getCluster(env, cl.id);
  const pool = c.spec.pools.find((p) => p.id === row.pool);
  if (!pool) throw new HttpError(409, `pool ${row.pool} is no longer in the cluster's spec`);
  const plan = weightsPlan(pool, env);
  if (plan.source !== "r2") throw new HttpError(409, `pool ${pool.id} does not use weights_source r2`);
  if (plan.unsupported.length) throw new HttpError(409, `not in the R2 weights store: ${plan.unsupported.join(", ")}`);
  const { cfg, off } = r2Cfg(env);
  if (!cfg) throw new HttpError(503, off!);
  let out;
  try {
    out = await r2Plan(env, cfg, plan.trees);
  } catch (e) {
    throw new HttpError(502, (e as Error).message.slice(0, 300));
  }
  return new Response(JSON.stringify(out), { headers: { "content-type": "application/json", "cache-control": "no-store" } });
}
