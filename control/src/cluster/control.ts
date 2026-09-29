// Starting and cancelling cluster operations (API, cron auto-actions).
import type { Env } from "../env";
import { HttpError, newId, now } from "../util";
import type { OpKind } from "./do";

export function stubFor(env: Env, clusterId: string) {
  return env.CLUSTER_OPS.get(env.CLUSTER_OPS.idFromName(clusterId));
}

export async function startOp(env: Env, clusterId: string, kind: OpKind, params: unknown, actor: string): Promise<{ id: string }> {
  const id = newId("op");
  const t = now();
  await env.DB.prepare("INSERT INTO operations (id, cluster_id, kind, status, params, actor, created_at, updated_at) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)")
    .bind(id, clusterId, kind, JSON.stringify(params ?? {}), actor, t, t)
    .run();
  const r = await stubFor(env, clusterId).fetch("https://ops/op/start", {
    method: "POST",
    body: JSON.stringify({ id, cluster: clusterId, kind, params: params ?? {}, actor }),
  });
  if (!r.ok) {
    const j = (await r.json().catch(() => ({}))) as { error?: string };
    await env.DB.prepare("UPDATE operations SET status = 'failed', error = ?, updated_at = ? WHERE id = ?").bind(j.error || `HTTP ${r.status}`, now(), id).run();
    throw new HttpError(r.status === 409 ? 409 : 502, j.error || `could not start ${kind}`);
  }
  return { id };
}

export async function cancelOp(env: Env, clusterId: string, actor: string) {
  const r = await stubFor(env, clusterId).fetch("https://ops/op/cancel", { method: "POST", body: JSON.stringify({ actor }) });
  return r.json();
}
export async function currentOp(env: Env, clusterId: string) {
  const r = await stubFor(env, clusterId).fetch("https://ops/op");
  return ((await r.json()) as { op: unknown }).op;
}
