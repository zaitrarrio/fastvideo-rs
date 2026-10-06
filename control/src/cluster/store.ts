// Clusters in D1: spec, state and sealed secrets.
import { seal, unseal } from "../crypto";
import type { Env } from "../env";
import { HttpError, now, parseJson } from "../util";
import type { ClusterSecrets, ClusterState, PodRec } from "./payloads";
import { migrateSpec, type ClusterSpec } from "./spec";

export interface ClusterRow {
  id: string;
  name: string;
  spec: string;
  state: string;
  status: string;
  deadline: number | null;
  secrets: string | null;
  ingest_hash: string | null;
  source: string;
  created_at: number;
  updated_at: number;
  created_by: string;
}
export interface Cluster {
  id: string;
  name: string;
  spec: ClusterSpec;
  state: ClusterState;
  status: string;
  deadline: number | null; // ms
  source: string;
  created_at: number;
  updated_at: number;
  created_by: string;
  sealedSecrets: string | null;
}

export const emptyState = (): ClusterState => ({ images: {}, workers: {} });

export function fromRow(r: ClusterRow): Cluster {
  return {
    id: r.id,
    name: r.name,
    spec: migrateSpec(parseJson<ClusterSpec>(r.spec, {} as ClusterSpec)),
    state: { ...emptyState(), ...parseJson<ClusterState>(r.state, emptyState()) },
    status: r.status,
    deadline: r.deadline,
    source: r.source,
    created_at: r.created_at,
    updated_at: r.updated_at,
    created_by: r.created_by,
    sealedSecrets: r.secrets,
  };
}

export async function getCluster(env: Env, idOrName: string): Promise<Cluster> {
  const r = await env.DB.prepare("SELECT * FROM clusters WHERE id = ? OR name = ?").bind(idOrName, idOrName).first<ClusterRow>();
  if (!r) throw new HttpError(404, `no cluster ${idOrName}`);
  return fromRow(r);
}
export async function listClusters(env: Env): Promise<Cluster[]> {
  const r = await env.DB.prepare("SELECT * FROM clusters ORDER BY created_at DESC").all<ClusterRow>();
  return (r.results || []).map(fromRow);
}

export async function secretsOf(env: Env, c: Cluster): Promise<ClusterSecrets> {
  if (!c.sealedSecrets) throw new HttpError(500, `cluster ${c.name} has no secrets`);
  return JSON.parse(await unseal(env.CONTROL_KEK, c.sealedSecrets, `cluster:${c.id}`)) as ClusterSecrets;
}
export async function saveSecrets(env: Env, c: Cluster, s: ClusterSecrets): Promise<void> {
  c.sealedSecrets = await seal(env.CONTROL_KEK, JSON.stringify(s), `cluster:${c.id}`);
  await env.DB.prepare("UPDATE clusters SET secrets = ?, updated_at = ? WHERE id = ?").bind(c.sealedSecrets, now(), c.id).run();
}
export async function saveState(env: Env, c: Cluster, patch: { status?: string; deadline?: number | null } = {}): Promise<void> {
  if (patch.status !== undefined) c.status = patch.status;
  if (patch.deadline !== undefined) c.deadline = patch.deadline;
  c.updated_at = now();
  await env.DB.prepare("UPDATE clusters SET state = ?, status = ?, deadline = ?, updated_at = ? WHERE id = ?")
    .bind(JSON.stringify(c.state), c.status, c.deadline, c.updated_at, c.id)
    .run();
}
export async function saveSpec(env: Env, c: Cluster): Promise<void> {
  await env.DB.prepare("UPDATE clusters SET spec = ?, updated_at = ? WHERE id = ?").bind(JSON.stringify(c.spec), now(), c.id).run();
}

export async function recordPod(env: Env, c: Cluster, rec: PodRec, role: "gateway" | "worker", slot: string, envHash: string | null) {
  await env.DB.prepare(
    `INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, slot, image, gpu, dc, cost_per_hr, url, env_hash, env_applied_at, created_at, status)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'creating')
     ON CONFLICT (pod_id) DO UPDATE SET slot = excluded.slot, image = excluded.image, env_hash = excluded.env_hash, env_applied_at = excluded.env_applied_at`,
  )
    .bind(rec.pod, c.id, role, rec.pool ?? null, slot, rec.image, rec.gpu ?? (rec.cpu ? `cpu:${rec.cpu}` : null), rec.dc ?? null, rec.dph, rec.url ?? null, envHash, envHash ? now() : null, rec.created * 1000)
    .run();
}
export async function podUpdate(env: Env, podId: string, f: { status?: string; env_hash?: string; slot?: string; image?: string; ready?: boolean; deleted?: boolean }) {
  const sets: string[] = [];
  const vals: unknown[] = [];
  if (f.status) sets.push("status = ?"), vals.push(f.status);
  if (f.env_hash) sets.push("env_hash = ?", "env_applied_at = ?"), vals.push(f.env_hash, now());
  if (f.slot) sets.push("slot = ?"), vals.push(f.slot);
  if (f.image) sets.push("image = ?"), vals.push(f.image);
  if (f.ready) sets.push("ready_at = COALESCE(ready_at, ?)"), vals.push(now());
  if (f.deleted) sets.push("deleted_at = ?", "status = 'deleted'"), vals.push(now());
  if (!sets.length) return;
  await env.DB.prepare(`UPDATE cluster_pods SET ${sets.join(", ")} WHERE pod_id = ?`).bind(...vals, podId).run();
}
export async function livePods(env: Env, clusterId: string) {
  const r = await env.DB.prepare("SELECT * FROM cluster_pods WHERE cluster_id = ? AND deleted_at IS NULL ORDER BY role DESC, pool, created_at").bind(clusterId).all<any>();
  return r.results || [];
}

/** Every pod record in the state (gateway first). */
export function allPods(state: ClusterState): { rec: PodRec; role: "gateway" | "worker"; slot: string }[] {
  const out: { rec: PodRec; role: "gateway" | "worker"; slot: string }[] = [];
  if (state.gateway) out.push({ rec: state.gateway, role: "gateway", slot: "gateway" });
  for (const l of Object.values(state.workers || {})) for (const r of l) out.push({ rec: r, role: "worker", slot: "workers" });
  for (const l of Object.values(state.rolling || {})) for (const r of l) out.push({ rec: r, role: "worker", slot: "rolling" });
  for (const r of state.retired || []) out.push({ rec: r, role: "worker", slot: "retired" });
  return out;
}
