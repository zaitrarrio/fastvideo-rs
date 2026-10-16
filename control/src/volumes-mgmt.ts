// Runpod network volumes: create in any region, list, delete, and track
// costs. Runpod bills network volumes per GB-month (the GraphQL
// `networkVolumes` view carries the price); fv-control records spend in D1
// (volume_costs) so the console can show unit prices and totals.
// GraphQL docs: saveNetworkVolume / deleteNetworkVolume mutations.
import type { Env } from "./env";
import { runpod } from "./runpod";
import { now } from "./util";

export interface RunpodVolume {
  id: string;
  name: string;
  size: number; // GB
  dataCenterId: string;
  createdAt?: string;
}

const LIST_Q = `{
  myself {
    networkVolumes {
      id name size dataCenterId
    }
  }
}`;

export async function listRunpodVolumes(env: Env): Promise<RunpodVolume[]> {
  const d = await runpod.gql<any>(env, LIST_Q);
  return (d?.myself?.networkVolumes || []).map((v: any) => ({
    id: String(v.id),
    name: String(v.name || v.id),
    size: Number(v.size) || 0,
    dataCenterId: String(v.dataCenterId || ""),
  }));
}

/** Create a network volume in a region (size GB). Returns the volume id. */
export async function createRunpodVolume(env: Env, o: { name: string; size: number; dataCenterId: string }): Promise<string> {
  const d = await runpod.gql<any>(
    env,
    `mutation($input: SaveNetworkVolumeInput!) { saveNetworkVolume(input: $input) { id name size dataCenterId } }`,
    { input: { name: o.name, size: o.size, dataCenterId: o.dataCenterId } },
  );
  const v = d?.saveNetworkVolume;
  if (!v?.id) throw new Error("runpod did not return a volume id");
  await env.DB.prepare(
    `INSERT INTO volumes (id, name, provider, size_gb, region, status, created_at, created_by) VALUES (?, ?, 'runpod', ?, ?, 'active', ?, ?)`,
  ).bind(String(v.id), String(v.name || o.name), o.size, o.dataCenterId, now(), "console").run();
  return String(v.id);
}

/** Delete a network volume (Runpod + the D1 record). */
export async function deleteRunpodVolume(env: Env, id: string): Promise<void> {
  await runpod.gql(env, `mutation($id: String!) { deleteNetworkVolume(id: $id) }`, { id });
  await env.DB.prepare(`UPDATE volumes SET status = 'deleted', deleted_at = ? WHERE id = ?`).bind(now(), id).run();
}

/** Record a cost sample (called by the collector): unit price and total. */
export async function recordVolumeCost(env: Env, o: { volume_id: string; usd_per_gb_mo: number; size_gb: number; at?: number }): Promise<void> {
  const at = o.at ?? now();
  await env.DB.prepare(
    `INSERT INTO volume_costs (volume_id, usd_per_gb_mo, size_gb, month_usd, at) VALUES (?, ?, ?, ?, ?)`,
  ).bind(o.volume_id, o.usd_per_gb_mo, o.size_gb, (o.usd_per_gb_mo * o.size_gb), at).run();
}

/** The latest known unit price per volume (USD/GB-month). */
export async function volumePrices(env: Env): Promise<Record<string, { usd_per_gb_mo: number; size_gb: number; at: number }>> {
  const r = await env.DB.prepare(
    `SELECT volume_id, usd_per_gb_mo, size_gb, at FROM volume_costs WHERE (volume_id, at) IN (SELECT volume_id, MAX(at) FROM volume_costs GROUP BY volume_id)`,
  ).all<{ volume_id: string; usd_per_gb_mo: number; size_gb: number; at: number }>();
  const out: Record<string, { usd_per_gb_mo: number; size_gb: number; at: number }> = {};
  for (const row of r.results || []) out[row.volume_id] = { usd_per_gb_mo: row.usd_per_gb_mo, size_gb: row.size_gb, at: row.at };
  return out;
}

/** Volumes fv-control tracks, joined with their latest price. */
export async function trackedVolumes(env: Env): Promise<any[]> {
  const r = await env.DB.prepare(`SELECT * FROM volumes ORDER BY created_at DESC`).all<any>();
  const prices = await volumePrices(env);
  return (r.results || []).map((v) => ({ ...v, price: prices[v.id] || null }));
}
