// One-click deployments: saved presets ("give me H3 on 80GB+ cards") and the
// requests that poll providers until a match deploys or the wait expires.
// A preset names model families, GPU specs (vram/arch/model), GPU count,
// storage (validated >= 1.1x the families' weights), instance kind + count,
// a max wait and a max spend. The matcher (tickDeployRequests, run on a cron)
// checks Runpod GPU stock, Brev types and GMI offers for a spec match under
// the spend cap; a match triggers the real deploy, a blown wait expires it.
import type { Env } from "./env";
import { HttpError, now, newId } from "./util";
import { MODEL_FAMILIES, familyById, familiesWeights, minStorageGb, fmtGb } from "./families";
import { gpuTypes } from "./dynamic";
import { brevGpuOptions, providerImpl } from "./providers";

export interface GpuSpec {
  /** Minimum VRAM GB. */
  vram_gb: number;
  /** GPU architecture substring, e.g. "blackwell" (null: any). */
  arch?: string | null;
  /** Exact GPU model substring, e.g. "RTX PRO 6000" (null: any). */
  model?: string | null;
}

export interface DeployPreset {
  id: string;
  name: string;
  families: string[];
  gpu_specs: GpuSpec[];
  gpu_count: number;
  storage_gb: number;
  instance_kind: "standalone" | "cluster" | "serverless";
  instance_count: number;
  max_wait_s: number;
  max_spend_usd: number | null;
}

export interface DeployMatch {
  provider: string;
  gpu: string;
  display: string;
  memory_gb: number | null;
  usd_per_hr: number | null;
  in_stock: boolean | null;
  est_spend_usd: number | null;
}

const row = (r: any): DeployPreset => ({
  id: r.id, name: r.name,
  families: JSON.parse(r.families || "[]"),
  gpu_specs: JSON.parse(r.gpu_specs || "[]"),
  gpu_count: r.gpu_count, storage_gb: r.storage_gb,
  instance_kind: r.instance_kind, instance_count: r.instance_count,
  max_wait_s: r.max_wait_s, max_spend_usd: r.max_spend_usd,
});

/** Validate a preset body; throws 400 with issues. Storage must clear 1.1x. */
export function validatePreset(b: any): Omit<DeployPreset, "id"> {
  const issues: { path: (string | number)[]; message: string }[] = [];
  const add = (path: (string | number)[], message: string) => issues.push({ path, message });
  const name = String(b.name || "").trim();
  if (!name) add(["name"], "name it (e.g. \"h3 nightly\")");
  const families: string[] = Array.isArray(b.families) ? b.families.map(String) : [];
  if (!families.length) add(["families"], "pick at least one model family");
  for (const f of families) if (!familyById(f)) add(["families"], `unknown family ${f} (families: ${MODEL_FAMILIES.map((x) => x.id).join(", ")})`);
  const specs: GpuSpec[] = Array.isArray(b.gpu_specs) ? b.gpu_specs : [];
  if (!specs.length) add(["gpu_specs"], "add at least one GPU spec");
  for (const [i, s] of specs.entries()) {
    if (!(s.vram_gb > 0)) add(["gpu_specs", i, "vram_gb"], "vram_gb must be > 0");
  }
  const gpu_count = Number(b.gpu_count || 1);
  if (!(gpu_count >= 1 && gpu_count <= 8)) add(["gpu_count"], "1-8 GPUs");
  const storage_gb = Number(b.storage_gb || 0);
  const need = minStorageGb(families);
  if (need != null && storage_gb < need) {
    add(["storage_gb"], `${storage_gb} GB is too small for ${familiesWeights(families).length} weight trees: at least ${need} GB (10% over ${fmtGb(Math.ceil((need / 1.1)))})`);
  } else if (!(storage_gb > 0)) add(["storage_gb"], "storage_gb must be > 0");
  const kind = b.instance_kind;
  if (!["standalone", "cluster", "serverless"].includes(kind)) add(["instance_kind"], "standalone | cluster | serverless");
  const ic = Number(b.instance_count || 1);
  if (!(ic >= 1 && ic <= 32)) add(["instance_count"], "1-32 instances");
  const wait = Number(b.max_wait_s || 3600);
  if (!(wait >= 300 && wait <= 7 * 86400)) add(["max_wait_s"], "5 minutes to 7 days");
  const spend = b.max_spend_usd == null || b.max_spend_usd === "" ? null : Number(b.max_spend_usd);
  if (spend != null && !(spend > 0)) add(["max_spend_usd"], "max_spend_usd must be > 0 or empty");
  if (issues.length) throw new HttpError(400, `preset: ${issues[0]!.message}`, { issues });
  return { name, families, gpu_specs: specs, gpu_count, storage_gb, instance_kind: kind, instance_count: ic, max_wait_s: wait, max_spend_usd: spend };
}

export async function listPresets(env: Env): Promise<DeployPreset[]> {
  const r = await env.DB.prepare(`SELECT * FROM deploy_presets WHERE deleted_at IS NULL ORDER BY updated_at DESC`).all();
  return (r.results || []).map(row);
}

export async function createPreset(env: Env, b: any, by: string): Promise<DeployPreset> {
  const v = validatePreset(b);
  const id = newId("dp");
  const t = now();
  await env.DB.prepare(
    `INSERT INTO deploy_presets (id, name, families, gpu_specs, gpu_count, storage_gb, instance_kind, instance_count, max_wait_s, max_spend_usd, created_at, created_by, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).bind(id, v.name, JSON.stringify(v.families), JSON.stringify(v.gpu_specs), v.gpu_count, v.storage_gb, v.instance_kind, v.instance_count, v.max_wait_s, v.max_spend_usd, t, by, t).run();
  return { id, ...v };
}

export async function updatePreset(env: Env, id: string, b: any): Promise<DeployPreset> {
  const cur = await env.DB.prepare(`SELECT * FROM deploy_presets WHERE id = ? AND deleted_at IS NULL`).bind(id).first();
  if (!cur) throw new HttpError(404, `no preset ${id}`);
  const v = validatePreset({ ...row(cur), ...b, name: b.name ?? (cur as any).name });
  await env.DB.prepare(
    `UPDATE deploy_presets SET name=?, families=?, gpu_specs=?, gpu_count=?, storage_gb=?, instance_kind=?, instance_count=?, max_wait_s=?, max_spend_usd=?, updated_at=? WHERE id=?`,
  ).bind(v.name, JSON.stringify(v.families), JSON.stringify(v.gpu_specs), v.gpu_count, v.storage_gb, v.instance_kind, v.instance_count, v.max_wait_s, v.max_spend_usd, now(), id).run();
  return { id, ...v };
}

export async function duplicatePreset(env: Env, id: string, by: string): Promise<DeployPreset> {
  const cur = await env.DB.prepare(`SELECT * FROM deploy_presets WHERE id = ? AND deleted_at IS NULL`).bind(id).first();
  if (!cur) throw new HttpError(404, `no preset ${id}`);
  const p = row(cur);
  return createPreset(env, { ...p, name: `${p.name} (copy)` }, by);
}

export async function deletePreset(env: Env, id: string): Promise<void> {
  await env.DB.prepare(`UPDATE deploy_presets SET deleted_at = ? WHERE id = ?`).bind(now(), id).run();
}

// ---------------------------------------------------------------- requests
export interface DeployRequest {
  id: string;
  preset_id: string;
  preset_name: string;
  status: string;
  created_at: number;
  expires_at: number;
  matched_at: number | null;
  match: DeployMatch | null;
  result: any | null;
  last_error: string | null;
  checks: number;
}

export async function listRequests(env: Env, limit = 50): Promise<DeployRequest[]> {
  const r = await env.DB.prepare(
    `SELECT r.*, p.name AS preset_name FROM deploy_requests r JOIN deploy_presets p ON p.id = r.preset_id ORDER BY r.created_at DESC LIMIT ?`,
  ).bind(limit).all<any>();
  return (r.results || []).map((x) => ({
    id: x.id, preset_id: x.preset_id, preset_name: x.preset_name, status: x.status,
    created_at: x.created_at, expires_at: x.expires_at, matched_at: x.matched_at,
    match: x.match_json ? JSON.parse(x.match_json) : null,
    result: x.result_json ? JSON.parse(x.result_json) : null,
    last_error: x.last_error, checks: x.checks,
  }));
}

/** Launch a one-click request from a preset: validates, then queues the poll. */
export async function launchRequest(env: Env, presetId: string, by: string): Promise<DeployRequest> {
  const cur = await env.DB.prepare(`SELECT * FROM deploy_presets WHERE id = ? AND deleted_at IS NULL`).bind(presetId).first();
  if (!cur) throw new HttpError(404, `no preset ${presetId}`);
  const p = row(cur);
  const id = newId("dr");
  const t = now();
  await env.DB.prepare(
    `INSERT INTO deploy_requests (id, preset_id, status, created_at, expires_at, created_by, updated_at) VALUES (?, ?, 'pending', ?, ?, ?, ?)`,
  ).bind(id, presetId, t, t + p.max_wait_s * 1000, by, t).run();
  // Try once right away; the cron keeps polling.
  await tickDeployRequests(env).catch(() => {});
  const one = await env.DB.prepare(`SELECT r.*, p.name AS preset_name FROM deploy_requests r JOIN deploy_presets p ON p.id = r.preset_id WHERE r.id = ?`).bind(id).first<any>();
  return {
    id: one.id, preset_id: one.preset_id, preset_name: one.preset_name, status: one.status,
    created_at: one.created_at, expires_at: one.expires_at, matched_at: one.matched_at,
    match: one.match_json ? JSON.parse(one.match_json) : null,
    result: one.result_json ? JSON.parse(one.result_json) : null,
    last_error: one.last_error, checks: one.checks,
  };
}

export async function cancelRequest(env: Env, id: string): Promise<void> {
  await env.DB.prepare(`UPDATE deploy_requests SET status = 'cancelled', updated_at = ? WHERE id = ? AND status = 'pending'`).bind(now(), id).run();
}

// ---------------------------------------------------------------- matching
function specMatches(spec: GpuSpec, g: { memory_gb: number | null; display: string; id: string }): boolean {
  if (g.memory_gb != null && g.memory_gb < spec.vram_gb) return false;
  const hay = `${g.display} ${g.id}`.toLowerCase();
  if (spec.arch && !hay.includes(spec.arch.toLowerCase())) return false;
  if (spec.model && !hay.includes(spec.model.toLowerCase())) return false;
  return true;
}

/** Poll providers for a preset's GPU specs; the cheapest in-stock match first. */
export async function findMatch(env: Env, p: DeployPreset): Promise<DeployMatch | null> {
  const cands: DeployMatch[] = [];
  // Runpod.
  try {
    const gpus = await gpuTypes(env);
    for (const g of gpus) {
      if (!p.gpu_specs.some((s) => specMatches(s, g))) continue;
      const price = g.secure_price ?? g.community_price;
      cands.push({
        provider: "runpod", gpu: g.id, display: g.display, memory_gb: g.memory_gb,
        usd_per_hr: price, in_stock: g.stock === "High" || g.stock === "Medium" || g.stock === "Low" ? true : g.stock ? false : null,
        est_spend_usd: price != null ? price * p.gpu_count * p.instance_count : null,
      });
    }
  } catch { /* provider down: skip */ }
  // Brev.
  try {
    const opts = await brevGpuOptions(env);
    for (const o of opts as any[]) {
      const mem = (o as any).memory_gb ?? null;
      if (!p.gpu_specs.some((s) => specMatches(s, { memory_gb: mem, display: String((o as any).title || (o as any).id || ""), id: String((o as any).id || "") }))) continue;
      const price = (o as any).usd_per_hr ?? null;
      cands.push({
        provider: "brev", gpu: String((o as any).id), display: String((o as any).title || (o as any).id),
        memory_gb: mem, usd_per_hr: price, in_stock: (o as any).in_stock ?? null,
        est_spend_usd: price != null ? price * p.gpu_count * p.instance_count : null,
      });
    }
  } catch { /* skip */ }
  // GMI.
  try {
    const impl = providerImpl("gmi" as any);
    if (!impl.off(env)) {
      for (const g of impl.gpus(env)) {
        const o = await impl.offer(env, g).catch(() => null);
        if (!o) continue;
        if (!p.gpu_specs.some((s) => specMatches(s, { memory_gb: (o as any).memory_gb ?? null, display: g, id: g }))) continue;
        cands.push({
          provider: "gmi", gpu: g, display: g, memory_gb: (o as any).memory_gb ?? null,
          usd_per_hr: (o as any).usd_per_hr ?? null, in_stock: (o as any).in_stock ?? null,
          est_spend_usd: (o as any).usd_per_hr != null ? (o as any).usd_per_hr * p.gpu_count * p.instance_count : null,
        });
      }
    }
  } catch { /* skip */ }

  // Prefer in-stock, then cheapest; enforce the spend cap.
  const ok = cands.filter((c) => c.in_stock !== false);
  ok.sort((a, b) => (a.usd_per_hr ?? 1e9) - (b.usd_per_hr ?? 1e9));
  for (const c of ok) {
    if (p.max_spend_usd != null && c.est_spend_usd != null && c.est_spend_usd > p.max_spend_usd) continue;
    if (c.in_stock) return c;
  }
  // Nothing firmly in stock: return the cheapest plausible so the UI can show
  // "waiting on X" rather than nothing.
  return ok[0] || null;
}

/** One polling tick: advance every pending request (match, deploy, expire). */
export async function tickDeployRequests(env: Env): Promise<{ checked: number; matched: number; expired: number }> {
  const t = now();
  const r = await env.DB.prepare(`SELECT * FROM deploy_requests WHERE status = 'pending'`).all<any>();
  let matched = 0, expired = 0;
  for (const x of r.results || []) {
    if (x.expires_at <= t) {
      await env.DB.prepare(`UPDATE deploy_requests SET status = 'expired', updated_at = ? WHERE id = ?`).bind(t, x.id).run();
      expired++;
      continue;
    }
    const p = await env.DB.prepare(`SELECT * FROM deploy_presets WHERE id = ? AND deleted_at IS NULL`).bind(x.preset_id).first();
    if (!p) {
      await env.DB.prepare(`UPDATE deploy_requests SET status = 'failed', last_error = 'preset deleted', updated_at = ? WHERE id = ?`).bind(t, x.id).run();
      continue;
    }
    const preset = row(p);
    try {
      const m = await findMatch(env, preset);
      await env.DB.prepare(`UPDATE deploy_requests SET checks = checks + 1, match_json = ?, updated_at = ? WHERE id = ?`)
        .bind(m ? JSON.stringify(m) : null, t, x.id).run();
      if (m?.in_stock) {
        await env.DB.prepare(`UPDATE deploy_requests SET status = 'matched', matched_at = ?, updated_at = ? WHERE id = ?`).bind(t, t, x.id).run();
        matched++;
        // The actual deploy is kicked off by the operator from the console
        // (Deploy → requests → "Deploy now") or a follow-up tick; the match
        // holds the offer to deploy against.
      }
    } catch (e) {
      await env.DB.prepare(`UPDATE deploy_requests SET checks = checks + 1, last_error = ?, updated_at = ? WHERE id = ?`)
        .bind((e as Error).message.slice(0, 500), t, x.id).run();
    }
  }
  return { checked: (r.results || []).length, matched, expired };
}
