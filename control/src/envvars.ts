// Environment variables at four levels (docs/control/README.md "Env"):
// account ("Runpod level", every controller cluster), cluster, pool (every
// worker of one pool of one cluster: survives restarts, rolls and new pods
// of a scale-up; never the gateway), pod.
// Resolution: pod > pool > cluster > account > system (the controller's own
// keys, which the user layers may not set: RESERVED_KEYS). Secret values are
// sealed in D1 and masked in every response.
import { isReserved, SECRET_SYSTEM_KEYS } from "./cluster/payloads";
import { seal, unseal } from "./crypto";
import type { Env } from "./env";
import { HttpError, now } from "./util";

export type Scope = "account" | "cluster" | "pool" | "pod";
/** The scope_id of a pool's variables: `<cluster id>:<pool id>`. */
export const poolScopeId = (clusterId: string, pool: string) => `${clusterId}:${pool}`;
export interface VarRow {
  scope: Scope;
  scope_id: string;
  key: string;
  value: string;
  secret: number;
  updated_at: number;
  updated_by: string;
}
export interface EffectiveVar {
  key: string;
  value: string; // masked when secret
  secret: boolean;
  source: "system" | Scope;
  overrides?: string[]; // the lower layers it overrides
  runpod_secret_ref?: boolean;
}

export const MASK = "••••••••";
const KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
const aad = (scope: string, id: string, key: string) => `env:${scope}:${id}:${key}`;

export function validateVar(key: string, value: unknown) {
  if (!KEY_RE.test(key)) throw new HttpError(400, `invalid variable name: ${key}`);
  if (isReserved(key)) throw new HttpError(400, `${key} is set by the controller and cannot be overridden`);
  if (typeof value !== "string") throw new HttpError(400, "value must be a string");
  if (value.length > 32768) throw new HttpError(400, "value too long (32 KiB max)");
}

export async function listVars(env: Env, scope: Scope, scopeId: string): Promise<VarRow[]> {
  const r = await env.DB.prepare("SELECT * FROM env_vars WHERE scope = ? AND scope_id = ? ORDER BY key").bind(scope, scopeId).all<VarRow>();
  return r.results || [];
}
export function maskRow(r: VarRow) {
  return { scope: r.scope, scope_id: r.scope_id, key: r.key, value: r.secret ? MASK : r.value, secret: !!r.secret, updated_at: r.updated_at, updated_by: r.updated_by };
}

export async function setVar(env: Env, scope: Scope, scopeId: string, key: string, value: string, secret: boolean, by: string) {
  validateVar(key, value);
  const before = await env.DB.prepare("SELECT secret, value FROM env_vars WHERE scope = ? AND scope_id = ? AND key = ?").bind(scope, scopeId, key).first<{ secret: number; value: string }>();
  const stored = secret ? await seal(env.CONTROL_KEK, value, aad(scope, scopeId, key)) : value;
  await env.DB.prepare(
    `INSERT INTO env_vars (scope, scope_id, key, value, secret, updated_at, updated_by) VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (scope, scope_id, key) DO UPDATE SET value = excluded.value, secret = excluded.secret, updated_at = excluded.updated_at, updated_by = excluded.updated_by`,
  )
    .bind(scope, scopeId, key, stored, secret ? 1 : 0, now(), by)
    .run();
  const show = (s: { secret: number | boolean; value: string } | null) => (s ? { value: s.secret ? MASK : s.value, secret: !!s.secret } : null);
  return { before: show(before), after: show({ secret, value }) };
}
export async function deleteVar(env: Env, scope: Scope, scopeId: string, key: string) {
  const before = await env.DB.prepare("SELECT secret, value FROM env_vars WHERE scope = ? AND scope_id = ? AND key = ?").bind(scope, scopeId, key).first<{ secret: number; value: string }>();
  if (!before) throw new HttpError(404, "no such variable");
  await env.DB.prepare("DELETE FROM env_vars WHERE scope = ? AND scope_id = ? AND key = ?").bind(scope, scopeId, key).run();
  return { before: { value: before.secret ? MASK : before.value, secret: !!before.secret } };
}

async function plain(env: Env, r: VarRow): Promise<string> {
  return r.secret ? unseal(env.CONTROL_KEK, r.value, aad(r.scope, r.scope_id, r.key)) : r.value;
}

/** The user layers of one pod, lowest first (`pool`: a worker's pool; the gateway has none). */
async function layers(env: Env, clusterId: string, podId: string | null, pool?: string | null): Promise<{ scope: Scope; rows: VarRow[] }[]> {
  const out: { scope: Scope; rows: VarRow[] }[] = [
    { scope: "account", rows: await listVars(env, "account", "") },
    { scope: "cluster", rows: await listVars(env, "cluster", clusterId) },
  ];
  if (pool) out.push({ scope: "pool", rows: await listVars(env, "pool", poolScopeId(clusterId, pool)) });
  if (podId) out.push({ scope: "pod", rows: await listVars(env, "pod", podId) });
  return out;
}

/** The env a pod gets: system < account < cluster < pool < pod (plain values: only for Runpod payloads). */
export async function resolvePlain(env: Env, clusterId: string, podId: string | null, system: Record<string, string>, pool?: string | null): Promise<Record<string, string>> {
  const out = { ...system };
  for (const layer of await layers(env, clusterId, podId, pool)) for (const r of layer.rows) if (!isReserved(r.key)) out[r.key] = await plain(env, r);
  return out;
}

/** The same resolution for display: every value of a secret masked, with its source and what it overrides. */
export async function resolveView(env: Env, clusterId: string, podId: string | null, system: Record<string, string>, pool?: string | null): Promise<EffectiveVar[]> {
  const m = new Map<string, EffectiveVar>();
  for (const [k, v] of Object.entries(system)) {
    const secret = SECRET_SYSTEM_KEYS.has(k);
    m.set(k, { key: k, value: secret ? MASK : v.length > 200 ? `${v.slice(0, 60)}… (${v.length} chars)` : v, secret, source: "system", runpod_secret_ref: /^\{\{ RUNPOD_SECRET_/.test(v) });
  }
  for (const layer of await layers(env, clusterId, podId, pool)) {
    for (const r of layer.rows) {
      if (isReserved(r.key)) continue;
      const prev = m.get(r.key);
      m.set(r.key, {
        key: r.key,
        value: r.secret ? MASK : r.value,
        secret: !!r.secret,
        source: layer.scope,
        overrides: prev ? [...(prev.overrides || []), prev.source] : undefined,
        runpod_secret_ref: /^\{\{ RUNPOD_SECRET_/.test(r.value) && !r.secret,
      });
    }
  }
  return [...m.values()].sort((a, b) => a.key.localeCompare(b.key));
}
