import { randomToken } from "./crypto";
import type { Env } from "./env";

export const now = () => Date.now();
export const newId = (prefix: string) => `${prefix}_${randomToken("", 8)}`;
export const utcDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);

export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
    public extra?: Record<string, unknown>,
  ) {
    super(message);
  }
}

export function parseJson<T>(s: string | null | undefined, fallback: T): T {
  if (!s) return fallback;
  try {
    return JSON.parse(s) as T;
  } catch {
    return fallback;
  }
}

/** Removes every secret this Worker holds from a string (error messages from upstreams). */
export function scrub(env: Env, s: string): string {
  let out = s;
  for (const v of [env.RUNPOD_API_KEY, env.CLOUDRIFT_API_KEY, env.CLOUDFLARE_API_KEY, env.GITHUB_PAT, env.GITHUB_RUNNER_PAT, env.BUILD_CACHE_R2_SECRET_ACCESS_KEY, env.CONTROL_KEK, env.SESSION_SECRET]) {
    if (v && v.length >= 8) out = out.split(v).join("[redacted]");
  }
  return out;
}

/** Audit log: every mutating action (who, what, when, before/after; never a secret value). */
export async function audit(
  env: Env,
  a: { actor: string; action: string; target?: string; before?: unknown; after?: unknown; ok?: boolean; detail?: string; ip?: string },
): Promise<void> {
  const j = (v: unknown) => (v === undefined ? null : scrub(env, JSON.stringify(v)).slice(0, 20000));
  await env.DB.prepare(
    "INSERT INTO audit (at, actor, action, target, before, after, ok, detail, ip) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
  )
    .bind(now(), a.actor, a.action, a.target ?? null, j(a.before), j(a.after), a.ok === false ? 0 : 1, a.detail ? scrub(env, a.detail).slice(0, 2000) : null, a.ip ?? null)
    .run();
}

export async function getSetting<T>(env: Env, key: string, fallback: T): Promise<T> {
  const r = await env.DB.prepare("SELECT value FROM settings WHERE key = ?").bind(key).first<{ value: string }>();
  return r ? { ...(fallback as object), ...parseJson<object>(r.value, {}) } as T : fallback;
}
export async function putSetting(env: Env, key: string, value: unknown, by: string): Promise<void> {
  await env.DB.prepare(
    "INSERT INTO settings (key, value, updated_at, updated_by) VALUES (?, ?, ?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at, updated_by = excluded.updated_by",
  )
    .bind(key, JSON.stringify(value), now(), by)
    .run();
}

/** Fixed-window rate limit in D1: true when the call is allowed. */
export async function rateLimit(env: Env, key: string, limit: number, windowS: number): Promise<boolean> {
  const w = Math.floor(now() / 1000 / windowS);
  const r = await env.DB.prepare(
    "INSERT INTO rate_limits (key, window, count) VALUES (?, ?, 1) ON CONFLICT (key, window) DO UPDATE SET count = count + 1 RETURNING count",
  )
    .bind(key, w)
    .first<{ count: number }>();
  return (r?.count ?? 1) <= limit;
}

export async function fetchWithTimeout(input: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<Response> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), init.timeoutMs ?? 20000);
  try {
    return await fetch(input, { ...init, signal: ctl.signal });
  } finally {
    clearTimeout(t);
  }
}
