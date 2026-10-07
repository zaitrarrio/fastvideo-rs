// Controller auth (docs/control/README.md "Auth"). Two modes:
// - Cloudflare Access: ACCESS_TEAM_DOMAIN + ACCESS_AUD set. Every request
//   must carry a valid Access JWT (RS256, the team's JWKS, aud, iss, exp),
//   and the identity must be in OWNER_EMAILS when that is set.
// - Owner passphrase (fallback): PBKDF2 hash in OWNER_PASSPHRASE_HASH, a
//   signed session cookie (D1-backed, revocable), CSRF token on every
//   mutating request, Origin check, D1 rate limits on login.
// API tokens (Bearer fvc_…, SHA-256 in D1, scope read | admin) work in the
// passphrase mode for scripts and agents; behind Access they need an Access
// service token too (the Access JWT is always checked first).
import type { Context, MiddlewareHandler } from "hono";
import { b64url, hmacB64url, randomToken, safeEqual, sha256Hex, unb64, verifyPassphrase } from "./crypto";
import type { Env, Vars } from "./env";
import { audit, HttpError, now, rateLimit } from "./util";

type C = Context<{ Bindings: Env; Variables: Vars }>;

export const SESSION_COOKIE = "__Host-fvc_session";
export const SESSION_TTL_MS = 12 * 3600 * 1000;

export const accessMode = (env: Env) => !!(env.ACCESS_TEAM_DOMAIN && env.ACCESS_AUD);

// ---------------- Cloudflare Access JWT
let jwksCache: { url: string; at: number; keys: JsonWebKey[] } | null = null;
async function accessKeys(team: string): Promise<JsonWebKey[]> {
  const url = `${team.replace(/\/$/, "")}/cdn-cgi/access/certs`;
  if (jwksCache && jwksCache.url === url && now() - jwksCache.at < 600_000) return jwksCache.keys;
  const r = await fetch(url);
  if (!r.ok) throw new HttpError(503, "Access certs unavailable");
  const j = (await r.json()) as { keys: JsonWebKey[] };
  jwksCache = { url, at: now(), keys: j.keys };
  return j.keys;
}
function b64urlJson(s: string): any {
  return JSON.parse(new TextDecoder().decode(unb64(s)));
}
export async function verifyAccessJwt(env: Env, token: string): Promise<{ email: string; sub: string }> {
  const parts = token.split(".");
  if (parts.length !== 3) throw new HttpError(401, "bad Access token");
  const [h, p, s] = parts as [string, string, string];
  const header = b64urlJson(h);
  const payload = b64urlJson(p);
  if (header.alg !== "RS256") throw new HttpError(401, "bad Access token alg");
  const keys = await accessKeys(env.ACCESS_TEAM_DOMAIN!);
  const jwk = keys.find((k: any) => k.kid === header.kid);
  if (!jwk) throw new HttpError(401, "unknown Access key");
  const key = await crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
  const ok = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", key, unb64(s), new TextEncoder().encode(`${h}.${p}`));
  if (!ok) throw new HttpError(401, "bad Access signature");
  const aud = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
  if (!aud.includes(env.ACCESS_AUD)) throw new HttpError(401, "Access aud mismatch");
  if (payload.iss && payload.iss.replace(/\/$/, "") !== env.ACCESS_TEAM_DOMAIN!.replace(/\/$/, "")) throw new HttpError(401, "Access iss mismatch");
  if (typeof payload.exp !== "number" || payload.exp * 1000 < now()) throw new HttpError(401, "Access token expired");
  const email = String(payload.email || payload.common_name || payload.sub || "");
  const allowed = (env.OWNER_EMAILS || "").split(",").map((x) => x.trim().toLowerCase()).filter(Boolean);
  if (allowed.length && !allowed.includes(email.toLowerCase())) throw new HttpError(403, "not an owner");
  return { email, sub: String(payload.sub || "") };
}

// ---------------- sessions (passphrase mode)
async function signSession(env: Env, id: string, exp: number) {
  return `${id}.${exp}.${await hmacB64url(env.SESSION_SECRET, `s1|${id}|${exp}`)}`;
}
export async function csrfFor(env: Env, sessionId: string) {
  return hmacB64url(env.SESSION_SECRET, `csrf|${sessionId}`);
}
function cookie(c: C, name: string): string | undefined {
  const h = c.req.header("cookie") || "";
  for (const part of h.split(/;\s*/)) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i) === name) return decodeURIComponent(part.slice(i + 1));
  }
  return undefined;
}
async function sessionFromCookie(c: C): Promise<{ id: string; actor: string } | null> {
  const v = cookie(c, SESSION_COOKIE);
  if (!v) return null;
  const [id, expS, sig] = v.split(".");
  if (!id || !expS || !sig) return null;
  const exp = Number(expS);
  if (!(exp > now())) return null;
  const want = await hmacB64url(c.env.SESSION_SECRET, `s1|${id}|${exp}`);
  if (!safeEqual(want, sig)) return null;
  const row = await c.env.DB.prepare("SELECT actor, expires_at, revoked_at FROM sessions WHERE id = ?").bind(id).first<{
    actor: string;
    expires_at: number;
    revoked_at: number | null;
  }>();
  if (!row || row.revoked_at || row.expires_at < now()) return null;
  return { id, actor: row.actor };
}
export function clientIp(c: C): string {
  return c.req.header("cf-connecting-ip") || c.req.header("x-forwarded-for")?.split(",")[0]?.trim() || "local";
}

export async function login(c: C, passphrase: string): Promise<{ csrf: string }> {
  const env = c.env;
  if (accessMode(env)) throw new HttpError(400, "this controller uses Cloudflare Access; no passphrase login");
  if (!env.OWNER_PASSPHRASE_HASH) throw new HttpError(503, "no owner passphrase configured (OWNER_PASSPHRASE_HASH)");
  const ip = clientIp(c);
  const okIp = await rateLimit(env, `login:${ip}`, 5, 900);
  const okAll = await rateLimit(env, "login:*", 30, 3600);
  if (!okIp || !okAll) {
    await audit(env, { actor: `anon:${ip}`, action: "auth.login", ok: false, detail: "rate limited", ip });
    throw new HttpError(429, "too many login attempts; try again later");
  }
  const ok = typeof passphrase === "string" && passphrase.length > 0 && passphrase.length < 1024 && (await verifyPassphrase(passphrase, env.SESSION_SECRET, env.OWNER_PASSPHRASE_HASH));
  if (!ok) {
    await audit(env, { actor: `anon:${ip}`, action: "auth.login", ok: false, detail: "bad passphrase", ip });
    throw new HttpError(401, "wrong passphrase");
  }
  const id = randomToken("", 16);
  const exp = now() + SESSION_TTL_MS;
  await env.DB.prepare("INSERT INTO sessions (id, actor, created_at, expires_at, ip, ua) VALUES (?, 'owner', ?, ?, ?, ?)")
    .bind(id, now(), exp, ip, (c.req.header("user-agent") || "").slice(0, 200))
    .run();
  await env.DB.prepare("DELETE FROM sessions WHERE expires_at < ?").bind(now() - 86400_000).run();
  c.header("set-cookie", `${SESSION_COOKIE}=${await signSession(env, id, exp)}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=${SESSION_TTL_MS / 1000}`);
  await audit(env, { actor: "owner", action: "auth.login", target: `session:${id.slice(0, 6)}`, ip });
  return { csrf: await csrfFor(env, id) };
}
export async function logout(c: C): Promise<void> {
  const s = await sessionFromCookie(c);
  if (s) {
    await c.env.DB.prepare("UPDATE sessions SET revoked_at = ? WHERE id = ?").bind(now(), s.id).run();
    await audit(c.env, { actor: s.actor, action: "auth.logout", target: `session:${s.id.slice(0, 6)}`, ip: clientIp(c) });
  }
  c.header("set-cookie", `${SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0`);
}

// ---------------- API tokens
export async function mintApiToken(env: Env, name: string, scope: "read" | "admin" | "ci", by: string, ttlDays?: number) {
  const token = randomToken("fvc_", 32);
  const id = `tok_${randomToken("", 6)}`;
  const exp = ttlDays ? now() + ttlDays * 86400_000 : null;
  await env.DB.prepare("INSERT INTO api_tokens (id, name, hash, scope, created_at, created_by, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
    .bind(id, name, await sha256Hex(token), scope, now(), by, exp)
    .run();
  return { token, id, name, scope, expires_at: exp };
}
async function apiTokenActor(env: Env, bearer: string): Promise<{ actor: string; scope: "read" | "admin" | "ci" } | null> {
  if (!bearer.startsWith("fvc_")) return null;
  const row = await env.DB.prepare("SELECT id, name, scope, expires_at, revoked_at FROM api_tokens WHERE hash = ?")
    .bind(await sha256Hex(bearer))
    .first<{ id: string; name: string; scope: "read" | "admin" | "ci"; expires_at: number | null; revoked_at: number | null }>();
  if (!row || row.revoked_at || (row.expires_at && row.expires_at < now())) return null;
  await env.DB.prepare("UPDATE api_tokens SET last_used_at = ? WHERE id = ?").bind(now(), row.id).run();
  return { actor: `token:${row.name}`, scope: row.scope };
}

const MUTATING = new Set(["POST", "PUT", "PATCH", "DELETE"]);

/** Guards /api/*: sets actor/authKind/scope, enforces CSRF + Origin on cookie sessions and read-only tokens. */
export const requireAuth: MiddlewareHandler<{ Bindings: Env; Variables: Vars }> = async (c, next) => {
  const env = c.env;
  const method = c.req.method.toUpperCase();
  if (accessMode(env)) {
    const jwt = c.req.header("cf-access-jwt-assertion") || cookie(c, "CF_Authorization");
    if (!jwt) throw new HttpError(401, "Cloudflare Access token missing");
    const id = await verifyAccessJwt(env, jwt);
    c.set("actor", `access:${id.email}`);
    c.set("authKind", "access");
    c.set("scope", "admin");
    return next();
  }
  const auth = c.req.header("authorization") || "";
  if (auth.toLowerCase().startsWith("bearer ")) {
    const t = await apiTokenActor(env, auth.slice(7).trim());
    if (!t) throw new HttpError(401, "invalid API token");
    // A `ci` token (a GitHub workflow's secret) reaches only /api/ci/* (docs/dev/build-pods-fv-control.md §6).
    if (t.scope === "ci") {
      if (!c.req.path.startsWith("/api/ci/")) throw new HttpError(403, "a ci token only reaches /api/ci/*");
    } else if (MUTATING.has(method) && t.scope !== "admin") throw new HttpError(403, "read-only token");
    c.set("actor", t.actor);
    c.set("authKind", "token");
    c.set("scope", t.scope);
    return next();
  }
  const s = await sessionFromCookie(c);
  if (!s) throw new HttpError(401, "login required");
  if (MUTATING.has(method)) {
    const origin = c.req.header("origin");
    if (origin && origin !== new URL(c.req.url).origin) throw new HttpError(403, "cross-origin request refused");
    const got = c.req.header("x-csrf-token") || "";
    if (!safeEqual(got, await csrfFor(env, s.id))) throw new HttpError(403, "CSRF token missing or wrong");
  }
  c.set("actor", s.actor);
  c.set("authKind", "session");
  c.set("sessionId", s.id);
  c.set("scope", "admin");
  return next();
};

/**
 * Guards /serverless/<endpoint>/*, the serverless console (src/serverless/console.ts). Same identities as
 * /api (Access, an fvc_ token, the session cookie), but the pages are fv-serve's console and send no CSRF
 * token: a cookie session's mutating request needs an Origin header naming this origin instead (the
 * console's own fetches send it; the cookie is SameSite=Strict too). Any other Authorization header (an
 * fv-serve API key a browser kept) is ignored: the proxy holds no fv-serve keys. A page without a session
 * goes to the dashboard's login.
 */
export const requireConsoleAuth: MiddlewareHandler<{ Bindings: Env; Variables: Vars }> = async (c, next) => {
  const env = c.env;
  const method = c.req.method.toUpperCase();
  const page = method === "GET" && /\/console(\/|$)/.test(c.req.path) && !c.req.path.includes("/console/assets/");
  const refuse = (status: number, message: string) => (page && status === 401 ? c.redirect("/#/serverless", 302) : c.json({ error: { kind: status === 401 ? "unauthorized" : "forbidden", message } }, status as 401));
  if (accessMode(env)) {
    const jwt = c.req.header("cf-access-jwt-assertion") || cookie(c, "CF_Authorization");
    if (!jwt) return refuse(401, "Cloudflare Access token missing");
    try {
      const id = await verifyAccessJwt(env, jwt);
      c.set("actor", `access:${id.email}`);
    } catch (e) {
      return refuse(e instanceof HttpError ? e.status : 401, (e as Error).message);
    }
    c.set("authKind", "access");
    c.set("scope", "admin");
    return next();
  }
  const auth = c.req.header("authorization") || "";
  if (/^bearer fvc_/i.test(auth)) {
    const t = await apiTokenActor(env, auth.slice(7).trim());
    if (!t || t.scope === "ci") return refuse(401, "invalid API token");
    c.set("actor", t.actor);
    c.set("authKind", "token");
    c.set("scope", t.scope);
    return next();
  }
  const s = await sessionFromCookie(c);
  if (!s) return refuse(401, "login required (fv-control)");
  if (MUTATING.has(method)) {
    const origin = c.req.header("origin");
    if (!origin || origin !== new URL(c.req.url).origin) return refuse(403, "cross-origin request refused (no Origin header naming this origin)");
    const site = c.req.header("sec-fetch-site");
    if (site && site !== "same-origin") return refuse(403, "cross-site request refused");
  }
  c.set("actor", s.actor);
  c.set("authKind", "session");
  c.set("sessionId", s.id);
  c.set("scope", "admin");
  return next();
};

export async function whoami(c: C) {
  const env = c.env;
  if (accessMode(env)) {
    const jwt = c.req.header("cf-access-jwt-assertion") || cookie(c, "CF_Authorization");
    if (!jwt) return { authenticated: false, mode: "access" };
    try {
      const id = await verifyAccessJwt(env, jwt);
      return { authenticated: true, mode: "access", actor: `access:${id.email}` };
    } catch {
      return { authenticated: false, mode: "access" };
    }
  }
  const s = await sessionFromCookie(c);
  if (!s) return { authenticated: false, mode: "passphrase", configured: !!env.OWNER_PASSPHRASE_HASH };
  return { authenticated: true, mode: "passphrase", actor: s.actor, csrf: await csrfFor(env, s.id) };
}

export const _test = { signSession, b64url };
