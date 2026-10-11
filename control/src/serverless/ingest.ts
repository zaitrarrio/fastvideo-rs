// What a weights_source r2 serverless worker may ask fv-control with its endpoint's weights token
// (serverless/weights.ts): its trees' plan (weights-plan.ts) and its boot reports (POST /ingest/v1/sls-report:
// the cold start's breakdown, serverless_boots, and a line per report in the log store).
import { sha256Hex } from "../crypto";
import type { Env } from "../env";
import { HttpError, now, parseJson, rateLimit, scrub } from "../util";
import { logSource, type SlsRow } from "./ops";
import { applyReport, SLS_TOKEN_GRACE_MS, type BootRow } from "./weights";

/** The live endpoint a weights token belongs to (its current one, or the previous one within the rotation grace); null
 * for any other token (revoked by a delete, expired, or never ours). */
export async function slsTokenRow(env: Env, token: string, at = now()): Promise<SlsRow | null> {
  if (!/^fvw_[0-9a-f]{64}$/.test(token)) return null;
  const h = await sha256Hex(token);
  return env.DB.prepare(
    "SELECT * FROM serverless_endpoints WHERE deleted_at IS NULL AND status IN ('creating', 'active', 'scaled-down') AND (weights_token_hash = ? OR (weights_token_prev_hash = ? AND weights_token_rotated_at > ?)) LIMIT 1",
  )
    .bind(h, h, at - SLS_TOKEN_GRACE_MS)
    .first<SlsRow>();
}

const BOOT_RE = /^([A-Za-z0-9_.-]{1,64})@(\d{9,11})$/;
const PHASES = new Set(["boot", "cache", "weights", "serve", "failed"]);

/** POST /ingest/v1/sls-report: {pod: "<worker>@<container start>", phase: boot | cache | weights | serve | failed, detail}. */
export async function slsReport(env: Env, req: Request): Promise<{ ok: true }> {
  const auth = req.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) throw new HttpError(401, "weights token required");
  const row = await slsTokenRow(env, auth.slice(7).trim());
  if (!row) throw new HttpError(401, "invalid weights token");
  if (!(await rateLimit(env, `sls-report:${row.id}`, 600, 60))) throw new HttpError(429, "report rate limit (600/min per endpoint)");
  const text = await req.text();
  if (text.length > 4096) throw new HttpError(413, "request too large");
  const b = parseJson<{ pod?: unknown; phase?: unknown; detail?: unknown }>(text, {}) || {};
  const m = typeof b.pod === "string" ? BOOT_RE.exec(b.pod) : null;
  if (!m) throw new HttpError(400, "pod: <worker id>@<container start, unix s>");
  const phase = String(b.phase || "");
  if (!PHASES.has(phase)) throw new HttpError(400, `phase: one of ${[...PHASES].join(", ")}`);
  const detail = scrub(env, String(b.detail ?? "")).slice(0, 300);
  const t = now();
  const prev = (await env.DB.prepare("SELECT * FROM serverless_boots WHERE endpoint = ? AND boot = ?").bind(row.id, b.pod).first<BootRow>()) ?? { worker_id: m[1]!, boot: b.pod as string, boot_at: t };
  const next = applyReport(prev, phase, detail, t) as BootRow;
  await env.DB.prepare(
    `INSERT INTO serverless_boots (endpoint, worker_id, boot, boot_at, cache, missing, fetch_start, fetch_end, fetch_bytes, fetch_s, fetch_mbps, serve_at, error, last_detail, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (endpoint, boot) DO UPDATE SET cache = excluded.cache, missing = excluded.missing, fetch_start = excluded.fetch_start, fetch_end = excluded.fetch_end,
       fetch_bytes = excluded.fetch_bytes, fetch_s = excluded.fetch_s, fetch_mbps = excluded.fetch_mbps, serve_at = excluded.serve_at, error = excluded.error,
       last_detail = excluded.last_detail, updated_at = excluded.updated_at`,
  )
    .bind(row.id, m[1], b.pod, next.boot_at, next.cache ?? null, next.missing ?? null, next.fetch_start ?? null, next.fetch_end ?? null, next.fetch_bytes ?? null, next.fetch_s ?? null, next.fetch_mbps ?? null, next.serve_at ?? null, next.error ?? null, next.last_detail ?? null, t)
    .run();
  // The log store (source serverless:<id>, pod = the worker): the boot's phases next to Runpod's container log.
  await env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg, fields) VALUES (?, ?, ?, ?, ?, ?, ?)")
    .bind(logSource(row), m[1], t, phase === "failed" || /failed/.test(detail) ? "error" : "info", "fv.boot", `[${phase}] ${detail}`, JSON.stringify({ endpoint: row.endpoint_id, boot: b.pod, phase, source: "sls-report" }))
    .run();
  return { ok: true };
}

/** The endpoint's latest boots (newest first). */
export async function recentBoots(env: Env, rowId: string, limit = 10): Promise<BootRow[]> {
  const r = await env.DB.prepare("SELECT worker_id, boot, boot_at, cache, missing, fetch_start, fetch_end, fetch_bytes, fetch_s, fetch_mbps, serve_at, error, last_detail, updated_at FROM serverless_boots WHERE endpoint = ? ORDER BY boot_at DESC LIMIT ?")
    .bind(rowId, limit)
    .all<BootRow>()
    .catch(() => ({ results: [] as BootRow[] }));
  return r.results || [];
}
