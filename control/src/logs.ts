// Logs (docs/control/README.md "Logs"): fv-serve ships batches of its JSON
// tracing lines to POST /ingest/v1/logs with its cluster's ingest token.
// Every batch lands in R2 (NDJSON, the archive; bucket lifecycle does the
// retention), the recent tail in D1 (24 h, searchable), and live-tail
// WebSockets get it through the cluster's Durable Object.
import { recordBoot } from "./boottime";
import { stubFor } from "./cluster/control";
import { sha256Hex } from "./crypto";
import type { Env } from "./env";
import { HttpError, now, rateLimit, scrub, utcDay } from "./util";

export const INGEST_MAX_BYTES = 1_048_576;
export const INGEST_MAX_LINES = 2000;
const LEVELS = ["trace", "debug", "info", "warn", "error"];

export interface LogLine {
  ts: number;
  level: string;
  target?: string;
  msg: string;
  fields?: Record<string, unknown>;
}

function normLine(x: any): LogLine | null {
  if (!x || typeof x !== "object") return null;
  const tsRaw = x.ts ?? x.timestamp ?? x.time;
  const ts = typeof tsRaw === "number" ? (tsRaw < 1e12 ? tsRaw * 1000 : tsRaw) : Date.parse(String(tsRaw || "")) || now();
  const level = String(x.level || "info").toLowerCase();
  const fields = x.fields && typeof x.fields === "object" ? { ...x.fields } : {};
  const msg = String(x.msg ?? x.message ?? fields.message ?? "");
  delete (fields as any).message;
  return {
    ts,
    level: LEVELS.includes(level) ? level : "info",
    target: x.target ? String(x.target).slice(0, 200) : undefined,
    msg: msg.slice(0, 8192),
    fields: Object.keys(fields).length ? fields : undefined,
  };
}

/** POST /ingest/v1/logs: {pod, lines: [...]} or NDJSON with ?pod=. Returns the accepted count. */
export async function ingest(env: Env, req: Request, ctx?: ExecutionContext): Promise<{ accepted: number }> {
  const auth = req.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) throw new HttpError(401, "ingest token required");
  const hash = await sha256Hex(auth.slice(7).trim());
  const cl = await env.DB.prepare("SELECT id FROM clusters WHERE ingest_hash = ?").bind(hash).first<{ id: string }>();
  if (!cl) throw new HttpError(401, "invalid ingest token");
  if (!(await rateLimit(env, `ingest:${cl.id}`, 600, 60))) throw new HttpError(429, "ingest rate limit (600 batches/min per cluster)");
  const len = Number(req.headers.get("content-length") || 0);
  if (len > INGEST_MAX_BYTES) throw new HttpError(413, "batch too large (1 MiB max)");
  const text = await req.text();
  if (text.length > INGEST_MAX_BYTES) throw new HttpError(413, "batch too large (1 MiB max)");
  let pod = new URL(req.url).searchParams.get("pod") || "";
  let raw: any[] = [];
  const ct = req.headers.get("content-type") || "";
  if (ct.includes("ndjson")) raw = text.split("\n").filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return null; } });
  else {
    let j: any;
    try {
      j = JSON.parse(text);
    } catch {
      throw new HttpError(400, "body: JSON {pod, lines} or NDJSON");
    }
    pod = String(j.pod || pod);
    raw = Array.isArray(j.lines) ? j.lines : [];
  }
  if (!/^[a-z0-9]{6,40}$/i.test(pod)) throw new HttpError(400, "pod: the Runpod pod id");
  const known = await env.DB.prepare("SELECT 1 AS x FROM cluster_pods WHERE pod_id = ? AND cluster_id = ?").bind(pod, cl.id).first();
  if (!known) throw new HttpError(403, "pod is not part of this cluster");
  const lines = raw.slice(0, INGEST_MAX_LINES).map(normLine).filter((x): x is LogLine => !!x);
  if (!lines.length) return { accepted: 0 };
  // Never keep one of the controller's own secrets, should a pod print it.
  for (const l of lines) {
    l.msg = scrub(env, l.msg);
    if (l.fields) l.fields = JSON.parse(scrub(env, JSON.stringify(l.fields)));
  }
  const t = now();
  const day = utcDay(lines[0]!.ts);
  const hour = new Date(lines[0]!.ts).toISOString().slice(11, 13);
  const key = `logs/${cl.id}/${pod}/${day}/${hour}/${t}-${Math.random().toString(36).slice(2, 8)}.ndjson`;
  await env.LOGS.put(key, lines.map((l) => JSON.stringify(l)).join("\n") + "\n", { httpMetadata: { contentType: "application/x-ndjson" } });
  const stmt = env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg, fields) VALUES (?, ?, ?, ?, ?, ?, ?)");
  const stmts = lines.map((l) => stmt.bind(cl.id, pod, l.ts, l.level, l.target ?? null, l.msg, l.fields ? JSON.stringify(l.fields).slice(0, 8192) : null));
  for (let i = 0; i < stmts.length; i += 100) await env.DB.batch(stmts.slice(i, i + 100));
  // Shipped tracing events also mark boot phases (model resident, warm-up, ready, edge link).
  await recordBoot(env, pod, cl.id, lines.map((l) => ({ stream: "shipped" as const, ts: l.ts, text: `${l.target ?? ""}: ${l.msg} ${Object.entries(l.fields || {}).map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(" ")}` }))).catch(() => []);
  const fanout = stubFor(env, cl.id).fetch("https://ops/broadcast", { method: "POST", body: JSON.stringify({ pod, lines }) }).catch(() => null);
  if (ctx) ctx.waitUntil(fanout);
  else await fanout;
  return { accepted: lines.length };
}

/**
 * Log lines, oldest first. Without after_id: the newest `limit` lines. With
 * after_id: the `limit` lines after it (follow a tail by passing the last id
 * back). source: runpod (the Runpod container / system log the controller
 * captures, podlogs.ts), control (the controller's own lines about the pod:
 * boot milestones, boottime.ts) or serve (what fv-serve ships).
 */
export async function searchLogs(env: Env, q: { pod?: string; cluster?: string; text?: string; level?: string; since?: number; until?: number; limit?: number; after_id?: number; source?: string }) {
  const where: string[] = [];
  const vals: unknown[] = [];
  if (q.pod) where.push("pod_id = ?"), vals.push(q.pod);
  if (q.cluster) where.push("cluster_id = ?"), vals.push(q.cluster);
  if (q.level && LEVELS.includes(q.level)) where.push(`level IN (${LEVELS.slice(LEVELS.indexOf(q.level)).map(() => "?").join(",")})`), vals.push(...LEVELS.slice(LEVELS.indexOf(q.level)));
  if (q.text) where.push("(msg LIKE ? OR fields LIKE ?)"), vals.push(`%${q.text}%`, `%${q.text}%`);
  if (q.since) where.push("ts >= ?"), vals.push(q.since);
  if (q.until) where.push("ts <= ?"), vals.push(q.until);
  if (q.after_id) where.push("id > ?"), vals.push(q.after_id);
  if (q.source === "runpod") where.push("target LIKE 'runpod.%'");
  else if (q.source === "control") where.push("target LIKE 'fv-control.%'");
  else if (q.source === "serve") where.push("(target IS NULL OR (target NOT LIKE 'runpod.%' AND target NOT LIKE 'fv-control.%'))");
  const limit = Math.min(Math.max(q.limit || 200, 1), 2000);
  const r = await env.DB.prepare(`SELECT id, cluster_id, pod_id, ts, level, target, msg, fields FROM log_lines ${where.length ? "WHERE " + where.join(" AND ") : ""} ORDER BY id ${q.after_id ? "ASC" : "DESC"} LIMIT ${limit}`)
    .bind(...vals)
    .all<any>();
  const rows = q.after_id ? r.results || [] : (r.results || []).reverse();
  return rows.map((x) => ({ ...x, fields: x.fields ? JSON.parse(x.fields) : undefined }));
}

/** The R2 archive of one pod and day as one NDJSON stream. */
export async function downloadLogs(env: Env, clusterId: string, pod: string, day: string): Promise<ReadableStream> {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new HttpError(400, "day: YYYY-MM-DD");
  const prefix = `logs/${clusterId}/${pod}/${day}/`;
  const keys: string[] = [];
  let cursor: string | undefined;
  do {
    const l = await env.LOGS.list({ prefix, cursor, limit: 1000 });
    keys.push(...l.objects.map((o) => o.key));
    cursor = l.truncated ? l.cursor : undefined;
  } while (cursor && keys.length < 20000);
  keys.sort();
  const { readable, writable } = new TransformStream();
  (async () => {
    const w = writable.getWriter();
    for (const k of keys) {
      const o = await env.LOGS.get(k);
      if (o) await w.write(new Uint8Array(await o.arrayBuffer()));
    }
    await w.close();
  })();
  return readable;
}
