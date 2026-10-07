// The log explorer's query API (docs/control/README.md "Logs"): one line
// shape over every log fv-control keeps, filtered server-side and paged with
// keyset cursors, so the UI never downloads everything.
//
// Sources (`src`):
//   pod      D1 log_lines: what pods ship (cluster workers, and any other pod
//            that ships with an ingest token, e.g. standalone pods): 24 h tail
//   op       the operations' step logs (operations.log)
//   audit    the audit table
//   runpod   Runpod's own container/system tail of one pod (needs `pod`)
//   archive  the R2 NDJSON archive of one pod, older than the D1 tail (needs `pod`)
//
// Every line has a `uid` (`<source prefix>:<key>`); the global order is
// (ts, uid), and a cursor is the (ts, uid) of the last line returned, so all
// sources page together. Regex and case-sensitive searches are applied in the
// Worker after a LIKE pre-filter (D1 has no REGEXP); such a scan stops after a
// budget of rows and says how far it got (`partial`, `searched_until`).
import type { Env } from "./env";
import { runpod } from "./runpod";
import { validate } from "./schemas";
import { HttpError, now, parseJson, scrub } from "./util";

export const LEVELS = ["trace", "debug", "info", "warn", "error"] as const;
export type Level = (typeof LEVELS)[number];
export const SOURCES = ["pod", "op", "audit", "runpod", "archive"] as const;
export type Source = (typeof SOURCES)[number];
export const DEFAULT_SOURCES: Source[] = ["pod", "op", "audit"];
const PREFIX: Record<Source, string> = { pod: "p", op: "o", audit: "a", runpod: "r", archive: "z" };
export const D1_TAIL_MS = 24 * 3600_000;

export interface XLine {
  uid: string;
  source: Source;
  ts: number;
  level: Level;
  cluster_id: string | null;
  cluster: string | null;
  pool: string | null;
  pod_id: string | null;
  pod_name: string | null;
  target: string | null;
  msg: string;
  fields?: Record<string, unknown>;
}
export interface Cursor {
  ts: number;
  uid: string;
}
export interface LogQuery {
  sources: Source[];
  cluster?: string;
  pool?: string;
  pod?: string;
  op?: string;
  levels: Level[];
  since?: number;
  until?: number;
  text?: string;
  regex: boolean;
  caseSensitive: boolean;
  order: "asc" | "desc";
  limit: number;
  cursor?: Cursor;
}

// ---------------------------------------------------------------- parsing
const REL = /^-?(\d+(?:\.\d+)?)\s*(s|m|min|h|d|w)$/;
const UNIT: Record<string, number> = { s: 1000, m: 60_000, min: 60_000, h: 3_600_000, d: 86_400_000, w: 604_800_000 };
/** A time: unix ms (or s), an ISO date, or relative to now ("15m", "-2h", "7d"); undefined when empty. */
export function parseTime(v: string | undefined | null, t = now()): number | undefined {
  if (v === undefined || v === null || v === "") return undefined;
  const s = String(v).trim();
  if (s === "now") return t;
  const rel = REL.exec(s);
  if (rel) return Math.round(t - Number(rel[1]) * UNIT[rel[2]!]!);
  if (/^\d+$/.test(s)) {
    const n = Number(s);
    return n < 1e12 ? n * 1000 : n;
  }
  const d = Date.parse(s);
  if (Number.isNaN(d)) throw new HttpError(400, `time: ${s.slice(0, 40)} (unix ms, ISO 8601, or relative like 15m, 2h, 7d)`);
  return d;
}
export function encodeCursor(c: Cursor): string {
  return btoa(JSON.stringify([c.ts, c.uid])).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
}
export function decodeCursor(s: string | undefined): Cursor | undefined {
  if (!s) return undefined;
  try {
    const j = JSON.parse(atob(s.replace(/-/g, "+").replace(/_/g, "/")));
    if (Array.isArray(j) && typeof j[0] === "number" && typeof j[1] === "string") return { ts: j[0], uid: j[1] };
  } catch {
    /* fallthrough */
  }
  throw new HttpError(400, "cursor: not one this API returned");
}
const csv = (v?: string) => (v ? v.split(",").map((x) => x.trim()).filter(Boolean) : []);
const ID = /^[A-Za-z0-9_.:-]{1,80}$/;
/** The query from URL parameters (the same names the UI keeps in its URL). */
export function parseLogQuery(p: Record<string, string | undefined>, t = now()): LogQuery {
  // The filter fields' shapes (src/schemas.ts LogQueryZ, what the explorer's controls are bound to); empty values are "unset".
  const given = Object.fromEntries(Object.entries(p).filter(([, v]) => v !== undefined && v !== ""));
  const v = validate("log-query", given);
  if (!v.ok) throw new HttpError(400, v.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; "), { issues: v.issues });
  const srcs = csv(p.src || p.sources);
  for (const s of srcs) if (!SOURCES.includes(s as Source)) throw new HttpError(400, `src: ${SOURCES.join(", ")}`);
  let levels: Level[];
  const lv = csv(p.lv || p.levels);
  if (lv.length) {
    for (const l of lv) if (!LEVELS.includes(l as Level)) throw new HttpError(400, `lv: ${LEVELS.join(", ")}`);
    levels = lv as Level[];
  } else if (p.level) {
    const i = LEVELS.indexOf(p.level as Level);
    if (i < 0) throw new HttpError(400, `level: ${LEVELS.join(", ")}`);
    levels = LEVELS.slice(i) as Level[];
  } else levels = [...LEVELS];
  for (const k of ["cluster", "pool", "pod", "op"] as const) if (p[k] && !ID.test(p[k]!)) throw new HttpError(400, `${k}: invalid`);
  const text = p.q ? String(p.q).slice(0, 500) : undefined;
  const regex = p.re === "1" || p.re === "true";
  if (text && regex) compileRegex(text, p.cs === "1");
  const limit = Math.min(Math.max(Number(p.limit) || 200, 1), 1000);
  const since = parseTime(p.since ?? p.from, t);
  const until = parseTime(p.until ?? p.to, t);
  if (since !== undefined && until !== undefined && since > until) throw new HttpError(400, "since is after until");
  return {
    sources: srcs.length ? (srcs as Source[]) : p.pod && !p.op ? ["pod"] : [...DEFAULT_SOURCES],
    cluster: p.cluster || undefined,
    pool: p.pool || undefined,
    pod: p.pod || undefined,
    op: p.op || undefined,
    levels,
    since,
    until,
    text: text || undefined,
    regex,
    caseSensitive: p.cs === "1" || p.cs === "true",
    order: p.order === "asc" ? "asc" : "desc",
    limit,
    cursor: decodeCursor(p.cursor),
  };
}

// ---------------------------------------------------------------- matching
export function compileRegex(src: string, cs: boolean): RegExp {
  if (src.length > 300) throw new HttpError(400, "regex: at most 300 characters");
  try {
    return new RegExp(src, cs ? "" : "i");
  } catch (e) {
    throw new HttpError(400, `regex: ${(e as Error).message}`);
  }
}
/**
 * The longest literal a regex match must contain (for a LIKE pre-filter), or
 * null when there is none we can be sure of: alternation, or only classes.
 * Only top-level runs count; a quantifier that allows zero repeats drops the
 * character before it.
 */
export function regexLiteral(src: string): string | null {
  if (src.includes("|")) return null;
  let best = "";
  let run = "";
  let depth = 0;
  const flush = () => {
    if (run.length > best.length) best = run;
    run = "";
  };
  for (let i = 0; i < src.length; i++) {
    const c = src[i]!;
    if (c === "\\") {
      const n = src[i + 1];
      i++;
      if (n !== undefined && /[.*+?^${}()|[\]\\/\-]/.test(n) && depth === 0) {
        const q = src[i + 1];
        if (q === "?" || q === "*" || (q === "{" && /^\{0/.test(src.slice(i + 1)))) flush();
        else run += n;
      } else flush();
      continue;
    }
    if (c === "(") {
      depth++;
      flush();
      continue;
    }
    if (c === ")") {
      depth = Math.max(0, depth - 1);
      continue;
    }
    if (c === "[") {
      flush();
      const j = src.indexOf("]", i + 2);
      i = j < 0 ? src.length : j;
      continue;
    }
    if (depth > 0) continue;
    if (c === "?" || c === "*" || c === "{") {
      run = run.slice(0, -1);
      flush();
      if (c === "{") i = Math.max(i, src.indexOf("}", i));
      continue;
    }
    if (c === "+") {
      flush();
      continue;
    }
    if (".^$".includes(c)) {
      flush();
      continue;
    }
    run += c;
  }
  flush();
  return best.length >= 2 ? best : null;
}
export function likeEscape(s: string): string {
  return s.replace(/[\\%_]/g, (m) => "\\" + m);
}
/** The searchable text of a line: message, target, pod and the structured fields. */
export function haystack(l: Pick<XLine, "msg" | "target" | "fields" | "pod_id">): string {
  return `${l.msg}\n${l.target ?? ""}\n${l.pod_id ?? ""}\n${l.fields ? JSON.stringify(l.fields) : ""}`;
}
export function textMatcher(q: Pick<LogQuery, "text" | "regex" | "caseSensitive">): ((l: XLine) => boolean) | null {
  if (!q.text) return null;
  if (q.regex) {
    const re = compileRegex(q.text, q.caseSensitive);
    return (l) => re.test(haystack(l));
  }
  if (q.caseSensitive) return (l) => haystack(l).includes(q.text!);
  const t = q.text.toLowerCase();
  return (l) => haystack(l).toLowerCase().includes(t);
}

/** (ts, uid) order; +1 when a comes after b. */
export const cmpKey = (a: Cursor, b: Cursor) => (a.ts !== b.ts ? (a.ts < b.ts ? -1 : 1) : a.uid < b.uid ? -1 : a.uid > b.uid ? 1 : 0);
/** Whether a line lies strictly beyond the cursor in the query's direction. */
export const beyond = (l: Cursor, c: Cursor | undefined, order: "asc" | "desc") => !c || (order === "desc" ? cmpKey(l, c) < 0 : cmpKey(l, c) > 0);
const padId = (n: number) => String(n).padStart(12, "0");

/** Op and Runpod lines carry no level: infer one from the text. */
export function inferLevel(msg: string): Level {
  if (/\b(error|failed|failure|refused|cannot|panic|fatal|exception|traceback)\b/i.test(msg)) return "error";
  if (/\b(warn(ing)?|no stock|not available|no instances|retry(ing)?|timed? ?out|unreachable|over the floor|cancel+ed)\b/i.test(msg)) return "warn";
  if (/\bdebug\b/i.test(msg)) return "debug";
  return "info";
}

// ---------------------------------------------------------------- sources
interface SourceResult {
  lines: XLine[];
  /** Nothing more beyond the last line (or the frontier) in this direction. */
  exhausted: boolean;
  /** Scanned this far without filling the page (a budget-limited search). */
  frontier?: Cursor;
}
interface Ctx {
  env: Env;
  q: LogQuery;
  t: number;
  match: ((l: XLine) => boolean) | null;
  names: Map<string, string>;
  /** tail: audit rows past this id only. */
  afterAuditId?: number;
}

/** WHERE parts for log_lines shared by the query, the export and the facets. */
export function podWhere(q: LogQuery, opts: { text?: boolean } = {}): { where: string[]; vals: unknown[] } {
  const where: string[] = [];
  const vals: unknown[] = [];
  if (q.pod) where.push("l.pod_id = ?"), vals.push(q.pod);
  if (q.cluster) where.push("l.cluster_id = ?"), vals.push(q.cluster);
  if (q.pool) {
    where.push(`l.pod_id IN (SELECT pod_id FROM cluster_pods WHERE pool = ?${q.cluster ? " AND cluster_id = ?" : ""})`);
    vals.push(q.pool, ...(q.cluster ? [q.cluster] : []));
  }
  if (q.levels.length < LEVELS.length) where.push(`l.level IN (${q.levels.map(() => "?").join(",")})`), vals.push(...q.levels);
  if (q.since !== undefined) where.push("l.ts >= ?"), vals.push(q.since);
  if (q.until !== undefined) where.push("l.ts <= ?"), vals.push(q.until);
  if (opts.text !== false && q.text) {
    const lit = q.regex ? regexLiteral(q.text) : q.text;
    if (lit) {
      const pat = `%${likeEscape(lit)}%`;
      where.push("(l.msg LIKE ? ESCAPE '\\' OR l.fields LIKE ? ESCAPE '\\' OR l.target LIKE ? ESCAPE '\\' OR l.pod_id LIKE ? ESCAPE '\\')");
      vals.push(pat, pat, pat, pat);
    }
  }
  return { where, vals };
}
/** The keyset condition for one source's integer ids at the cursor's ts. */
function keyset(col: string, idCol: string, prefix: string, c: Cursor | undefined, order: "asc" | "desc"): { sql: string; vals: unknown[] } | null {
  if (!c) return null;
  const op = order === "desc" ? "<" : ">";
  const [cp, cid] = c.uid.split(":");
  if (cp === prefix) return { sql: `(${col} ${op} ? OR (${col} = ? AND ${idCol} ${op} ?))`, vals: [c.ts, c.ts, Number(cid)] };
  // Another source's line at the same ts: ours at that ts are on the far side when our prefix sorts beyond it.
  const sameTsBeyond = order === "desc" ? `${prefix}:` < c.uid : `${prefix}:` > c.uid;
  return sameTsBeyond ? { sql: `${col} ${op}= ?`, vals: [c.ts] } : { sql: `${col} ${op} ?`, vals: [c.ts] };
}

const POD_COLS = "l.id, l.cluster_id, l.pod_id, l.ts, l.level, l.target, l.msg, l.fields, cp.pool AS pool, pd.name AS pod_name";
const POD_FROM = "log_lines l LEFT JOIN cluster_pods cp ON cp.pod_id = l.pod_id LEFT JOIN pods pd ON pd.pod_id = l.pod_id";
function podLine(row: any, names: Map<string, string>): XLine {
  return {
    uid: `p:${padId(row.id)}`,
    source: "pod",
    ts: row.ts,
    level: row.level,
    cluster_id: row.cluster_id || null,
    cluster: names.get(row.cluster_id) ?? null,
    pool: row.pool ?? null,
    pod_id: row.pod_id,
    pod_name: row.pod_name ?? null,
    target: row.target ?? null,
    msg: row.msg,
    ...(row.fields ? { fields: parseJson(row.fields, { _raw: row.fields }) } : {}),
  };
}

async function podSource(x: Ctx): Promise<SourceResult> {
  const { env, q } = x;
  const base = podWhere(q);
  // LIKE is the whole filter for a plain case-insensitive search; otherwise the Worker checks every candidate row.
  const jsFilter = q.text && (q.regex || q.caseSensitive) ? x.match : null;
  const batch = jsFilter ? Math.max(q.limit * 2, 500) : q.limit + 1;
  const maxBatches = jsFilter ? 8 : 1;
  const out: XLine[] = [];
  let cur = q.cursor;
  for (let b = 0; b < maxBatches; b++) {
    const ks = keyset("l.ts", "l.id", "p", cur, q.order);
    const where = [...base.where, ...(ks ? [ks.sql] : [])];
    const dir = q.order === "desc" ? "DESC" : "ASC";
    const r = await env.DB.prepare(`SELECT ${POD_COLS} FROM ${POD_FROM} ${where.length ? "WHERE " + where.join(" AND ") : ""} ORDER BY l.ts ${dir}, l.id ${dir} LIMIT ${batch}`)
      .bind(...base.vals, ...(ks ? ks.vals : []))
      .all<any>();
    const rows = r.results || [];
    for (const row of rows) {
      const l = podLine(row, x.names);
      cur = { ts: l.ts, uid: l.uid };
      if (!jsFilter || jsFilter(l)) out.push(l);
      if (out.length > q.limit) break;
    }
    if (out.length > q.limit) return { lines: out.slice(0, q.limit), exhausted: false };
    if (rows.length < batch) return { lines: out, exhausted: true };
  }
  return { lines: out, exhausted: false, frontier: cur };
}

async function opSource(x: Ctx): Promise<SourceResult> {
  const { env, q } = x;
  const where: string[] = [];
  const vals: unknown[] = [];
  if (q.op) where.push("id = ?"), vals.push(q.op);
  if (q.cluster) where.push("cluster_id = ?"), vals.push(q.cluster);
  const lo = Math.max(q.since ?? -Infinity, q.order === "asc" && q.cursor ? q.cursor.ts : -Infinity);
  const hi = Math.min(q.until ?? Infinity, q.order === "desc" && q.cursor ? q.cursor.ts : Infinity);
  if (Number.isFinite(lo)) where.push("updated_at >= ?"), vals.push(lo);
  if (Number.isFinite(hi)) where.push("created_at <= ?"), vals.push(hi);
  const OPS = 200;
  const r = await env.DB.prepare(
    `SELECT id, cluster_id, kind, status, error, actor, params, log, created_at, updated_at FROM operations ${where.length ? "WHERE " + where.join(" AND ") : ""} ORDER BY created_at ${q.order === "desc" ? "DESC" : "ASC"} LIMIT ${OPS}`,
  )
    .bind(...vals)
    .all<any>();
  const rows = r.results || [];
  const lines: XLine[] = [];
  for (const o of rows) {
    const log = parseJson<{ at: number; msg: string }[]>(o.log, []);
    const entries = [...log];
    if (o.status === "failed" && o.error) entries.push({ at: o.updated_at, msg: `failed: ${o.error}` });
    entries.forEach((e, i) => {
      const pod = /\b([a-z0-9]{14})\b/.exec(e.msg)?.[1] ?? null;
      const pool = /^([a-z][a-z0-9-]{0,30}): /.exec(e.msg)?.[1] ?? null;
      const l: XLine = {
        uid: `o:${o.id}:${String(i).padStart(5, "0")}`,
        source: "op",
        ts: e.at,
        level: /^WARNING\b/.test(e.msg) ? "warn" : inferLevel(e.msg),
        cluster_id: o.cluster_id,
        cluster: x.names.get(o.cluster_id) ?? null,
        pool,
        pod_id: pod,
        pod_name: null,
        target: `op ${o.kind}`,
        msg: e.msg,
        fields: { op_id: o.id, op_kind: o.kind, op_status: o.status, actor: o.actor, ...(o.params ? { params: parseJson(o.params, o.params) } : {}) },
      };
      if (q.pod && l.pod_id !== q.pod && !e.msg.includes(q.pod)) return;
      if (q.pool && l.pool !== q.pool) return;
      if (!q.levels.includes(l.level)) return;
      if (q.since !== undefined && l.ts < q.since) return;
      if (q.until !== undefined && l.ts > q.until) return;
      if (!beyond(l, q.cursor, q.order)) return;
      if (x.match && !x.match(l)) return;
      lines.push(l);
    });
  }
  sortLines(lines, q.order);
  // Fewer operations than the cap: everything in range was seen.
  return { lines: lines.slice(0, q.limit), exhausted: rows.length < OPS && lines.length <= q.limit };
}

async function auditSource(x: Ctx): Promise<SourceResult> {
  const { env, q } = x;
  const where: string[] = [];
  const vals: unknown[] = [];
  if (q.cluster) {
    const name = x.names.get(q.cluster) ?? q.cluster;
    where.push("(target = ? OR target LIKE ? ESCAPE '\\' OR target LIKE ? ESCAPE '\\')");
    vals.push(name, `%${likeEscape(q.cluster)}%`, `cluster-spec:${likeEscape(q.cluster)}`);
  }
  if (q.pod) where.push("(target = ? OR detail LIKE ? ESCAPE '\\' OR after LIKE ? ESCAPE '\\')"), vals.push(q.pod, `%${likeEscape(q.pod)}%`, `%${likeEscape(q.pod)}%`);
  if (q.op) where.push("(after LIKE ? ESCAPE '\\')"), vals.push(`%${likeEscape(q.op)}%`);
  if (x.afterAuditId !== undefined) where.push("id > ?"), vals.push(x.afterAuditId);
  // Level: ok rows are info, failed rows error.
  const wantInfo = q.levels.includes("info");
  const wantErr = q.levels.includes("error");
  if (!wantInfo && !wantErr) return { lines: [], exhausted: true };
  if (!wantInfo) where.push("ok = 0");
  if (!wantErr) where.push("ok = 1");
  if (q.since !== undefined) where.push("at >= ?"), vals.push(q.since);
  if (q.until !== undefined) where.push("at <= ?"), vals.push(q.until);
  const lit = q.text ? (q.regex ? regexLiteral(q.text) : q.text) : null;
  if (lit) {
    const pat = `%${likeEscape(lit)}%`;
    where.push("(action LIKE ? ESCAPE '\\' OR target LIKE ? ESCAPE '\\' OR actor LIKE ? ESCAPE '\\' OR detail LIKE ? ESCAPE '\\' OR after LIKE ? ESCAPE '\\' OR before LIKE ? ESCAPE '\\')");
    vals.push(pat, pat, pat, pat, pat, pat);
  }
  const ks = keyset("at", "id", "a", q.cursor, q.order);
  if (ks) where.push(ks.sql), vals.push(...ks.vals);
  const jsFilter = q.text && (q.regex || q.caseSensitive) ? x.match : null;
  const lim = jsFilter ? Math.max(q.limit * 3, 300) : q.limit + 1;
  const dir = q.order === "desc" ? "DESC" : "ASC";
  const r = await env.DB.prepare(`SELECT * FROM audit ${where.length ? "WHERE " + where.join(" AND ") : ""} ORDER BY at ${dir}, id ${dir} LIMIT ${lim}`)
    .bind(...vals)
    .all<any>();
  const rows = r.results || [];
  const lines: XLine[] = [];
  let last: Cursor | undefined;
  for (const a of rows) {
    const clusterId = [...x.names.entries()].find(([, n]) => n === a.target)?.[0] ?? (/(c_[0-9a-f]{16})/.exec(a.target || "")?.[1] || null);
    const l: XLine = {
      uid: `a:${padId(a.id)}`,
      source: "audit",
      ts: a.at,
      level: a.ok ? "info" : "error",
      cluster_id: clusterId,
      cluster: clusterId ? (x.names.get(clusterId) ?? null) : null,
      pool: null,
      pod_id: /^[a-z0-9]{14}$/.test(a.target || "") ? a.target : null,
      pod_name: null,
      target: "audit",
      msg: `${a.action}${a.target ? ` ${a.target}` : ""}${a.detail ? `: ${a.detail}` : ""}`,
      fields: {
        actor: a.actor,
        action: a.action,
        target: a.target,
        ok: !!a.ok,
        ...(a.ip ? { ip: a.ip } : {}),
        ...(a.before ? { before: clip(parseJson(a.before, a.before)) } : {}),
        ...(a.after ? { after: clip(parseJson(a.after, a.after)) } : {}),
      },
    };
    last = { ts: l.ts, uid: l.uid };
    if (!jsFilter || jsFilter(l)) lines.push(l);
    if (lines.length > q.limit) break;
  }
  if (lines.length > q.limit) return { lines: lines.slice(0, q.limit), exhausted: false };
  if (rows.length < lim) return { lines, exhausted: true };
  return jsFilter ? { lines, exhausted: false, frontier: last } : { lines: lines.slice(0, q.limit), exhausted: false };
}
/** Large audit snapshots (whole specs) are cut so a page stays small. */
function clip(v: unknown): unknown {
  const s = JSON.stringify(v);
  return s && s.length > 4000 ? { _truncated: true, preview: s.slice(0, 4000) } : v;
}

const RP_TS = /^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s*/;
/** Runpod's own tail lines of one pod as log lines (a leading timestamp is used when there is one). */
export function runpodLines(pod: string, container: string[], system: string[], t: number): XLine[] {
  const out: XLine[] = [];
  const add = (arr: string[], kind: "container" | "system") => {
    let lastTs = t - arr.length * 1000;
    arr.forEach((raw, i) => {
      const m = RP_TS.exec(raw);
      const parsed = m ? Date.parse(m[1]!.replace(" ", "T")) : NaN;
      const ts = Number.isFinite(parsed) ? parsed : lastTs + 1;
      lastTs = ts;
      const msg = m ? raw.slice(m[0].length) : raw;
      out.push({ uid: `r:${kind[0]}${String(i).padStart(5, "0")}`, source: "runpod", ts, level: inferLevel(msg), cluster_id: null, cluster: null, pool: null, pod_id: pod, pod_name: null, target: `runpod ${kind}`, msg, fields: { stream: kind } });
    });
  };
  add(container, "container");
  add(system, "system");
  return out;
}
async function runpodSource(x: Ctx): Promise<SourceResult> {
  const { q } = x;
  if (!q.pod) return { lines: [], exhausted: true };
  const l = await runpod.logs(x.env, q.pod).catch(() => ({ container: [] as string[], system: [] as string[] }));
  const sc = (a: string[]) => a.slice(-1000).map((s) => scrub(x.env, s));
  const lines = runpodLines(q.pod, sc(l.container), sc(l.system), x.t).filter(
    (r) => q.levels.includes(r.level) && (q.since === undefined || r.ts >= q.since) && (q.until === undefined || r.ts <= q.until) && beyond(r, q.cursor, q.order) && (!x.match || x.match(r)),
  );
  sortLines(lines, q.order);
  return { lines: lines.slice(0, q.limit), exhausted: lines.length <= q.limit };
}

const HOUR = 3_600_000;
/** R2 hour prefixes of one pod between two times, in the query's order. */
export function archiveHours(since: number, until: number, order: "asc" | "desc"): string[] {
  const out: string[] = [];
  for (let h = Math.floor(since / HOUR) * HOUR; h <= until; h += HOUR) {
    const iso = new Date(h).toISOString();
    out.push(`${iso.slice(0, 10)}/${iso.slice(11, 13)}/`);
    if (out.length > 24 * 31) break;
  }
  return order === "desc" ? out.reverse() : out;
}
async function archiveSource(x: Ctx): Promise<SourceResult> {
  const { env, q } = x;
  if (!q.pod) return { lines: [], exhausted: true };
  // Only what the D1 tail no longer has: the two never overlap.
  const cutoff = x.t - D1_TAIL_MS;
  const until = Math.min(q.until ?? cutoff, cutoff, q.order === "desc" && q.cursor ? q.cursor.ts : Infinity);
  const since = Math.max(q.since ?? cutoff - 7 * 86_400_000, cutoff - 30 * 86_400_000, q.order === "asc" && q.cursor ? q.cursor.ts : -Infinity);
  if (since > until) return { lines: [], exhausted: true };
  const row = await env.DB.prepare("SELECT cluster_id, pool FROM cluster_pods WHERE pod_id = ?").bind(q.pod).first<{ cluster_id: string; pool: string | null }>();
  const clusterId = row?.cluster_id ?? q.cluster;
  if (!clusterId) return { lines: [], exhausted: true };
  const hours = archiveHours(since, until, q.order);
  const out: XLine[] = [];
  let objects = 0;
  const MAX_OBJECTS = 150;
  for (let hi = 0; hi < hours.length; hi++) {
    const prefix = `logs/${clusterId}/${q.pod}/${hours[hi]}`;
    const keys: string[] = [];
    let cursor: string | undefined;
    do {
      const l = await env.LOGS.list({ prefix, cursor, limit: 1000 });
      keys.push(...l.objects.map((o) => o.key));
      cursor = l.truncated ? l.cursor : undefined;
    } while (cursor && keys.length < 2000);
    keys.sort();
    for (const k of keys) {
      const o = await env.LOGS.get(k);
      objects++;
      if (!o) continue;
      const text = await o.text();
      const base = k.slice(prefix.length).replace(/\.ndjson$/, "");
      text.split("\n").forEach((raw, i) => {
        if (!raw) return;
        const j = parseJson<any>(raw, null);
        if (!j) return;
        const l: XLine = {
          uid: `z:${hours[hi]!.replace(/\//g, "")}${base}:${String(i).padStart(5, "0")}`,
          source: "archive",
          ts: Number(j.ts) || 0,
          level: LEVELS.includes(j.level) ? j.level : "info",
          cluster_id: clusterId,
          cluster: x.names.get(clusterId) ?? null,
          pool: row?.pool ?? null,
          pod_id: q.pod!,
          pod_name: null,
          target: j.target ?? null,
          msg: String(j.msg ?? ""),
          ...(j.fields ? { fields: j.fields } : {}),
        };
        if (!q.levels.includes(l.level) || l.ts < since || l.ts > until || !beyond(l, q.cursor, q.order) || (x.match && !x.match(l))) return;
        out.push(l);
      });
    }
    // Whole hours only: lines of one hour can sit in any of its objects.
    if (out.length >= q.limit || objects >= MAX_OBJECTS) {
      sortLines(out, q.order);
      const more = hi < hours.length - 1;
      if (out.length > q.limit) return { lines: out.slice(0, q.limit), exhausted: false };
      if (!more) return { lines: out, exhausted: true };
      // Budget spent before the page filled: the next page starts at the next hour.
      const edgeHour = Date.parse(`${hours[hi]!.slice(0, 10)}T${hours[hi]!.slice(11, 13)}:00:00Z`);
      return { lines: out, exhausted: false, frontier: { ts: q.order === "desc" ? edgeHour : edgeHour + HOUR - 1, uid: q.order === "desc" ? "" : "~" } };
    }
  }
  sortLines(out, q.order);
  return { lines: out.slice(0, q.limit), exhausted: out.length <= q.limit };
}

export function sortLines(lines: XLine[], order: "asc" | "desc") {
  lines.sort((a, b) => (order === "desc" ? -cmpKey(a, b) : cmpKey(a, b)));
}

async function clusterNames(env: Env): Promise<Map<string, string>> {
  const r = await env.DB.prepare("SELECT id, name FROM clusters").all<{ id: string; name: string }>();
  return new Map((r.results || []).map((c) => [c.id, c.name]));
}

export interface QueryResult {
  lines: XLine[];
  /** Pass back as `cursor` for the next page in the same direction; null when there is nothing more. */
  next: string | null;
  /** The search stopped at its scan budget before filling the page: it covered lines up to this time. */
  partial: boolean;
  searched_until: number | null;
  sources: Source[];
  order: "asc" | "desc";
}
const RUNNERS: Record<Source, (x: Ctx) => Promise<SourceResult>> = { pod: podSource, op: opSource, audit: auditSource, runpod: runpodSource, archive: archiveSource };

/** One page of the merged, filtered lines of every requested source. */
export async function queryLogs(env: Env, q: LogQuery, t = now()): Promise<QueryResult> {
  const x: Ctx = { env, q, t, match: textMatcher(q), names: await clusterNames(env) };
  const results = await Promise.all(q.sources.map((s) => RUNNERS[s](x)));
  const all = results.flatMap((r) => r.lines);
  sortLines(all, q.order);
  // A budget-limited source covered lines only up to its frontier: nothing past the nearest frontier is complete.
  const frontiers = results.filter((r) => r.frontier).map((r) => r.frontier!);
  const fr = frontiers.length ? frontiers.reduce((a, b) => (q.order === "desc" ? (cmpKey(a, b) > 0 ? a : b) : cmpKey(a, b) < 0 ? a : b)) : undefined;
  let lines = fr ? all.filter((l) => !beyond(l, fr, q.order) || cmpKey(l, fr) === 0) : all;
  const cut = lines.length > q.limit;
  lines = lines.slice(0, q.limit);
  const more = cut || results.some((r) => !r.exhausted);
  let next: Cursor | undefined;
  if (more) next = lines.length && (!fr || lines.length >= q.limit) ? { ts: lines[lines.length - 1]!.ts, uid: lines[lines.length - 1]!.uid } : fr;
  return {
    lines,
    next: next ? encodeCursor(next) : null,
    partial: !!fr && lines.length < q.limit,
    searched_until: fr && lines.length < q.limit ? fr.ts : null,
    sources: q.sources,
    order: q.order,
  };
}

// ---------------------------------------------------------------- tail
export interface TailState {
  p: number; // highest log_lines id seen
  a: number; // highest audit id seen
  o: number; // newest op line time seen
}
/** New lines since a tail state (ingest order, not line time: a late batch is not missed). */
export async function tailLogs(env: Env, q: LogQuery, st: TailState | null, t = now()): Promise<{ lines: XLine[]; state: TailState }> {
  const ids = await env.DB.prepare("SELECT (SELECT COALESCE(MAX(id), 0) FROM log_lines) AS p, (SELECT COALESCE(MAX(id), 0) FROM audit) AS a").first<{ p: number; a: number }>();
  const next: TailState = { p: ids?.p ?? 0, a: ids?.a ?? 0, o: st?.o ?? t };
  if (!st) return { lines: [], state: next };
  const x: Ctx = { env, q: { ...q, cursor: undefined, order: "asc", limit: 500 }, t, match: textMatcher(q), names: await clusterNames(env) };
  const lines: XLine[] = [];
  if (q.sources.includes("pod") && next.p > st.p) {
    const w = podWhere(x.q);
    const r = await env.DB.prepare(`SELECT ${POD_COLS} FROM ${POD_FROM} WHERE l.id > ? AND l.id <= ?${w.where.length ? " AND " + w.where.join(" AND ") : ""} ORDER BY l.id LIMIT 1000`)
      .bind(st.p, next.p, ...w.vals)
      .all<any>();
    for (const row of r.results || []) {
      const l = podLine(row, x.names);
      if (!x.match || x.match(l)) lines.push(l);
    }
  }
  if (q.sources.includes("audit") && next.a > st.a) {
    const r = await auditSource({ ...x, afterAuditId: st.a, q: { ...x.q, until: undefined } });
    lines.push(...r.lines);
  }
  if (q.sources.includes("op")) {
    const r = await opSource({ ...x, q: { ...x.q, since: st.o + 1, until: undefined } });
    lines.push(...r.lines);
    for (const l of r.lines) next.o = Math.max(next.o, l.ts);
  }
  sortLines(lines, "asc");
  return { lines, state: next };
}
export const encodeTail = (s: TailState) => `${s.p}.${s.a}.${s.o}`;
export function decodeTail(s: string | undefined): TailState | null {
  if (!s) return null;
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(s);
  if (!m) throw new HttpError(400, "tail: the state the last tail call returned");
  return { p: Number(m[1]), a: Number(m[2]), o: Number(m[3]) };
}

// ---------------------------------------------------------------- context
/** N lines before and after one line, from the same pod (pod lines), operation (op lines) or the audit log. */
export async function lineContext(env: Env, uid: string, before: number, after: number): Promise<{ lines: XLine[]; anchor: string }> {
  before = Math.min(Math.max(before, 0), 200);
  after = Math.min(Math.max(after, 0), 200);
  const names = await clusterNames(env);
  const [prefix, rest] = [uid.slice(0, 1), uid.slice(2)];
  if (prefix === "p") {
    const row = await env.DB.prepare(`SELECT ${POD_COLS} FROM ${POD_FROM} WHERE l.id = ?`).bind(Number(rest)).first<any>();
    if (!row) throw new HttpError(404, "that line is no longer in the 24 h tail");
    const self = podLine(row, names);
    const base = { sources: ["pod"] as Source[], pod: row.pod_id, levels: [...LEVELS], regex: false, caseSensitive: false };
    const cur = { ts: self.ts, uid: self.uid };
    const [b, a] = await Promise.all([
      before ? queryLogs(env, { ...base, order: "desc", limit: before, cursor: cur }).then((r) => r.lines) : [],
      after ? queryLogs(env, { ...base, order: "asc", limit: after, cursor: cur }).then((r) => r.lines) : [],
    ]);
    return { lines: [...b.reverse(), self, ...a], anchor: uid };
  }
  if (prefix === "o") {
    const opId = rest.split(":")[0]!;
    const r = await opSource({ env, q: { sources: ["op"], op: opId, levels: [...LEVELS], regex: false, caseSensitive: false, order: "asc", limit: 5000 }, t: now(), match: null, names });
    const i = r.lines.findIndex((l) => l.uid === uid);
    if (i < 0) throw new HttpError(404, "no such operation line");
    return { lines: r.lines.slice(Math.max(0, i - before), i + after + 1), anchor: uid };
  }
  if (prefix === "a") {
    const row = await env.DB.prepare("SELECT at FROM audit WHERE id = ?").bind(Number(rest)).first<{ at: number }>();
    if (!row) throw new HttpError(404, "no such audit row");
    const base = { sources: ["audit"] as Source[], levels: [...LEVELS], regex: false, caseSensitive: false };
    const cur = { ts: row.at, uid };
    const [b, a] = await Promise.all([
      before ? queryLogs(env, { ...base, order: "desc", limit: before, cursor: cur }).then((r) => r.lines) : [],
      queryLogs(env, { ...base, order: "asc", limit: after + 1, cursor: { ts: row.at, uid: `a:${padId(Number(rest) - 1)}` } }).then((r) => r.lines),
    ]);
    return { lines: [...b.reverse(), ...a], anchor: uid };
  }
  throw new HttpError(400, "context: pod, op and audit lines");
}

// ---------------------------------------------------------------- facets
/** Counts for the explorer's pickers and its time histogram (pod lines in the D1 tail, filters but the text applied). */
export async function logFacets(env: Env, q: LogQuery, buckets = 60, t = now()) {
  const since = q.since ?? t - D1_TAIL_MS;
  const until = q.until ?? t;
  const step = Math.max(1000, Math.ceil((until - since) / Math.min(Math.max(buckets, 4), 200)));
  const qq: LogQuery = { ...q, since, until };
  const w = podWhere(qq, { text: false });
  const where = w.where.length ? "WHERE " + w.where.join(" AND ") : "";
  const [hist, pods] = await Promise.all([
    q.sources.includes("pod")
      ? env.DB.prepare(`SELECT ((l.ts - ?) / ?) AS b, l.level AS level, COUNT(*) AS n FROM log_lines l ${where} GROUP BY b, l.level`).bind(since, step, ...w.vals).all<{ b: number; level: string; n: number }>()
      : Promise.resolve({ results: [] as { b: number; level: string; n: number }[] }),
    env.DB.prepare(
      `SELECT l.pod_id, l.cluster_id, cp.pool, pd.name, COUNT(*) AS n, SUM(CASE WHEN l.level = 'error' THEN 1 ELSE 0 END) AS errors, SUM(CASE WHEN l.level = 'warn' THEN 1 ELSE 0 END) AS warns, MAX(l.ts) AS last
       FROM log_lines l LEFT JOIN cluster_pods cp ON cp.pod_id = l.pod_id LEFT JOIN pods pd ON pd.pod_id = l.pod_id WHERE l.ts >= ? GROUP BY l.pod_id ORDER BY last DESC LIMIT 300`,
    )
      .bind(t - D1_TAIL_MS)
      .all<any>(),
  ]);
  const histogram = new Map<number, Record<string, number>>();
  for (const r of hist.results || []) {
    const b = Math.floor(Number(r.b));
    const e = histogram.get(b) || {};
    e[r.level] = (e[r.level] || 0) + r.n;
    histogram.set(b, e);
  }
  return {
    since,
    until,
    step_ms: step,
    histogram: [...histogram.entries()].sort((a, b) => a[0] - b[0]).map(([b, c]) => ({ t: since + b * step, counts: c })),
    pods: pods.results || [],
  };
}
