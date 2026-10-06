// The log explorer's state without the DOM: the view (filters, sort,
// grouping) and its URL form, the rows a buffer of lines turns into, the
// selection, find matches and text export. Tested in test/unit/logs-ui.test.ts.

export const LEVELS = ["trace", "debug", "info", "warn", "error"] as const;
export type Level = (typeof LEVELS)[number];
export const SOURCES = ["pod", "op", "audit", "runpod", "archive"] as const;
export type Source = (typeof SOURCES)[number];
export const SOURCE_LABEL: Record<Source, string> = { pod: "Pods", op: "Operations", audit: "Audit", runpod: "Runpod tail", archive: "Archive (R2)" };

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

export interface View {
  src: Source[];
  cluster: string;
  pool: string;
  pod: string;
  op: string;
  lv: Level[];
  q: string;
  re: boolean;
  cs: boolean;
  /** filter: the server returns matching lines only; find: every line, matches highlighted (n / N). */
  mode: "filter" | "find";
  from: string;
  to: string;
  order: "asc" | "desc";
  sort: "time" | "level" | "source" | "pod";
  group: "none" | "pod" | "level";
  ctx: number;
  live: boolean;
  sel: string;
  /** Jump: load around this time (unix ms). */
  at: number | null;
}
export const DEFAULT_VIEW: View = {
  src: ["pod", "op", "audit"],
  cluster: "",
  pool: "",
  pod: "",
  op: "",
  lv: ["info", "warn", "error"],
  q: "",
  re: false,
  cs: false,
  mode: "filter",
  from: "1h",
  to: "",
  order: "asc",
  sort: "time",
  group: "none",
  ctx: 5,
  live: false,
  sel: "",
  at: null,
};
export const RANGES = ["5m", "15m", "1h", "6h", "24h", "3d", "7d"];

const list = <T extends string>(v: string | null, all: readonly T[]): T[] | null => {
  if (v === null) return null;
  const out = v.split(",").filter((x): x is T => (all as readonly string[]).includes(x));
  return out;
};
/** The view from the hash's query string (`#/logs?…`); unknown values fall back to the defaults. */
export function viewFromParams(p: URLSearchParams): View {
  const v: View = { ...DEFAULT_VIEW, src: [...DEFAULT_VIEW.src], lv: [...DEFAULT_VIEW.lv] };
  const src = list(p.get("src"), SOURCES);
  if (src && src.length) v.src = src;
  else if (p.get("pod") && !p.get("src")) v.src = ["pod", "op", "audit"];
  const lv = list(p.get("lv"), LEVELS);
  if (lv && lv.length) v.lv = lv;
  else if (p.get("level") && LEVELS.includes(p.get("level") as Level)) v.lv = LEVELS.slice(LEVELS.indexOf(p.get("level") as Level)) as Level[];
  for (const k of ["cluster", "pool", "pod", "op", "q", "sel"] as const) v[k] = p.get(k) ?? "";
  v.re = p.get("re") === "1";
  v.cs = p.get("cs") === "1";
  v.mode = p.get("mode") === "find" ? "find" : "filter";
  if (p.has("from")) v.from = p.get("from") || "";
  v.to = p.get("to") || "";
  v.order = p.get("order") === "desc" ? "desc" : "asc";
  const sort = p.get("sort");
  v.sort = sort === "level" || sort === "source" || sort === "pod" ? sort : "time";
  const g = p.get("group");
  v.group = g === "pod" || g === "level" ? g : "none";
  const ctx = Number(p.get("ctx"));
  v.ctx = Number.isInteger(ctx) && ctx >= 0 && ctx <= 200 && p.has("ctx") ? ctx : DEFAULT_VIEW.ctx;
  v.live = p.get("live") === "1";
  const at = Number(p.get("at"));
  v.at = p.get("at") && Number.isFinite(at) ? at : null;
  return v;
}
/** The view as query parameters: only what differs from the defaults, in a stable order. */
export function viewToParams(v: View): URLSearchParams {
  const p = new URLSearchParams();
  const d = DEFAULT_VIEW;
  if (v.src.join(",") !== d.src.join(",")) p.set("src", v.src.join(","));
  for (const k of ["cluster", "pool", "pod", "op"] as const) if (v[k]) p.set(k, v[k]);
  if (v.lv.join(",") !== d.lv.join(",")) p.set("lv", v.lv.join(","));
  if (v.q) p.set("q", v.q);
  if (v.re) p.set("re", "1");
  if (v.cs) p.set("cs", "1");
  if (v.mode !== d.mode) p.set("mode", v.mode);
  if (v.from !== d.from) p.set("from", v.from);
  if (v.to) p.set("to", v.to);
  if (v.order !== d.order) p.set("order", v.order);
  if (v.sort !== d.sort) p.set("sort", v.sort);
  if (v.group !== d.group) p.set("group", v.group);
  if (v.ctx !== d.ctx) p.set("ctx", String(v.ctx));
  if (v.live) p.set("live", "1");
  if (v.at !== null) p.set("at", String(v.at));
  if (v.sel) p.set("sel", v.sel);
  return p;
}
/** The API parameters of a view: the filters the server applies (find mode keeps the text local). */
export function serverParams(v: View): URLSearchParams {
  const p = new URLSearchParams();
  p.set("src", v.src.join(","));
  for (const k of ["cluster", "pool", "pod", "op"] as const) if (v[k]) p.set(k, v[k]);
  p.set("lv", v.lv.join(","));
  if (v.q && v.mode === "filter") {
    p.set("q", v.q);
    if (v.re) p.set("re", "1");
    if (v.cs) p.set("cs", "1");
  }
  if (v.from) p.set("since", v.from);
  if (v.to) p.set("until", v.to);
  return p;
}

// ---------------------------------------------------------------- matching
export function matcher(q: string, re: boolean, cs: boolean): ((s: string) => boolean) | null {
  if (!q) return null;
  if (re) {
    let r: RegExp;
    try {
      r = new RegExp(q, cs ? "" : "i");
    } catch {
      return null;
    }
    return (s) => r.test(s);
  }
  if (cs) return (s) => s.includes(q);
  const t = q.toLowerCase();
  return (s) => s.toLowerCase().includes(t);
}
/** [start, end) ranges of matches in a string, for highlighting. */
export function matchRanges(s: string, q: string, re: boolean, cs: boolean, max = 50): [number, number][] {
  if (!q) return [];
  const out: [number, number][] = [];
  if (re) {
    let r: RegExp;
    try {
      r = new RegExp(q, cs ? "g" : "gi");
    } catch {
      return [];
    }
    for (const m of s.matchAll(r)) {
      if (m[0].length === 0) continue;
      out.push([m.index!, m.index! + m[0].length]);
      if (out.length >= max) break;
    }
    return out;
  }
  const hay = cs ? s : s.toLowerCase();
  const needle = cs ? q : q.toLowerCase();
  for (let i = hay.indexOf(needle); i >= 0 && out.length < max; i = hay.indexOf(needle, i + needle.length)) out.push([i, i + needle.length]);
  return out;
}
export const lineText = (l: XLine) => `${l.msg}${l.fields ? " " + JSON.stringify(l.fields) : ""}`;
export const searchable = (l: XLine) => `${l.msg}\n${l.target ?? ""}\n${l.pod_id ?? ""}\n${l.fields ? JSON.stringify(l.fields) : ""}`;

// ---------------------------------------------------------------- rows
export type Row = { kind: "line"; line: XLine } | { kind: "group"; key: string; label: string; count: number };
const LEVEL_RANK: Record<string, number> = { error: 0, warn: 1, info: 2, debug: 3, trace: 4 };
const SOURCE_RANK: Record<string, number> = { pod: 0, runpod: 1, archive: 2, op: 3, audit: 4 };
export const podKey = (l: XLine) => l.pod_id || (l.source === "op" ? "operations" : l.source === "audit" ? "audit" : "—");
/** The rows to draw: the lines in display order (time, or level / source / pod then time), with group headers. */
export function buildRows(lines: XLine[], v: Pick<View, "order" | "sort" | "group">): Row[] {
  const time = (a: XLine, b: XLine) => (a.ts !== b.ts ? a.ts - b.ts : a.uid < b.uid ? -1 : a.uid > b.uid ? 1 : 0) * (v.order === "desc" ? -1 : 1);
  const by: ((a: XLine, b: XLine) => number)[] = [];
  if (v.group === "pod") by.push((a, b) => podKey(a).localeCompare(podKey(b)));
  if (v.group === "level") by.push((a, b) => LEVEL_RANK[a.level]! - LEVEL_RANK[b.level]!);
  if (v.sort === "level") by.push((a, b) => LEVEL_RANK[a.level]! - LEVEL_RANK[b.level]!);
  if (v.sort === "source") by.push((a, b) => SOURCE_RANK[a.source]! - SOURCE_RANK[b.source]!);
  if (v.sort === "pod") by.push((a, b) => podKey(a).localeCompare(podKey(b)));
  by.push(time);
  const sorted = by.length === 1 && isSorted(lines, time) ? lines : [...lines].sort((a, b) => {
    for (const f of by) {
      const c = f(a, b);
      if (c) return c;
    }
    return 0;
  });
  if (v.group === "none") return sorted.map((line) => ({ kind: "line", line }));
  const keyOf = v.group === "pod" ? podKey : (l: XLine) => l.level;
  const rows: Row[] = [];
  let cur: { kind: "group"; key: string; label: string; count: number } | null = null;
  for (const line of sorted) {
    const k = keyOf(line);
    if (!cur || cur.key !== k) {
      const label: string = v.group === "pod" ? groupLabel(line) : k.toUpperCase();
      cur = { kind: "group", key: k, label, count: 0 };
      rows.push(cur);
    }
    cur.count++;
    rows.push({ kind: "line", line });
  }
  return rows;
}
function isSorted(lines: XLine[], cmp: (a: XLine, b: XLine) => number) {
  for (let i = 1; i < lines.length; i++) if (cmp(lines[i - 1]!, lines[i]!) > 0) return false;
  return true;
}
const groupLabel = (l: XLine): string => (l.pod_id ? `${l.pod_name || l.pod_id}${l.pool ? ` · ${l.pool}` : ""}${l.cluster ? ` · ${l.cluster}` : ""}` : podKey(l));

/** Merges new lines into a buffer kept in ascending (ts, uid) order, without duplicates. */
export function mergeLines(buf: XLine[], add: XLine[], seen: Set<string>): XLine[] {
  const fresh = add.filter((l) => !seen.has(l.uid));
  if (!fresh.length) return buf;
  for (const l of fresh) seen.add(l.uid);
  const cmp = (a: XLine, b: XLine) => (a.ts !== b.ts ? a.ts - b.ts : a.uid < b.uid ? -1 : a.uid > b.uid ? 1 : 0);
  fresh.sort(cmp);
  // Common cases: everything after the end, or everything before the start.
  if (!buf.length || cmp(fresh[0]!, buf[buf.length - 1]!) > 0) return [...buf, ...fresh];
  if (cmp(fresh[fresh.length - 1]!, buf[0]!) < 0) return [...fresh, ...buf];
  return [...buf, ...fresh].sort(cmp);
}

// ---------------------------------------------------------------- selection
/** Line rows between two uids (inclusive), in row order; a single uid when only one is set. */
export function selectedRange(rows: Row[], anchor: string, focus: string): string[] {
  const idx = (u: string) => rows.findIndex((r) => r.kind === "line" && r.line.uid === u);
  const a = idx(anchor);
  const f = focus ? idx(focus) : a;
  if (a < 0 && f < 0) return [];
  const [lo, hi] = a < 0 ? [f, f] : f < 0 ? [a, a] : a <= f ? [a, f] : [f, a];
  const out: string[] = [];
  for (let i = lo; i <= hi; i++) {
    const r = rows[i]!;
    if (r.kind === "line") out.push(r.line.uid);
  }
  return out;
}
/** The next (dir 1) or previous (-1) matching line row after row `from` (wraps); -1 when none. */
export function nextMatch(rows: Row[], from: number, dir: 1 | -1, test: (l: XLine) => boolean): number {
  const n = rows.length;
  for (let k = 1; k <= n; k++) {
    const i = (((from + dir * k) % n) + n) % n;
    const r = rows[i]!;
    if (r.kind === "line" && test(r.line)) return i;
  }
  return -1;
}

// ---------------------------------------------------------------- format
export const iso = (ts: number) => new Date(ts).toISOString().replace("T", " ").replace("Z", "");
export const shortTime = (ts: number) => new Date(ts).toISOString().slice(11, 23);
export function formatLine(l: XLine): string {
  const where = [l.source, l.cluster, l.pool, l.pod_id].filter(Boolean).join(" ");
  return `${new Date(l.ts).toISOString()} ${l.level.toUpperCase().padEnd(5)} [${where}] ${l.target ? `${l.target}: ` : ""}${l.msg}${l.fields ? " " + JSON.stringify(l.fields) : ""}`;
}
/** A stable color slot (1-8, the chart series colors) per pod, cluster or other key. */
export function colorSlot(key: string): number {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 16777619);
  return (Math.abs(h) % 8) + 1;
}
/** "2026-10-06 12:00", "12:00" (today, UTC), unix ms, or relative ("10m" ago) → ms. */
export function parseJump(s: string, t = Date.now()): number | null {
  const v = s.trim();
  if (!v) return null;
  const rel = /^(\d+(?:\.\d+)?)\s*(s|m|h|d)$/.exec(v);
  if (rel) return t - Number(rel[1]) * ({ s: 1e3, m: 6e4, h: 3.6e6, d: 8.64e7 } as Record<string, number>)[rel[2]!]!;
  if (/^\d{12,}$/.test(v)) return Number(v);
  const hm = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(v);
  if (hm) {
    const d = new Date(t);
    return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), Number(hm[1]), Number(hm[2]), Number(hm[3] || 0));
  }
  const iso = /[zZ]|[+-]\d{2}:?\d{2}$/.test(v) ? v : `${v.replace(" ", "T")}Z`;
  const d = Date.parse(iso);
  return Number.isNaN(d) ? null : d;
}
/** "1h" → "last 1 h"; an absolute value → its UTC time. */
export function describeRange(from: string, to: string): string {
  const f = /^\d+(m|h|d)$/.test(from) ? `last ${from.replace(/(\d+)(\w)/, "$1 $2")}` : from ? `from ${from}` : "all of the tail";
  return to ? `${f} to ${to}` : f;
}

/** The API's cursor format (src/logquery.ts encodeCursor): a position (ts, uid) to page from. */
export function encodeCursor(ts: number, uid: string): string {
  return btoa(JSON.stringify([ts, uid])).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
}
