// Pod logs from boot (docs/control/README.md §6): fv-control copies the
// Runpod container and system log of every controller pod (cluster workers
// and standalone pods) into its own log store, from the pod's first line.
//
// fv-serve's log shipping (log_ship.rs) only starts once fv-serve has read
// its config and set up tracing, and images built before 2026-09-29 have no
// shipping at all. An image pull that fails, a wrong entrypoint, or a config
// error that exits fv-serve (the 2026-10-06 h3-and-ltx crash loop) never
// reaches the ingest. Runpod keeps only a ~70-line tail and forgets it when
// the pod is deleted, so the controller reads that tail while the pod lives:
// every minute from the cron, and every 20 s while an `up` waits for the pod.
// Lines land in D1 `log_lines` (target runpod.container / runpod.system) and
// in R2 next to the shipped lines, with a per-pod cursor so none is stored
// twice.
//
// The same tail drives the boot diagnosis (`diagnose`): pulling, starting,
// running, a crash loop, an image error, or stuck. `up` uses it to fail a
// pool early instead of waiting 30 min.
import { bootPhase, getBoot, recordBoot, type Boot } from "./boottime";
import { stubFor } from "./cluster/control";
import type { Env } from "./env";
import { runpod } from "./runpod";
import { now, parseJson, scrub, utcDay } from "./util";

export type Stream = "container" | "system";
export interface RawLine {
  ts: number; // ms (0: no timestamp on the line)
  text: string;
}
// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-9;]*[A-Za-z]/g;

/** "2026-10-06T20:41:02.230004105Z text" → {ts, text}; a line without a timestamp keeps ts 0. */
export function parseRunpodLine(s: string): RawLine {
  const clean = String(s).replace(ANSI, "").replace(/\r$/, "");
  const m = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})\s?(.*)$/s.exec(clean);
  if (!m) return { ts: 0, text: clean };
  const ts = Date.parse(`${m[1]}.${(m[2] || "0").padEnd(3, "0").slice(0, 3)}${m[3]}`);
  return Number.isFinite(ts) ? { ts, text: m[4]! } : { ts: 0, text: clean };
}

export interface Cursor {
  ts: number; // the newest timestamp stored
  n: number; // how many lines with exactly that timestamp were stored
}
/** The lines of a tail not stored yet, and the cursor after them. A line without a timestamp takes the one before it. */
export function newSince(lines: RawLine[], cur: Cursor): { fresh: RawLine[]; cursor: Cursor } {
  const fresh: RawLine[] = [];
  let seenAtCur = 0;
  let next = { ...cur };
  let prevTs = 0;
  for (const l0 of lines) {
    const l = { ...l0, ts: l0.ts || prevTs };
    prevTs = l.ts;
    if (l.ts < cur.ts) continue;
    if (l.ts === cur.ts) {
      seenAtCur++;
      if (seenAtCur <= cur.n) continue;
    }
    fresh.push(l);
    if (l.ts > next.ts) next = { ts: l.ts, n: 1 };
    else if (l.ts === next.ts) next = { ts: next.ts, n: next.n + 1 };
  }
  return { fresh, cursor: next };
}

/** A container line's level: fv-serve's JSON lines carry theirs; otherwise a guess from the text. */
export function lineLevel(text: string, stream: Stream): { level: string; msg: string; fields?: Record<string, unknown> } {
  if (text.startsWith("{")) {
    const j = parseJson<any>(text, null);
    if (j && typeof j === "object") {
      const lv = String(j.level || "info").toLowerCase();
      const fields = j.fields && typeof j.fields === "object" ? { ...j.fields } : {};
      const msg = String(j.msg ?? j.message ?? fields.message ?? text);
      delete fields.message;
      return { level: ["trace", "debug", "info", "warn", "error"].includes(lv) ? lv : "info", msg, fields: Object.keys(fields).length ? fields : undefined };
    }
  }
  if (isFatalLine(text, stream) || /\b(ERROR|FATAL|CRITICAL)\b|panicked at/.test(text)) return { level: "error", msg: text };
  if (/\bWARN(ING)?\b/.test(text)) return { level: "warn", msg: text };
  return { level: "info", msg: text };
}

// A container line that means the process (or its start) failed.
const CONTAINER_FATAL = [
  /^fv-serve: /, // fv-serve's main() prints "fv-serve: config: …" / "fv-serve: <error>" and exits
  /exec format error/i,
  /^(bash|sh|\/bin\/sh)(: line \d+)?: .*(No such file or directory|command not found|Permission denied)/,
  /panicked at /,
  /CUDA driver version is insufficient|no CUDA-capable device|CUDA_ERROR_NO_DEVICE/,
  /Segmentation fault|core dumped|Killed$/,
];
// A system line that means the container cannot be created or its image pulled.
const SYSTEM_FATAL = /failed to pull|manifest unknown|pull access denied|unauthorized: |denied: |no space left on device|error creating container|failed to create (shim|container)|image not found/i;
export function isFatalLine(text: string, stream: Stream): boolean {
  return stream === "system" ? SYSTEM_FATAL.test(text) : CONTAINER_FATAL.some((re) => re.test(text));
}

/** Container output missing this long after create: the pod is stuck (pull or host). */
export const STUCK_MS = 15 * 60_000;

/**
 * An image pull still running this long after it started: the host is slow
 * (pulls of the same ~1 GB image took 21 s on one EUR-IS-1 host and 10-19
 * min on others): `up` deletes the pod and places a new one.
 */
export const PULL_DEADLINE_MS = 6 * 60_000;

export type BootPhase = "pulling" | "starting" | "running" | "error" | "crashloop" | "image_error" | "stuck" | "slow_pull";
export interface Diagnosis {
  phase: BootPhase;
  detail: string;
  /** The pod will not become ready: stop waiting for it. */
  fatal: boolean;
  /** Fatal for this pod but not for its pool: place a new pod (slow pull). */
  replace?: boolean;
  at?: number;
}
/** What a pod's boot looks like from its Runpod log tail, uptime and timeline. */
export function diagnose(x: { ageMs: number; uptimeS?: number | null; container: string[]; system: string[]; boot?: Boot | null; now?: number }): Diagnosis {
  const t = x.now ?? now();
  const ps = x.boot?.t.pull_start;
  if (ps !== undefined && x.boot?.t.pull_end === undefined && x.boot?.t.container_start === undefined && t - ps > PULL_DEADLINE_MS)
    return { phase: "slow_pull", detail: `image pull still running ${Math.round((t - ps) / 60000)} min after it started (a slow host): placing a new pod`, fatal: true, replace: true };
  const sysFatal = x.system.filter((l) => isFatalLine(l, "system"));
  if (sysFatal.length) return { phase: "image_error", detail: sysFatal[sysFatal.length - 1]!.slice(0, 300), fatal: true };
  const fatal = x.container.filter((l) => isFatalLine(l, "container"));
  if (fatal.length >= 2) return { phase: "crashloop", detail: `crash loop (${fatal.length}× in the log tail): ${fatal[fatal.length - 1]!.slice(0, 300)}`, fatal: true };
  if (fatal.length === 1) return { phase: "error", detail: fatal[0]!.slice(0, 300), fatal: false };
  const lastSys = x.system[x.system.length - 1] || "";
  if (!x.container.length) {
    const up = typeof x.uptimeS === "number" && x.uptimeS > 0;
    if (!up && x.ageMs > STUCK_MS) return { phase: "stuck", detail: `no container output ${Math.round(x.ageMs / 60000)} min after create (last system line: ${lastSys.slice(0, 200) || "none"})`, fatal: true };
    const pulling = /Pulling|Downloading|Extracting|Waiting|Pull complete|Verifying/.test(lastSys) && !x.system.some((l) => /start container/.test(l));
    return { phase: pulling ? "pulling" : "starting", detail: lastSys.slice(0, 200), fatal: false };
  }
  return { phase: "running", detail: x.container[x.container.length - 1]!.slice(0, 200), fatal: false };
}

interface CursorRow {
  pod_id: string;
  container_ts: number;
  container_n: number;
  system_ts: number;
  system_n: number;
  diag: string | null;
}
export interface Capture {
  added: number;
  /** The whole tail (texts), for diagnose. */
  container: string[];
  system: string[];
}
/**
 * Copies the new lines of a pod's Runpod log tail into the log store. Null
 * when Runpod has no log for it (the pod is gone). `broadcast`: also to the
 * cluster's live-tail sockets (not from inside the cluster's own Durable Object).
 */
export async function captureRunpodLogs(env: Env, podId: string, clusterId: string, opts: { broadcast?: boolean } = {}): Promise<Capture | null> {
  let tail: { container: string[]; system: string[] };
  try {
    tail = await runpod.logs(env, podId);
  } catch {
    return null;
  }
  const cur = await env.DB.prepare("SELECT * FROM pod_log_cursors WHERE pod_id = ?").bind(podId).first<CursorRow>();
  // Scrubbed once, here: what is stored, the timeline and the diagnosis (which quotes lines) never carry a secret.
  const c = parse(tail.container.map((x) => scrub(env, String(x))));
  const s = parse(tail.system.map((x) => scrub(env, String(x))));
  const nc = newSince(c, { ts: cur?.container_ts ?? 0, n: cur?.container_n ?? 0 });
  const ns = newSince(s, { ts: cur?.system_ts ?? 0, n: cur?.system_n ?? 0 });
  const t = now();
  const rows = [
    ...ns.fresh.map((l) => ({ ...l, stream: "system" as const })),
    ...nc.fresh.map((l) => ({ ...l, stream: "container" as const })),
  ].map((l) => {
    const lv = lineLevel(l.text, l.stream);
    return { ts: l.ts || t, level: lv.level, target: `runpod.${l.stream}`, msg: lv.msg.slice(0, 8192), fields: lv.fields };
  });
  // The boot timeline (boottime.ts) from the same lines.
  await recordBoot(env, podId, clusterId, [...ns.fresh.map((l) => ({ stream: "system" as const, ts: l.ts, text: l.text })), ...nc.fresh.map((l) => ({ stream: "container" as const, ts: l.ts, text: l.text }))]).catch(() => []);
  if (rows.length) {
    rows.sort((a, b) => a.ts - b.ts);
    const first = rows[0]!.ts;
    const key = `logs/${clusterId}/${podId}/${utcDay(first)}/${new Date(first).toISOString().slice(11, 13)}/${t}-runpod.ndjson`;
    await env.LOGS.put(key, rows.map((r) => JSON.stringify(r)).join("\n") + "\n", { httpMetadata: { contentType: "application/x-ndjson" } }).catch(() => {});
    const stmt = env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg, fields) VALUES (?, ?, ?, ?, ?, ?, ?)");
    const stmts = rows.map((r) => stmt.bind(clusterId, podId, r.ts, r.level, r.target, r.msg, r.fields ? JSON.stringify(r.fields).slice(0, 8192) : null));
    for (let i = 0; i < stmts.length; i += 100) await env.DB.batch(stmts.slice(i, i + 100));
    if (opts.broadcast) await stubFor(env, clusterId).fetch("https://ops/broadcast", { method: "POST", body: JSON.stringify({ pod: podId, lines: rows }) }).catch(() => null);
  }
  await env.DB.prepare(
    `INSERT INTO pod_log_cursors (pod_id, cluster_id, container_ts, container_n, system_ts, system_n, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (pod_id) DO UPDATE SET container_ts = excluded.container_ts, container_n = excluded.container_n, system_ts = excluded.system_ts, system_n = excluded.system_n, updated_at = excluded.updated_at`,
  )
    .bind(podId, clusterId, nc.cursor.ts, nc.cursor.n, ns.cursor.ts, ns.cursor.n, t)
    .run();
  return { added: rows.length, container: c.map((l) => l.text), system: s.map((l) => l.text) };
}
const parse = (xs: string[]) => xs.map(parseRunpodLine);

/** Records a pod's latest boot diagnosis (GET /api/pods/:id/status). */
export async function saveDiagnosis(env: Env, podId: string, clusterId: string, d: Diagnosis): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO pod_log_cursors (pod_id, cluster_id, diag, updated_at) VALUES (?, ?, ?, ?)
     ON CONFLICT (pod_id) DO UPDATE SET diag = excluded.diag, updated_at = excluded.updated_at`,
  )
    .bind(podId, clusterId, JSON.stringify({ ...d, at: d.at ?? now() }), now())
    .run();
}
export async function getDiagnosis(env: Env, podId: string): Promise<Diagnosis | null> {
  const r = await env.DB.prepare("SELECT diag FROM pod_log_cursors WHERE pod_id = ?").bind(podId).first<{ diag: string | null }>();
  return r?.diag ? parseJson<Diagnosis | null>(r.diag, null) : null;
}

/** Capture + diagnosis for one pod (age from its create time). */
export async function checkPod(env: Env, podId: string, clusterId: string, createdMs: number, opts: { broadcast?: boolean; uptimeS?: number | null } = {}): Promise<Diagnosis | null> {
  const cap = await captureRunpodLogs(env, podId, clusterId, opts);
  if (!cap) return null;
  const boot = await getBoot(env, podId);
  const d = diagnose({ ageMs: now() - createdMs, uptimeS: opts.uptimeS, container: cap.container, system: cap.system, boot });
  // Where the boot is: the last phase reached and for how long.
  const ph = bootPhase(boot);
  if (ph && !d.fatal) d.detail = `${ph.phase} ${ph.since_s} s ago${d.detail ? `; ${d.detail}` : ""}`;
  await saveDiagnosis(env, podId, clusterId, d);
  return d;
}
