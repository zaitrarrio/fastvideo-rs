// Boot timeline (docs/control/README.md §6a): when each phase of a serve
// pod's boot happened, from the lines the controller already sees: the
// Runpod system log (machine, image pull, container start), the container
// log (the boot script's [fv-boot] markers, fv-serve's start, the engine's
// per-component load/io lines, model resident, warm-up, READY) and the
// lines fv-serve ships. The DO adds `ready` (ready at the edge, or a direct
// worker's /health). Stored on the pod's record (cluster_pods.boot); every
// new milestone is also written to the pod's log (target fv-control.boot).
//
// Measured 2026-10-06 (h3-turbo, latest = b51ddcd, RTX PRO 6000, EUR-IS-1):
// pull 21 s, container start +24 s, model load 181 s (text encoder 41 s,
// DiT 170 s at ~0.6 GB/s from the network volume), warm-up 60 s, ready at
// the edge ~4.6 min after create.
//
// Fast boot B (2026-10-07): fv-serve reports ready once the weights are
// resident and warms up in the background ("warmup started (background)",
// "warmup done (background) … seconds="), so `serve_ready` / `ready` come
// before `warmup_start` / `warmup_done`; rows are in time order and the
// warm-up rows say "(background)". Measured 2026-10-07 (h3-turbo, d708c91,
// RTX PRO 6000, EUR-IS-1): READY +158 s with the background warm-up, +88 s
// with the pre-quantized DiT tree too (docs/gaps/2026-10-07-fast-boot.md).
import type { Env } from "./env";
import { now, parseJson } from "./util";

/** The phases in boot order (an `at` per phase, ms). */
export const MILESTONES = [
  "create", // Runpod accepted the create (the controller's record)
  "machine", // a host took it: Runpod's first system line ("create container …")
  "pull_start",
  "pull_end", // "Digest: …" / "Status: … up to date"
  "container_start", // "start container …: begin"
  "boot_script", // [fv-boot] start (the boot command runs)
  "volume", // [fv-boot] volume: the weights tree is visible
  "serve_start", // fv-serve's first line
  "edge_link", // connected to the edge's family dispatcher (edge fronts)
  "load_start", // the engine starts loading the model
  "model_resident", // every component loaded
  "warmup_start", // background warm-up began (after ready; fast boot B)
  "warmup_done",
  "serve_ready", // FV-SERVE READY
  "ready", // ready at the edge (or /health AVAILABLE): what the controller waits for
] as const;
export type Milestone = (typeof MILESTONES)[number];

export interface Component {
  wall_s: number;
  gb?: number;
  gbps?: number;
  at: number;
}
export interface Boot {
  t: Partial<Record<Milestone, number>>;
  /** Per weight component (text_encoder, dit, vision_tower, …): load wall time and volume throughput. */
  components?: Record<string, Component>;
  load_s?: number;
  warmup_s?: number;
  /** `background`: the warm-up ran after ready (fast boot B); else it held readiness. */
  warmup_mode?: "background" | "blocking";
  volume?: string;
}

/** What one log line says about the boot: a milestone, or a component load. */
export function bootMatch(stream: "system" | "container" | "shipped", text: string): { m?: Milestone; component?: [string, Omit<Component, "at">]; load_s?: number; warmup_s?: number; warmup_mode?: "background" | "blocking"; volume?: string } | null {
  if (stream === "system") {
    if (/^create container /.test(text)) return { m: "machine" };
    if (/ Pulling from /.test(text)) return { m: "pull_start" };
    if (/^Digest: sha256:|^Status: (Downloaded newer image|Image is up to date)/.test(text)) return { m: "pull_end" };
    if (/^start container .*: begin/.test(text)) return { m: "container_start" };
    return null;
  }
  let x: RegExpExecArray | null;
  if (/\[fv-boot\] start/.test(text)) return { m: "boot_script" };
  if ((x = /\[fv-boot\] volume: (.*)$/.exec(text))) return { m: "volume", volume: x[1]!.slice(0, 200) };
  if (/\bfv_serve: fv-serve \d|^fv_serve: fv-serve /.test(text)) return { m: "serve_start" };
  if (/edge_link: worker: connected to the dispatcher/.test(text)) return { m: "edge_link" };
  if (/cuda::backend: loading\b/.test(text)) return { m: "load_start" };
  if ((x = /\[fastvideo\] load\/io \S+ (\S+) (\{.*\})/.exec(text))) {
    const j = parseJson<any>(x[2]!, null);
    if (j && typeof j.wall_s === "number" && x[1] !== "total") return { component: [x[1]!, { wall_s: j.wall_s, gb: j.viewed_gb, gbps: j.viewed_gbps }] };
    return null;
  }
  if ((x = /\] \S+ \S+ encoder: .*?vision tower ([\d.]+) GiB loaded in ([\d.]+) s/.exec(text))) return { component: ["vision_tower", { wall_s: Number(x[2]), gb: Number(x[1]) * 1.073741824 }] };
  if ((x = /model resident\b.*?seconds=([\d.]+)/.exec(text))) return { m: "model_resident", load_s: Number(x[1]) };
  if (/warmup started \(background\)/.test(text)) return { m: "warmup_start", warmup_mode: "background" };
  if ((x = /warmup done\b.*?seconds=([\d.]+)/.exec(text))) return { m: "warmup_done", warmup_s: Number(x[1]), warmup_mode: /warmup done \(background\)/.test(text) ? "background" : "blocking" };
  if (/^FV-SERVE READY|fastvideo_serve::app: ready\b/.test(text)) return { m: "serve_ready" };
  return null;
}

/** Folds lines into a timeline; returns the milestones that are new (first seen wins). */
export function foldBoot(boot: Boot, lines: { stream: "system" | "container" | "shipped"; ts: number; text: string }[]): Milestone[] {
  const added: Milestone[] = [];
  for (const l of lines) {
    const r = bootMatch(l.stream, l.text);
    if (!r || !l.ts) continue;
    if (r.m && boot.t[r.m] === undefined) {
      boot.t[r.m] = l.ts;
      added.push(r.m);
    }
    if (r.component && !boot.components?.[r.component[0]]) (boot.components ||= {})[r.component[0]] = { ...r.component[1], at: l.ts };
    if (r.load_s !== undefined && boot.load_s === undefined) boot.load_s = r.load_s;
    if (r.warmup_s !== undefined && boot.warmup_s === undefined) boot.warmup_s = r.warmup_s;
    if (r.warmup_mode && (!boot.warmup_mode || r.warmup_mode === "background")) boot.warmup_mode = r.warmup_mode;
    if (r.volume && !boot.volume) boot.volume = r.volume;
  }
  return added;
}

export interface BootRow {
  phase: string;
  at: number | null;
  /** Seconds after create. */
  t_s: number | null;
  /** Seconds this phase took (from the phase before it that is known). */
  took_s: number | null;
  detail?: string;
}
const PHASE_LABEL: Record<Milestone, string> = {
  create: "Runpod create accepted",
  machine: "machine assigned",
  pull_start: "image pull start",
  pull_end: "image pull end",
  container_start: "container start",
  boot_script: "boot script start",
  volume: "volume mounted and visible",
  serve_start: "fv-serve process start",
  edge_link: "connected to the edge dispatcher",
  load_start: "weights: load start",
  model_resident: "weights: every component resident",
  warmup_start: "warm-up started (background)",
  warmup_done: "warm-up done",
  serve_ready: "fv-serve READY",
  ready: "ready (at the edge / health)",
};
/** The timeline as rows in boot order, with time since create and each phase's duration; components follow the load. */
export function bootRows(boot: Boot | null): BootRow[] {
  if (!boot) return [];
  const t0 = boot.t.create ?? Math.min(...Object.values(boot.t).filter((x): x is number => typeof x === "number"));
  const rows: BootRow[] = [];
  let prev: number | null = null;
  // Time order (a background warm-up ends after ready); phases not reached yet keep their place at the end.
  const order = MILESTONES.map((m, i) => ({ m, i, at: boot.t[m] })).sort((a, b) => (a.at ?? Infinity) - (b.at ?? Infinity) || a.i - b.i).map((o) => o.m);
  for (const m of order) {
    const at = boot.t[m] ?? null;
    const detail =
      m === "pull_end" && boot.t.pull_start !== undefined && at !== null ? `pull ${Math.round((at - boot.t.pull_start) / 1000)} s` :
      m === "volume" ? boot.volume :
      m === "model_resident" && boot.load_s !== undefined ? `load ${boot.load_s.toFixed(1)} s` :
      m === "warmup_done" && boot.warmup_s !== undefined ? `warm-up ${boot.warmup_s.toFixed(1)} s` : undefined;
    const label = m === "warmup_done" && boot.warmup_mode === "background" ? "warm-up done (background)" : PHASE_LABEL[m];
    rows.push({ phase: label, at, t_s: at !== null && Number.isFinite(t0) ? Math.round((at - t0) / 100) / 10 : null, took_s: at !== null && prev !== null ? Math.round((at - prev) / 100) / 10 : null, ...(detail ? { detail } : {}) });
    if (m === "load_start")
      for (const [name, c] of Object.entries(boot.components || {}).sort((a, b) => a[1].at - b[1].at))
        rows.push({ phase: `weights: ${name}`, at: c.at, t_s: Number.isFinite(t0) ? Math.round((c.at - t0) / 100) / 10 : null, took_s: c.wall_s, detail: [c.gb !== undefined ? `${c.gb.toFixed(1)} GB` : null, c.gbps !== undefined ? `${c.gbps.toFixed(2)} GB/s` : null].filter(Boolean).join(" at ") || undefined });
    if (at !== null) prev = at;
  }
  return rows;
}

async function loadBoot(env: Env, podId: string): Promise<{ boot: Boot; ready_at: number | null } | null> {
  const r = await env.DB.prepare("SELECT boot, created_at, ready_at FROM cluster_pods WHERE pod_id = ?").bind(podId).first<{ boot: string | null; created_at: number; ready_at: number | null }>();
  if (!r) return null;
  const boot = parseJson<Boot>(r.boot, { t: {} });
  boot.t ||= {};
  boot.t.create ??= r.created_at;
  return { boot, ready_at: r.ready_at };
}
/** A controller pod's timeline (null: not a controller pod). */
export async function getBoot(env: Env, podId: string): Promise<Boot | null> {
  const r = await loadBoot(env, podId);
  if (!r) return null;
  if (r.ready_at && r.boot.t.ready === undefined) r.boot.t.ready = r.ready_at;
  return r.boot;
}
/** The last phase a pod reached, and how long ago: where a slow or stuck boot is. */
export function bootPhase(boot: Boot | null, t = now()): { phase: Milestone; since_s: number } | null {
  if (!boot) return null;
  let last: Milestone | null = null;
  for (const m of MILESTONES) if (boot.t[m] !== undefined && (last === null || boot.t[m]! >= boot.t[last]!)) last = m;
  return last ? { phase: last, since_s: Math.round((t - boot.t[last]!) / 1000) } : null;
}
/** Folds lines into a controller pod's timeline; writes each new milestone to the pod's log. No-op for a pod the controller does not own. */
export async function recordBoot(env: Env, podId: string, clusterId: string, lines: { stream: "system" | "container" | "shipped"; ts: number; text: string }[], extra?: Partial<Record<Milestone, number>>): Promise<Milestone[]> {
  const r = await loadBoot(env, podId);
  if (!r) return [];
  const boot = r.boot;
  const before = JSON.stringify(boot);
  const added = foldBoot(boot, lines);
  for (const [m, at] of Object.entries(extra || {}) as [Milestone, number][]) if (boot.t[m] === undefined) (boot.t[m] = at), added.push(m);
  if (JSON.stringify(boot) === before) return [];
  await env.DB.prepare("UPDATE cluster_pods SET boot = ? WHERE pod_id = ?").bind(JSON.stringify(boot), podId).run();
  if (added.length) {
    const t0 = boot.t.create!;
    const rows = bootRows(boot);
    const stmt = env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg, fields) VALUES (?, ?, ?, 'info', 'fv-control.boot', ?, ?)");
    await env.DB.batch(
      added.map((m) => {
        const row = rows.find((r) => r.at === boot.t[m] && r.phase.length) || null;
        const at = boot.t[m]!;
        return stmt.bind(clusterId, podId, at, `boot: ${m} at +${((at - t0) / 1000).toFixed(1)} s${row?.detail ? ` (${row.detail})` : ""}`, JSON.stringify({ milestone: m, t_s: (at - t0) / 1000 }));
      }),
    );
  }
  return added;
}
/** The DO / cron: the pod is ready (at the edge, or /health). */
export async function markReady(env: Env, podId: string, clusterId: string, at = now()): Promise<void> {
  await recordBoot(env, podId, clusterId, [], { ready: at });
}
