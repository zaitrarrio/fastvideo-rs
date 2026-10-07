// Serverless endpoints fv-control manages (docs/control/serverless.md):
// create, update, scale, extend, delete, test invoke, and the per-minute
// tick (health, deadline and floor backstops, cost from Runpod billing,
// worker log capture). Only endpoints recorded in D1 serverless_endpoints
// are ever touched; every mutation is audited.
import { policies, syncAlerts, type AlertIn } from "../alerts";
import type { ClusterSpec } from "../cluster/spec";
import type { Env } from "../env";
import { resolveClusterImages } from "../ghcr";
import { audit, getSetting, HttpError, newId, now, parseJson, putSetting, scrub, utcDay } from "../util";
import { endpointCreatePayload, endpointName, endpointUpdatePayload, invokeBody, lbPools, SLS_PREFIX, templateCreatePayload, templateName, templateUpdatePayload, v2CreatePayload, v2TemplateBoot } from "./payloads";
import { balanceFloor, sls, type LiveEndpoint, type SlsEnv } from "./runpod-sls";
import { normalizeEndpointSpec, specDiff, type EndpointSpec } from "./spec";
import { defaultCancelPath } from "../jobapi";

export interface SlsRow {
  id: string;
  name: string;
  endpoint_id: string | null;
  template_id: string | null;
  own_template: number;
  mode: "queue" | "lb";
  spec: string;
  image: string | null;
  status: string;
  deadline: number | null;
  deadline_action: string | null;
  health: string | null;
  health_at: number | null;
  workers: number | null;
  live_dph: number | null;
  cost_usd: number;
  billed_ms: number;
  billed_at: number | null;
  created_at: number;
  created_by: string;
  updated_at: number;
  deleted_at: number | null;
  last_error: string | null;
}
export type Actor = { actor: string; ip?: string };

/** Account-level limits for serverless endpoints (settings key `serverless`). */
export interface SlsPolicy {
  /** create / scale-up / invoke need balance >= the account floor + this ($). */
  balance_margin: number;
  /** Live endpoints at once. */
  max_endpoints: number;
  /** Sum of workers_max over live endpoints. */
  max_workers: number;
  /** Below the floor the tick scales every endpoint to 0/0 (the clusters' stop_on_floor applies too). */
  scale0_on_floor: boolean;
}
export const DEFAULT_SLS_POLICY: SlsPolicy = { balance_margin: 2, max_endpoints: 4, max_workers: 8, scale0_on_floor: true };
export async function slsPolicy(env: Env): Promise<SlsPolicy> {
  const p = await getSetting<SlsPolicy>(env, "serverless", DEFAULT_SLS_POLICY);
  const n = (v: unknown, d: number, lo: number, hi: number) => (typeof v === "number" && Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : d);
  return {
    balance_margin: n(p.balance_margin, 2, 0, 1000),
    max_endpoints: Math.round(n(p.max_endpoints, 4, 0, 50)),
    max_workers: Math.round(n(p.max_workers, 8, 0, 200)),
    scale0_on_floor: p.scale0_on_floor !== false,
  };
}
export function normalizeSlsPolicy(x: Partial<SlsPolicy>): SlsPolicy {
  const d = DEFAULT_SLS_POLICY;
  const n = (v: unknown, dv: number, lo: number, hi: number) => (typeof v === "number" && Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : dv);
  return { balance_margin: n(x.balance_margin, d.balance_margin, 0, 1000), max_endpoints: Math.round(n(x.max_endpoints, d.max_endpoints, 0, 50)), max_workers: Math.round(n(x.max_workers, d.max_workers, 0, 200)), scale0_on_floor: x.scale0_on_floor !== false };
}

export const LIVE = "deleted_at IS NULL AND status NOT IN ('deleted', 'gone', 'failed')";
export const specOf = (r: SlsRow): EndpointSpec => parseJson<EndpointSpec>(r.spec, {} as EndpointSpec);

// ---------------------------------------------------------------- ownership
/** One of fv-control's endpoints by its row id (se_…), Runpod endpoint id, or live name; 404 otherwise. */
export async function getRow(env: Env, key: string): Promise<SlsRow> {
  const r = await env.DB.prepare("SELECT * FROM serverless_endpoints WHERE id = ? OR endpoint_id = ? OR (name = ? AND deleted_at IS NULL) ORDER BY deleted_at IS NULL DESC, created_at DESC LIMIT 1")
    .bind(key, key, key)
    .first<SlsRow>();
  if (!r) throw new HttpError(404, "no serverless endpoint fv-control created with that id or name (fv-control only manages its own)");
  return r;
}
export async function listRows(env: Env, all = false): Promise<SlsRow[]> {
  const r = await env.DB.prepare(`SELECT * FROM serverless_endpoints ${all ? "" : "WHERE deleted_at IS NULL OR deleted_at > ?"} ORDER BY created_at DESC LIMIT 200`)
    .bind(...(all ? [] : [now() - 86400_000]))
    .all<SlsRow>();
  return r.results || [];
}
/** A Runpod endpoint fv-control may touch: recorded in D1 and named fvc-…; a second guard against acting on anyone else's. */
export function assertOurs(row: SlsRow, live?: { name?: string } | null): void {
  if (!row.endpoint_id) throw new HttpError(409, "the endpoint was never created on Runpod");
  if (live?.name && !String(live.name).startsWith(SLS_PREFIX)) throw new HttpError(403, `Runpod endpoint ${row.endpoint_id} is named ${live.name}, not ${SLS_PREFIX}…: refusing to touch it`);
}

// ---------------------------------------------------------------- money guards
/** create / scale-up / invoke: balance >= floor + margin (402 otherwise). */
export async function assertFloor(env: Env, what: string, balance?: number): Promise<number> {
  const pol = await slsPolicy(env);
  const b = balance ?? (await sls.live(env)).balance;
  const need = balanceFloor(env) + pol.balance_margin;
  if (!(b >= need)) throw new HttpError(402, `${what}: Runpod balance $${b.toFixed(2)} is below $${need.toFixed(2)} (floor $${balanceFloor(env)} + serverless margin $${pol.balance_margin})`);
  return b;
}
/** The account-wide limits: live endpoints and total workers_max. */
export function checkLimits(pol: SlsPolicy, live: { id: string; workers_max: number }[], next: { id?: string; workers_max: number }): void {
  const others = live.filter((r) => r.id !== next.id);
  if (!next.id && others.length + 1 > pol.max_endpoints) throw new HttpError(409, `serverless.max_endpoints: ${pol.max_endpoints} live endpoints already`);
  const total = others.reduce((s, r) => s + r.workers_max, 0) + next.workers_max;
  if (total > pol.max_workers) throw new HttpError(409, `serverless.max_workers: ${total} workers_max over every endpoint (limit ${pol.max_workers})`);
}
async function liveSizes(env: Env): Promise<{ id: string; workers_max: number }[]> {
  const r = await env.DB.prepare(`SELECT id, spec FROM serverless_endpoints WHERE ${LIVE}`).all<{ id: string; spec: string }>();
  return (r.results || []).map((x) => ({ id: x.id, workers_max: Number(parseJson<any>(x.spec, {}).workers_max ?? 0) }));
}

// ---------------------------------------------------------------- images
/** The spec's image, resolved to a digest the way clusters resolve theirs (ghcr.ts resolveClusterImages). */
export async function resolveImage(env: Env, spec: EndpointSpec): Promise<string> {
  const pseudo = { image: spec.image, pools: [{ id: "sls", variant: spec.variant }] } as unknown as ClusterSpec;
  const out = await resolveClusterImages(env, pseudo);
  return out.sls!;
}

// ---------------------------------------------------------------- rows
async function patchRow(env: Env, id: string, f: Partial<SlsRow>): Promise<void> {
  const keys = Object.keys(f);
  if (!keys.length) return;
  await env.DB.prepare(`UPDATE serverless_endpoints SET ${keys.map((k) => `${k} = ?`).join(", ")}, updated_at = ? WHERE id = ?`)
    .bind(...keys.map((k) => (f as any)[k]), now(), id)
    .run();
}
export const reload = (env: Env, id: string) => getRow(env, id);
const deadlineFor = (spec: EndpointSpec, from = now()) => (spec.deadline_min ? from + spec.deadline_min * 60_000 : null);

// ---------------------------------------------------------------- create
export async function createEndpoint(env: SlsEnv, who: Actor, input: unknown, o: { image?: string } = {}): Promise<SlsRow> {
  const spec = normalizeEndpointSpec(input);
  const pol = await slsPolicy(env);
  checkLimits(pol, await liveSizes(env), { workers_max: spec.workers_max });
  const exists = await env.DB.prepare("SELECT id FROM serverless_endpoints WHERE name = ? AND deleted_at IS NULL").bind(spec.name).first();
  if (exists) throw new HttpError(409, `a serverless endpoint named ${spec.name} exists`);
  await assertFloor(env, "create");
  const image = o.image ?? (await resolveImage(env, spec));
  const id = newId("se");
  const t = now();
  await env.DB.prepare("INSERT INTO serverless_endpoints (id, name, mode, spec, image, status, deadline, deadline_action, created_at, created_by, updated_at) VALUES (?, ?, ?, ?, ?, 'creating', ?, ?, ?, ?, ?)")
    .bind(id, spec.name, spec.mode, JSON.stringify(spec), image, deadlineFor(spec, t), spec.deadline_action, t, who.actor, t)
    .run();
  let tpl: string | null = null;
  let ep: string | null = null;
  try {
    if (spec.mode === "queue" && spec.compute === "GPU") {
      // GPU queue: a template, then the endpoint on it (REST v1; runpod-endpoint.sh's proven path).
      const tr = await sls.createTemplate(env, templateCreatePayload(spec, image, templateName(spec, t)));
      tpl = String(tr?.id || "");
      if (!tpl) throw new HttpError(502, "runpod: template create returned no id");
      await patchRow(env, id, { template_id: tpl, own_template: 1 });
      const er = await sls.createEndpoint(env, endpointCreatePayload(spec, tpl));
      ep = String(er?.id || "");
      if (!ep) throw new HttpError(502, "runpod: endpoint create returned no id");
    } else {
      // Load balancer, or CPU queue (REST v1 ignores computeType CPU): REST v2, which makes the template.
      const pools = spec.compute === "GPU" ? lbPools(spec.gpu_types || [], await sls.gpuCatalog(env).catch(() => [])) : [];
      const er = await sls.createLb(env, v2CreatePayload(spec, image, pools));
      ep = String(er?.id || "");
      if (!ep) throw new HttpError(502, "runpod: v2 endpoint create returned no id");
      const got = await sls.getEndpoint(env, ep);
      tpl = got?.templateId ?? er?.templateId ?? null;
      await patchRow(env, id, { template_id: tpl, own_template: tpl ? 1 : 0 });
      const boot = v2TemplateBoot(spec);
      if (boot) {
        if (!tpl) throw new HttpError(502, "runpod: the v2 endpoint has no template to give the config");
        await sls.updateTemplate(env, tpl, boot);
      }
    }
    // Runpod's create drops some fields (REST v1: flashboot false, workersMax 0; live 2026-10-06): set them again.
    await sls.patchEndpoint(env, ep, endpointUpdatePayload(spec));
    await patchRow(env, id, { endpoint_id: ep, status: "active" });
    await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.create", target: spec.name, after: { endpoint: ep, template: tpl, image, spec } });
    return await reload(env, id);
  } catch (e) {
    const msg = scrub(env, (e as Error).message).slice(0, 4000);
    // Nothing half-made stays behind: the endpoint, then the template.
    if (ep) await sls.deleteEndpoint(env, ep).catch(() => {});
    if (tpl) await sls.deleteTemplate(env, tpl).catch(() => {});
    await patchRow(env, id, { status: "failed", last_error: msg, deleted_at: now(), endpoint_id: ep });
    await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.create", target: spec.name, ok: false, detail: msg, after: { endpoint: ep, template: tpl, image } });
    throw e;
  }
}

// ---------------------------------------------------------------- update, scale, extend
export async function updateEndpoint(env: SlsEnv, who: Actor, row: SlsRow, input: unknown, o: { image?: string; action?: string } = {}): Promise<SlsRow> {
  if (row.deleted_at || !["active", "scaled-down"].includes(row.status)) throw new HttpError(409, `endpoint is ${row.status}`);
  assertOurs(row);
  const prev = specOf(row);
  const next = normalizeEndpointSpec({ ...(input as object), name: row.name });
  const d = specDiff(prev, next);
  if (d.recreate.length) throw new HttpError(409, `${d.recreate.join(", ")}: cannot change in place; delete the endpoint and create a new one`);
  const pol = await slsPolicy(env);
  if (d.scaleUp) {
    checkLimits(pol, await liveSizes(env), { id: row.id, workers_max: next.workers_max });
    await assertFloor(env, "scale up");
  }
  const live = await sls.getEndpoint(env, row.endpoint_id!);
  if (!live) {
    await patchRow(env, row.id, { status: "gone", deleted_at: now(), last_error: "Runpod has no such endpoint" });
    throw new HttpError(410, "Runpod no longer has this endpoint (marked gone)");
  }
  assertOurs(row, live);
  let image = row.image || "";
  const applied: string[] = [];
  if (d.template) {
    if (!row.template_id) throw new HttpError(409, "no template recorded for this endpoint");
    image = o.image ?? (await resolveImage(env, next));
    await sls.updateTemplate(env, row.template_id, templateUpdatePayload(next, image));
    applied.push("template");
  }
  if (d.endpoint) {
    await sls.patchEndpoint(env, row.endpoint_id!, endpointUpdatePayload(next));
    applied.push("endpoint");
  }
  const deadline = prev.deadline_min !== next.deadline_min ? deadlineFor(next) : row.deadline;
  const status = next.workers_max > 0 ? "active" : "scaled-down";
  await patchRow(env, row.id, { spec: JSON.stringify(next), image, deadline, deadline_action: next.deadline_action, status, last_error: null });
  await audit(env, { actor: who.actor, ip: who.ip, action: o.action || "serverless.update", target: row.name, before: prev, after: { ...next, applied, image } });
  return reload(env, row.id);
}
export async function scaleEndpoint(env: SlsEnv, who: Actor, row: SlsRow, s: { workers_min?: number; workers_max?: number }): Promise<SlsRow> {
  const prev = specOf(row);
  const next = { ...prev, ...(s.workers_min !== undefined ? { workers_min: Number(s.workers_min) } : {}), ...(s.workers_max !== undefined ? { workers_max: Number(s.workers_max) } : {}) };
  if (s.workers_max !== undefined && s.workers_min === undefined && next.workers_min > next.workers_max) next.workers_min = next.workers_max;
  return updateEndpoint(env, who, row, next, { action: "serverless.scale" });
}
export async function extendEndpoint(env: Env, who: Actor, row: SlsRow, minutes: number): Promise<SlsRow> {
  if (!(minutes >= 1 && minutes <= 7 * 1440)) throw new HttpError(400, "minutes: 1-10080");
  if (row.deleted_at) throw new HttpError(409, `endpoint is ${row.status}`);
  const deadline = Math.max(now(), row.deadline ?? now()) + minutes * 60_000;
  await patchRow(env, row.id, { deadline });
  await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.extend", target: row.name, before: { deadline: row.deadline }, after: { deadline, minutes } });
  return reload(env, row.id);
}

// ---------------------------------------------------------------- delete
/** Scales to 0/0 and deletes the endpoint, then its template when fv-control made it. A failure leaves
 * the row `deleting`: the tick retries every minute (Runpod refuses while a worker still runs). */
export async function deleteEndpoint(env: Env, who: Actor, row: SlsRow, reason = "by hand"): Promise<SlsRow> {
  if (row.status === "deleted") return row;
  if (row.status !== "deleting") {
    await patchRow(env, row.id, { status: "deleting" });
    await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.delete", target: row.name, before: { endpoint: row.endpoint_id, template: row.template_id, status: row.status }, detail: reason });
  }
  await finishDelete(env, await reload(env, row.id), who.actor);
  return reload(env, row.id);
}
export async function finishDelete(env: Env, row: SlsRow, actor: string): Promise<boolean> {
  try {
    if (row.endpoint_id) {
      const live = await sls.getEndpoint(env, row.endpoint_id);
      if (live) {
        assertOurs(row, live);
        await sls.patchEndpoint(env, row.endpoint_id, { workersMin: 0, workersMax: 0 }).catch(() => {});
        await sls.deleteEndpoint(env, row.endpoint_id);
      }
    }
    if (row.own_template && row.template_id) await sls.deleteTemplate(env, row.template_id);
    await patchRow(env, row.id, { status: "deleted", deleted_at: now(), workers: 0, live_dph: 0, last_error: null });
    await audit(env, { actor, action: "serverless.deleted", target: row.name, after: { endpoint: row.endpoint_id, template: row.own_template ? row.template_id : null } });
    return true;
  } catch (e) {
    await patchRow(env, row.id, { last_error: scrub(env, (e as Error).message).slice(0, 4000) });
    return false;
  }
}

// ---------------------------------------------------------------- test invoke
export const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);
export const clip = (env: Env, v: unknown, n = 8000) => (v === undefined || v === null ? null : scrub(env, typeof v === "string" ? v : JSON.stringify(v)).slice(0, n));
export interface InvokeIn {
  input?: unknown; // queue: the job input (default {kind: "info"})
  sync?: boolean; // queue: /runsync (default) or /run
  method?: string; // lb
  path?: string; // lb (default /ping)
  body?: unknown; // lb
}
export function readyWorkers(h: any): number {
  const w = h?.workers || {};
  return Number(w.idle || 0) + Number(w.ready || 0) + Number(w.running || 0);
}
/** An http job that waits on the fv-serve job it creates gets that API's cancel route as `cancel_path`
 * (unless it names one): a Runpod cancel then stops the fv-serve job too (the worker's job-stop). */
export function withCancelPath(input: any): any {
  const http = input?.kind === "http" || (input?.kind === undefined && typeof input?.path === "string");
  if (!http || !input.wait || input.cancel_path || String(input.method || "POST").toUpperCase() !== "POST") return input;
  const cp = defaultCancelPath(String(input.path || ""));
  return cp ? { ...input, cancel_path: cp } : input;
}
export async function invoke(env: SlsEnv, who: Actor, row: SlsRow, x: InvokeIn) {
  if (!["active"].includes(row.status) || !row.endpoint_id) throw new HttpError(409, `endpoint is ${row.status}${row.status === "scaled-down" ? " (scale it up first)" : ""}`);
  await assertFloor(env, "invoke");
  const spec = specOf(row);
  const health = await sls.health(env, row.endpoint_id).catch(() => null);
  const cold = health ? readyWorkers(health) === 0 : false;
  const t0 = now();
  if (row.mode === "lb") {
    const method = String(x.method || "GET").toUpperCase();
    const path = String(x.path || "/ping");
    if (!/^(GET|POST)$/.test(method) || !/^\/[A-Za-z0-9._~\/?=&%-]{0,300}$/.test(path)) throw new HttpError(400, "lb invoke: method GET|POST and a path starting with /");
    const r = await sls.lb(env, row.endpoint_id, method, path, x.body);
    const ins = await env.DB.prepare("INSERT INTO serverless_jobs (endpoint, route, status, cold, submitted_at, finished_at, wall_ms, input, output, actor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id")
      .bind(row.id, `lb:${method} ${path}`, `HTTP ${r.status}`, cold ? 1 : 0, t0, now(), r.ms, clip(env, x.body ?? null, 2000), clip(env, r.body), who.actor)
      .first<{ id: number }>();
    await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.invoke", target: row.name, after: { route: `${method} ${path}`, status: r.status, ms: r.ms } });
    return { job: ins?.id, status: `HTTP ${r.status}`, http_status: r.status, wall_ms: r.ms, cold, output: r.body };
  }
  const input = x.input ?? { kind: "info" };
  if (typeof input !== "object" || Array.isArray(input) || JSON.stringify(input).length > 65536) throw new HttpError(400, "input: a JSON object (64 KiB max)");
  const body = invokeBody(withCancelPath(input), spec.execution_timeout_s);
  const j = x.sync === false ? await sls.run(env, row.endpoint_id, body) : await sls.runsync(env, row.endpoint_id, body);
  const wall = now() - t0;
  const status = String(j?.status || "?");
  const done = TERMINAL.has(status);
  const ins = await env.DB.prepare(
    "INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, finished_at, delay_ms, exec_ms, wall_ms, worker_id, input, output, error, actor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
  )
    .bind(row.id, j?.id ?? null, x.sync === false ? "run" : "runsync", status, cold ? 1 : 0, t0, done ? now() : null, j?.delayTime ?? null, j?.executionTime ?? null, done ? wall : null, j?.workerId ?? null, clip(env, body.input, 2000), clip(env, j?.output), clip(env, j?.error, 2000), who.actor)
    .first<{ id: number }>();
  await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.invoke", target: row.name, after: { job: j?.id, status, wall_ms: wall, cold } });
  return { job: ins?.id, runpod_job: j?.id ?? null, status, done, cold, wall_ms: wall, delay_ms: j?.delayTime ?? null, exec_ms: j?.executionTime ?? null, worker_id: j?.workerId ?? null, output: j?.output ?? null, error: j?.error ? clip(env, j.error, 2000) : null };
}
/** Polls one of the endpoint's queue jobs (an invoke /runsync left running) and records the outcome. */
export async function pollJob(env: SlsEnv, row: SlsRow, jobRowId: number) {
  const jr = await env.DB.prepare("SELECT * FROM serverless_jobs WHERE id = ? AND endpoint = ?").bind(jobRowId, row.id).first<any>();
  if (!jr) throw new HttpError(404, "no such job of this endpoint");
  if (!jr.job_id || jr.finished_at || !row.endpoint_id) return jr;
  const j = await sls.status(env, row.endpoint_id, jr.job_id);
  const status = String(j?.status || jr.status);
  const done = TERMINAL.has(status);
  await env.DB.prepare("UPDATE serverless_jobs SET status = ?, delay_ms = COALESCE(?, delay_ms), exec_ms = COALESCE(?, exec_ms), worker_id = COALESCE(?, worker_id), output = COALESCE(?, output), error = COALESCE(?, error), finished_at = ?, wall_ms = ? WHERE id = ?")
    .bind(status, j?.delayTime ?? null, j?.executionTime ?? null, j?.workerId ?? null, clip(env, j?.output), clip(env, j?.error, 2000), done ? now() : null, done ? now() - jr.submitted_at : null, jr.id)
    .run();
  return env.DB.prepare("SELECT * FROM serverless_jobs WHERE id = ?").bind(jr.id).first<any>();
}
export async function recentJobs(env: Env, rowId: string, limit = 20) {
  const r = await env.DB.prepare("SELECT id, job_id, route, status, cold, submitted_at, finished_at, delay_ms, exec_ms, wall_ms, worker_id, input, output, error, actor FROM serverless_jobs WHERE endpoint = ? ORDER BY id DESC LIMIT ?").bind(rowId, limit).all<any>();
  return r.results || [];
}
/** Cold start: the queue wait (delayTime) of jobs submitted with no worker up; warm: the others. */
export function jobStats(jobs: { cold: number; delay_ms: number | null; exec_ms: number | null; wall_ms: number | null; status: string | null }[]) {
  const med = (xs: number[]) => (xs.length ? xs.sort((a, b) => a - b)[Math.floor((xs.length - 1) / 2)]! : null);
  const num = (xs: (number | null)[]) => xs.filter((x): x is number => typeof x === "number");
  const done = jobs.filter((j) => j.status === "COMPLETED" || /^HTTP 2/.test(j.status || ""));
  return {
    jobs: jobs.length,
    completed: done.length,
    failed: jobs.filter((j) => j.status === "FAILED" || j.status === "TIMED_OUT" || /^HTTP [45]/.test(j.status || "")).length,
    cold_start_ms: med(num(jobs.filter((j) => j.cold).map((j) => j.delay_ms ?? j.wall_ms))),
    warm_delay_ms: med(num(jobs.filter((j) => !j.cold).map((j) => j.delay_ms))),
    exec_ms: med(num(done.map((j) => j.exec_ms))),
  };
}

// ---------------------------------------------------------------- logs (into the shared log store)
/** A Runpod log line "2026-10-06T21:50:01.123456Z message" → {ts, msg}. */
export function parseRunpodLine(line: string, fallback: number): { ts: number; msg: string; level: string } {
  // fv-serve's tracing colours (ANSI SGR) are noise in the log store.
  line = line.replace(/\x1b\[[0-9;]*m/g, "");
  const m = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+(.*)$/.exec(line);
  const ts = m ? Date.parse(m[1]!.endsWith("Z") ? m[1]! : `${m[1]}Z`) : NaN;
  const msg = (m ? m[2]! : line).slice(0, 8192);
  const level = /\b(ERROR|error)\b/.test(msg) ? "error" : /\b(WARN|warn)\b/.test(msg) ? "warn" : "info";
  return { ts: Number.isFinite(ts) ? ts : fallback, msg, level };
}
/** The log store's source id of an endpoint's workers (log_lines.cluster_id; pod_id = the worker id). */
export const logSource = (row: Pick<SlsRow, "id">) => `serverless:${row.id}`;
/** Fetches a worker's Runpod log tail and replaces its rows in log_lines (idempotent; 24 h retention there). */
export async function captureLogs(env: Env, row: SlsRow, workerId: string, max = 300): Promise<{ container: string[]; system: string[]; stored: number }> {
  const l = await sls.logs(env, workerId);
  const s = (x: string[]) => x.slice(-1000).map((line) => scrub(env, String(line)));
  const container = s(l.container);
  const system = s(l.system);
  const t = now();
  const lines = [...system.slice(-50).map((x) => ({ ...parseRunpodLine(x, t), target: "runpod.system" })), ...container.slice(-max).map((x) => ({ ...parseRunpodLine(x, t), target: "runpod.container" }))];
  const src = logSource(row);
  const stmts = [env.DB.prepare("DELETE FROM log_lines WHERE cluster_id = ? AND pod_id = ?").bind(src, workerId)];
  for (const x of lines) stmts.push(env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg, fields) VALUES (?, ?, ?, ?, ?, ?, ?)").bind(src, workerId, x.ts, x.level, x.target, x.msg, JSON.stringify({ endpoint: row.endpoint_id, source: "runpod-serverless" })));
  for (let i = 0; i < stmts.length; i += 50) await env.DB.batch(stmts.slice(i, i + 50));
  return { container, system, stored: lines.length };
}

// ---------------------------------------------------------------- views
export function rowView(r: SlsRow) {
  const spec = specOf(r);
  return {
    id: r.id,
    name: r.name,
    runpod_name: endpointName(spec),
    endpoint_id: r.endpoint_id,
    template_id: r.template_id,
    own_template: !!r.own_template,
    mode: r.mode,
    status: r.status,
    image: r.image,
    deadline: r.deadline,
    deadline_action: r.deadline_action,
    health: parseJson<any>(r.health, null),
    health_at: r.health_at,
    workers: r.workers,
    live_dph: r.live_dph,
    cost_usd: r.cost_usd,
    billed_ms: r.billed_ms,
    billed_at: r.billed_at,
    created_at: r.created_at,
    created_by: r.created_by,
    updated_at: r.updated_at,
    deleted_at: r.deleted_at,
    last_error: r.last_error,
    spec,
    urls: r.endpoint_id ? (r.mode === "lb" ? { base: `https://${r.endpoint_id}.api.runpod.ai` } : { run: `https://api.runpod.ai/v2/${r.endpoint_id}/run`, runsync: `https://api.runpod.ai/v2/${r.endpoint_id}/runsync`, health: `https://api.runpod.ai/v2/${r.endpoint_id}/health` }) : null,
  };
}

// ---------------------------------------------------------------- the per-minute tick
export interface SlsTick {
  endpoints: number;
  actions: string[];
  billed?: number;
  error?: string;
}
const BILLING_EVERY_MS = 10 * 60_000;

/** Health, the deadline and floor backstops, retried deletes, billing and log capture for every endpoint
 * fv-control manages. Does nothing (no API call) when it has none. */
export async function serverlessTick(env: SlsEnv, o: { force_billing?: boolean } = {}): Promise<SlsTick> {
  const t = now();
  const rows = (await env.DB.prepare("SELECT * FROM serverless_endpoints WHERE deleted_at IS NULL OR deleted_at > ?").bind(t - 3 * 86400_000).all<SlsRow>()).results || [];
  const live = rows.filter((r) => !r.deleted_at);
  const actions: string[] = [];
  const alerts: AlertIn[] = [];
  if (!rows.length) {
    await syncAlerts(env, [], ["serverless"]);
    return { endpoints: 0, actions };
  }
  let view: { balance: number; endpoints: LiveEndpoint[] } | null = null;
  if (live.length) view = await sls.live(env);
  const byId = new Map((view?.endpoints || []).map((e) => [e.id, e]));
  const floor = balanceFloor(env);
  const pol = await slsPolicy(env);
  const stopOnFloor = (await policies(env)).stop_on_floor && pol.scale0_on_floor;
  for (const r of live) {
    try {
      if (r.status === "deleting") {
        const ok = await finishDelete(env, r, "policy:serverless_delete_retry");
        actions.push(`${r.name}: delete ${ok ? "done" : "retry later"}`);
        continue;
      }
      if (!r.endpoint_id || r.status === "creating") {
        // A create that never answered (the Worker died mid-call) for 10 min: failed.
        if (t - r.created_at > 10 * 60_000) await patchRow(env, r.id, { status: "failed", deleted_at: t, last_error: "create did not finish" });
        continue;
      }
      const le = byId.get(r.endpoint_id);
      if (view && !le) {
        await patchRow(env, r.id, { status: "gone", deleted_at: t, workers: 0, live_dph: 0, last_error: "Runpod no longer has this endpoint" });
        actions.push(`${r.name}: gone`);
        continue;
      }
      const pods = (le?.pods || []).filter((p) => p.desiredStatus !== "EXITED" && p.desiredStatus !== "TERMINATED");
      const dph = pods.reduce((s, p) => s + Number(p.costPerHr || 0), 0);
      const health = await sls.health(env, r.endpoint_id).catch((e) => ({ error: scrub(env, (e as Error).message).slice(0, 200) }));
      await patchRow(env, r.id, { health: JSON.stringify(health), health_at: t, workers: pods.length, live_dph: dph });
      // Worker logs into the shared log store (log_lines, source serverless:<id>).
      for (const p of pods.slice(0, 4)) await captureLogs(env, r, p.id, 200).catch(() => {});
      // Backstops: the deadline, then the balance floor.
      if (r.deadline && t >= r.deadline) {
        const who = { actor: "policy:serverless_deadline" };
        if (r.deadline_action === "delete") {
          await deleteEndpoint(env, who, r, "deadline backstop");
          actions.push(`${r.name}: deleted (deadline)`);
        } else if (r.status === "active") {
          await sls.patchEndpoint(env, r.endpoint_id, { workersMin: 0, workersMax: 0 });
          const sp = { ...specOf(r), workers_min: 0, workers_max: 0 };
          await patchRow(env, r.id, { status: "scaled-down", spec: JSON.stringify(sp), deadline: null });
          await audit(env, { actor: who.actor, action: "serverless.scale", target: r.name, after: { workers_min: 0, workers_max: 0 }, detail: "deadline backstop" });
          actions.push(`${r.name}: scaled to 0 (deadline)`);
        }
        alerts.push({ key: `serverless_deadline:${r.id}`, kind: "serverless", severity: "warn", target: r.name, message: `serverless ${r.name} passed its deadline: ${r.deadline_action === "delete" ? "deleted" : "scaled to 0"}`, action: r.deadline_action || undefined });
        continue;
      }
      if (view && view.balance < floor && stopOnFloor && r.status === "active") {
        await sls.patchEndpoint(env, r.endpoint_id, { workersMin: 0, workersMax: 0 });
        const sp = { ...specOf(r), workers_min: 0, workers_max: 0 };
        await patchRow(env, r.id, { status: "scaled-down", spec: JSON.stringify(sp) });
        await audit(env, { actor: "policy:balance_floor", action: "serverless.scale", target: r.name, after: { workers_min: 0, workers_max: 0 }, detail: `balance $${view.balance.toFixed(2)} below the floor $${floor}` });
        actions.push(`${r.name}: scaled to 0 (balance floor)`);
        alerts.push({ key: `serverless_floor:${r.id}`, kind: "serverless", severity: "critical", target: r.name, message: `balance $${view.balance.toFixed(2)} below the floor $${floor}: serverless ${r.name} scaled to 0`, action: "scale0" });
        continue;
      }
      if (r.deadline && r.deadline - t < 15 * 60_000) alerts.push({ key: `serverless_deadline:${r.id}`, kind: "serverless", severity: "info", target: r.name, message: `serverless ${r.name}: ${r.deadline_action === "delete" ? "deleted" : "scaled to 0"} at its deadline in ${Math.round((r.deadline - t) / 60000)} min (extend to keep it)` });
      if (specOf(r).workers_min > 0) alerts.push({ key: `serverless_min:${r.id}`, kind: "serverless", severity: "info", target: r.name, message: `serverless ${r.name} keeps ${specOf(r).workers_min} worker(s) always on (billed while idle)` });
    } catch (e) {
      const msg = scrub(env, (e as Error).message).slice(0, 300);
      await patchRow(env, r.id, { last_error: msg }).catch(() => {});
      alerts.push({ key: `serverless_error:${r.id}`, kind: "serverless", severity: "warn", target: r.name, message: `serverless ${r.name}: ${msg}` });
    }
  }
  // Cost: Runpod's billing per endpoint and UTC day (it lags; every 10 min), into the shared ledger.
  let billed: number | undefined;
  const last = await getSetting<{ at: number }>(env, "serverless_billing", { at: 0 });
  if (o.force_billing || t - last.at >= BILLING_EVERY_MS) {
    await putSetting(env, "serverless_billing", { at: t }, "cron");
    try {
      billed = await recordBilling(env, rows, t);
    } catch (e) {
      alerts.push({ key: "serverless_billing", kind: "serverless", severity: "info", message: `serverless billing: ${scrub(env, (e as Error).message).slice(0, 200)}` });
    }
  }
  await syncAlerts(env, alerts, ["serverless"]);
  return { endpoints: live.length, actions, ...(billed !== undefined ? { billed } : {}) };
}

/** Writes Runpod's billed spend of fv-control's endpoints into cost_daily (pod_id sls:<endpoint>, owner serverless:<name>). */
export async function recordBilling(env: Env, rows: SlsRow[], t = now()): Promise<number> {
  const ours = new Map(rows.filter((r) => r.endpoint_id).map((r) => [r.endpoint_id!, r]));
  if (!ours.size) return 0;
  const start = `${utcDay(t - 2 * 86400_000)}T00:00:00Z`;
  const recs = await sls.billing(env, start);
  const stmts: D1PreparedStatement[] = [];
  let n = 0;
  for (const b of recs) {
    const r = ours.get(b.endpointId);
    if (!r) continue;
    const day = b.time.slice(0, 10);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) continue;
    n++;
    stmts.push(
      env.DB.prepare(
        `INSERT INTO cost_daily (day, pod_id, cluster_id, owner, usd, minutes, idle_minutes) VALUES (?, ?, NULL, ?, ?, ?, 0)
         ON CONFLICT (day, pod_id) DO UPDATE SET usd = excluded.usd, minutes = excluded.minutes, owner = excluded.owner`,
      ).bind(day, `sls:${b.endpointId}`, `serverless:${r.name}`, b.amount, Math.round(b.timeBilledMs / 60000)),
    );
  }
  for (let i = 0; i < stmts.length; i += 50) await env.DB.batch(stmts.slice(i, i + 50));
  for (const r of ours.values()) {
    const s = await env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS usd, COALESCE(SUM(minutes), 0) AS minutes FROM cost_daily WHERE pod_id = ?").bind(`sls:${r.endpoint_id}`).first<{ usd: number; minutes: number }>();
    const billedMs = recs.filter((b) => b.endpointId === r.endpoint_id).reduce((a, b) => a + b.timeBilledMs, 0);
    await patchRow(env, r.id, { cost_usd: s?.usd ?? 0, billed_ms: Math.max(billedMs, (s?.minutes ?? 0) * 60000), billed_at: t });
  }
  return n;
}
