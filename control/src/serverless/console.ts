// The serverless console (docs/control/serverless.md "Console"): fv-serve's
// browser console (crates/fastvideo-serve/console, bundled unchanged by
// gen-configs.mjs) served by fv-control for one of its serverless endpoints
// at /serverless/<endpoint>/console, behind fv-control's login. The pages
// call their API under the same prefix, and fv-control answers each call:
//
//   pages, assets         the bundled files; the HTML gets three <meta> tags
//                         (common.js "Embedding": the prefix, the pages that
//                         are off, a note) and its /console links prefixed
//   GET capabilities,     a cached reply: one queue job (`kind: http`, GET)
//   /fal/schema[/…]       fills the cache; a stale entry is served and
//                         refreshed in the background only while a worker is
//                         up, so opening the console wakes no worker
//   GET /fv/v1/status     synthesised from Runpod's /health (one pool)
//   submits               a waiting `kind: http` job on /run (below); the
//                         reply names the Runpod job id, the id every later
//                         call of the page uses
//   status, result        Runpod's /status (no job) and the shared job store
//                         (JOBS_DB, fv-serve's D1 `jobs`); the final reply is
//                         the worker's own (the waiting job's output)
//   cancels               cancel.ts (Runpod cancel, then cancel_path or the
//                         API's own cancel route as a queue job)
//   uploads               R2 (the LOGS bucket, console-uploads/), read back
//                         by the worker through a signed public URL
//
// Why submits wait: a queue worker takes one job at a time
// (crates/fastvideo-deploy/src/runpod/worker.rs, concurrency 1) and Runpod
// scales a worker with no job down after idle_timeout_s, so an fv-serve job
// must run inside its queue job (`wait: true`) or its worker may stop under
// it. The waiting job reports `{state, poll_path}` as progress (dispatch.rs),
// which names the fv-serve job id; the page never needs it.
//
// Load-balancer endpoints: the same pages; the API calls pass through to
// https://<id>.api.runpod.ai with the Runpod key (server side), except the
// cached capabilities / schemas, the synthesised status and the uploads.
import type { Env } from "../env";
import { hmacB64url, randomToken, safeEqual, b64url, unb64 } from "../crypto";
import { audit, fetchWithTimeout, HttpError, now, parseJson, scrub } from "../util";
import { CONSOLE_ASSETS, CONSOLE_PAGES } from "./console-assets";
import { cancelSlsJob } from "./cancel";
import { assertFloor, readyWorkers, specOf, TERMINAL, withCancelPath, type Actor, type SlsRow } from "./ops";
import { endpointName } from "./payloads";
import { sls, slsBases, type SlsEnv } from "./runpod-sls";

/** Test knobs: how long a submit waits for a warm worker's own reply, a cache miss for its job (ms). */
export type ConsoleEnv = SlsEnv & { CONSOLE_SUBMIT_WAIT_MS?: string; CONSOLE_CACHE_WAIT_MS?: string; CONSOLE_POLL_MS?: string };

/** fv-serve's console CSP (crates/fastvideo-serve/src/console.rs CSP). */
export const CONSOLE_CSP =
  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: http: https:; media-src 'self' data: blob: http: https:; connect-src 'self' http: https:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'";
/** Pages a serverless endpoint does not serve in v1: live sessions (WebRTC, WHIP) need a reachable server, API keys are fv-control's. */
export const OFF_PAGES = ["stream", "live", "avatar", "director", "admin"];
/** Cached GETs: these paths only (the console's catalog calls). */
const CACHED = (p: string) => p === "/fv/v1/capabilities" || p === "/fal/schema" || /^\/fal\/schema\/[A-Za-z0-9._~/-]{1,200}$/.test(p);
const CACHE_TTL_MS = 30 * 60_000;
const UPLOAD_MAX = 64 << 20; // fv-serve's default body_max_mb
const BODY_MAX = 8 << 20; // a /run payload is at most 10 MB
const OUTPUT_MAX = 256 << 10;
export const UPLOAD_PREFIX = "console-uploads/";

export type Api = "native" | "openai_videos" | "minimax_v2" | "fal";
const n = (v: string | undefined, d: number) => (v !== undefined && v !== "" && Number.isFinite(Number(v)) ? Number(v) : d);

// ---------------------------------------------------------------- replies
const json = (code: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status: code, headers: { "content-type": "application/json", "cache-control": "no-store", ...headers } });
/** fv-serve's error shape (the console reads error.message). */
const err = (code: number, kind: string, message: string, headers: Record<string, string> = {}) => json(code, { error: { kind, message } }, headers);
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pageHeaders = (type: string, html: boolean): Record<string, string> => ({
  "content-type": type,
  "cache-control": "no-cache",
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
  ...(html ? { "content-security-policy": CONSOLE_CSP, "x-frame-options": "DENY" } : {}),
});

// ---------------------------------------------------------------- pages
export function consoleNote(row: SlsRow): string {
  const live = row.mode === "lb" ? "Requests go to its load balancer through fv-control." : "Each request runs as a Runpod queue job; with no worker up the first one waits for a cold start (minutes), and the status strip says so.";
  return `Serverless endpoint ${endpointName(specOf(row))} (${row.endpoint_id ?? "not created"}) through fv-control. ${live} Live pages (stream, live input, avatar, director) and API keys are not available on a serverless endpoint.`;
}
/** A console page with the embedding <meta> tags and its /console links under `prefix`. */
export function consoleHtml(name: string, prefix: string, note: string): string | null {
  const src = CONSOLE_PAGES[name];
  if (src === undefined) return null;
  const metas = `<meta name="fv-console-base" content="${esc(prefix)}">\n<meta name="fv-console-off" content="${OFF_PAGES.join(",")}">\n<meta name="fv-console-note" content="${esc(note)}">\n`;
  return src
    .replaceAll('href="/console', `href="${prefix}/console`)
    .replaceAll('src="/console', `src="${prefix}/console`)
    .replace(/<meta charset="utf-8">\n?/, (m) => `${m.endsWith("\n") ? m : `${m}\n`}${metas}`);
}
function offPage(prefix: string, what: string): Response {
  const body = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not available · fv-serve console</title><link rel="stylesheet" href="${esc(prefix)}/console/assets/console.css"></head>
<body><main class="page"><div class="page-head"><h1>Not available on a serverless endpoint</h1></div>
<div class="banner" data-off="${esc(what)}">The ${esc(what)} page needs a server the browser can reach (live sessions over WebRTC / WHIP, or fv-serve's own API keys). A serverless endpoint behind fv-control serves the generation pages and the Native API page; use a pod for live sessions.</div>
<p><a href="${esc(prefix)}/console">Models</a> · <a href="${esc(prefix)}/console/native">Native API</a></p></main></body></html>`;
  return new Response(body, { status: 404, headers: pageHeaders("text/html; charset=utf-8", true) });
}
function servePage(rest: string, prefix: string, row: SlsRow): Response | null {
  const p = rest.replace(/\/+$/, "") || "/console";
  if (p === "/console/assets" || p.startsWith("/console/assets/")) {
    const a = CONSOLE_ASSETS[p.slice("/console/assets/".length)];
    return a ? new Response(a.body, { headers: pageHeaders(a.type, false) }) : new Response("no such console asset", { status: 404 });
  }
  const page = (name: string) => new Response(consoleHtml(name, prefix, consoleNote(row)), { headers: pageHeaders("text/html; charset=utf-8", true) });
  if (p === "/console") return page("index");
  if (p === "/console/native") return page("native");
  if (["/console/admin", "/console/stream", "/console/live", "/console/avatar"].includes(p)) return offPage(prefix, p.split("/").pop()!);
  if (p.startsWith("/console/models/")) return /\/director$/.test(p) ? offPage(prefix, "director") : page("model");
  return null;
}

// ---------------------------------------------------------------- the routes the console calls
export type Route =
  | { kind: "cached"; path: string }
  | { kind: "status" }
  | { kind: "reactor-schema" }
  | { kind: "upload-initiate" }
  | { kind: "upload-put"; token: string }
  | { kind: "submit"; api: Api; path: string; app?: string }
  | { kind: "poll"; api: Api; id: string; view: "status" | "result" | "job"; app?: string }
  | { kind: "content"; id: string }
  | { kind: "cancel"; api: Api; id: string; app?: string }
  | { kind: "off"; what: string }
  | { kind: "fallback"; path: string }
  | { kind: "none" };

const ID = "([A-Za-z0-9_.-]{1,100})";
/** Which call of the console (or of its API snippets) a method and path are. `falApps`: fal app ids from the cached catalog. */
export function classify(method: string, path: string, search: URLSearchParams, falApps: string[] = []): Route {
  const m = method.toUpperCase();
  let x: RegExpExecArray | null;
  if (m === "GET" && CACHED(path)) return { kind: "cached", path };
  if (m === "GET" && path === "/fv/v1/status") return { kind: "status" };
  if (m === "GET" && path === "/schema") return { kind: "reactor-schema" };
  if (path.startsWith("/fv/v1/admin") || path.startsWith("/fv/v1/streams") || path.startsWith("/wma/") || path.startsWith("/sessions/") || path === "/start_session" || path === "/stop_session")
    return { kind: "off", what: path };
  if (m === "POST" && path === "/storage/upload/initiate") return { kind: "upload-initiate" };
  if (m === "PUT" && (x = /^\/storage\/upload\/put\/([A-Za-z0-9_.-]{20,2000})$/.exec(path))) return { kind: "upload-put", token: x[1]! };
  // Native, OpenAI videos, MiniMax.
  if (m === "POST" && path === "/fv/v1/jobs") return { kind: "submit", api: "native", path };
  if (m === "POST" && path === "/v1/videos") return { kind: "submit", api: "openai_videos", path };
  if (m === "POST" && path === "/v2/video_generation") return { kind: "submit", api: "minimax_v2", path };
  if ((x = new RegExp(`^/fv/v1/jobs/${ID}$`).exec(path))) return m === "DELETE" ? { kind: "cancel", api: "native", id: x[1]! } : m === "GET" ? { kind: "poll", api: "native", id: x[1]!, view: "job" } : { kind: "none" };
  if (m === "GET" && (x = new RegExp(`^/v1/videos/${ID}/content$`).exec(path))) return { kind: "content", id: x[1]! };
  if ((x = new RegExp(`^/v1/videos/${ID}$`).exec(path))) return m === "DELETE" ? { kind: "cancel", api: "openai_videos", id: x[1]! } : m === "GET" ? { kind: "poll", api: "openai_videos", id: x[1]!, view: "job" } : { kind: "none" };
  if (m === "GET" && path === "/v2/query/video_generation" && /^[A-Za-z0-9_.-]{1,100}$/.test(search.get("task_id") || "")) return { kind: "poll", api: "minimax_v2", id: search.get("task_id")!, view: "job" };
  if (m === "DELETE" && (x = new RegExp(`^/v2/video_generation/${ID}$`).exec(path))) return { kind: "cancel", api: "minimax_v2", id: x[1]! };
  // fal queue: /{app}/requests/{id}[/status|/cancel], POST /{app}/{sub…}.
  if ((x = new RegExp(`^/([A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~-]+)+)/requests/${ID}(/status|/cancel)?$`).exec(path))) {
    const app = x[1]!;
    const id = x[2]!;
    if (x[3] === "/cancel") return m === "PUT" ? { kind: "cancel", api: "fal", id, app } : { kind: "none" };
    if (m === "GET") return { kind: "poll", api: "fal", id, app, view: x[3] === "/status" ? "status" : "result" };
    return { kind: "none" };
  }
  if (m === "POST" && /^\/[A-Za-z0-9._~-]+(\/[A-Za-z0-9._~-]+){2,}$/.test(path) && !path.startsWith("/fv/") && !path.startsWith("/v1/") && !path.startsWith("/v2/")) {
    const app = falApps.find((a) => path.startsWith(`/${a}/`)) ?? path.split("/").slice(1, 3).join("/");
    return { kind: "submit", api: "fal", path, app };
  }
  if (m === "GET" && /^\/(v1|v2|fv\/v1)\//.test(path)) return { kind: "fallback", path };
  return { kind: "none" };
}

// ---------------------------------------------------------------- the queue job envelope
/** The `kind: http` job a submit becomes (crates/fastvideo-deploy/src/runpod/mod.rs HttpJob): it waits on the fv-serve job it creates
 * and, for native / OpenAI / MiniMax, DELETEs its cancel route when Runpod cancels it (a fal job's cancel is a PUT: cancel.ts sends it). */
export function wrapSubmit(path: string, contentType: string, raw: Uint8Array, timeoutS: number): Record<string, unknown> {
  const ct = (contentType || "application/json").split(";")[0]!.trim().toLowerCase();
  let body: unknown = undefined;
  let b64: string | undefined;
  if (ct === "application/json") {
    try {
      body = raw.byteLength ? JSON.parse(new TextDecoder().decode(raw)) : {};
    } catch {
      throw new HttpError(400, "the request body is not valid JSON");
    }
  } else if (raw.byteLength) {
    let s = "";
    for (let i = 0; i < raw.length; i += 0x8000) s += String.fromCharCode(...raw.subarray(i, i + 0x8000));
    b64 = btoa(s);
  }
  const job: Record<string, unknown> = { kind: "http", method: "POST", path, headers: { "content-type": contentType || "application/json" }, wait: true, timeout_s: timeoutS };
  if (b64 !== undefined) job.body_b64 = b64;
  else job.body = body;
  return withCancelPath(job);
}
/** The envelope as fv-control records it: long strings (data: URIs, base64) elided, still JSON (cancel.ts parses it). */
export function recordedInput(job: Record<string, unknown>): string {
  const s = JSON.stringify(job, (k, v) => (typeof v === "string" && v.length > 400 ? `${v.slice(0, 80)}…(${v.length} chars)` : v));
  return s.length <= 16000 ? s : JSON.stringify({ ...job, body: "(elided)", body_b64: undefined });
}

// ---------------------------------------------------------------- where a console job is
interface ConsoleJob {
  rec: any;
  api: Api;
  status: string; // Runpod's
  output: any; // the waiting job's progress ({state, poll_path}) or final output ({status, headers, body, submit?, poll_path?})
  error: string | null;
  input: any;
  fvId: string | null;
  store: { status: string; progress: number; job: any } | null;
}
/** The fv-serve job id in a waiting job's output: the submit reply (`submit`, or `body` of an immediate reply), else the progress poll path. */
export function fvIdOf(api: Api, out: any): string | null {
  const o = typeof out === "string" ? parseJson<any>(out, null) : out;
  if (!o || typeof o !== "object") return null;
  const idIn = (b: any) => {
    const v = api === "fal" ? b?.request_id : api === "minimax_v2" ? b?.task_id : b?.id;
    return typeof v === "string" || typeof v === "number" ? String(v) : null;
  };
  let id = o.submit ? idIn(o.submit) : typeof o.status === "number" && o.status < 300 ? idIn(o.body) : null;
  if (!id && typeof o.poll_path === "string") {
    const p = o.poll_path;
    const m = api === "fal" ? /\/requests\/([^/?]+)\/status/.exec(p) : api === "minimax_v2" ? /[?&]task_id=([^&]+)/.exec(p) : /\/([^/?]+)(?:\?.*)?$/.exec(p);
    id = m ? decodeURIComponent(m[1]!) : null;
  }
  return id;
}
const clipOut = (env: Env, v: unknown) => (v === undefined || v === null ? null : scrub(env, typeof v === "string" ? v : JSON.stringify(v)).slice(0, OUTPUT_MAX));

async function loadJob(env: ConsoleEnv, row: SlsRow, id: string, refresh = true): Promise<ConsoleJob | null> {
  let rec = await env.DB.prepare("SELECT * FROM serverless_jobs WHERE endpoint = ? AND job_id = ? AND route LIKE 'console:%' ORDER BY id DESC LIMIT 1").bind(row.id, id).first<any>();
  if (!rec) return null;
  if (refresh && !rec.finished_at && row.endpoint_id) {
    const j = await sls.status(env, row.endpoint_id, id).catch((e) => (e instanceof HttpError && (e.status === 404 || /not exist|not found/i.test(e.message)) ? { status: "GONE" } : null));
    if (j) {
      const status = String(j.status || rec.status);
      const done = TERMINAL.has(status) || status === "GONE";
      const errText = j.error ? scrub(env, typeof j.error === "string" ? j.error : JSON.stringify(j.error)).slice(0, 4000) : null;
      await env.DB.prepare("UPDATE serverless_jobs SET status = ?, delay_ms = COALESCE(?, delay_ms), exec_ms = COALESCE(?, exec_ms), worker_id = COALESCE(?, worker_id), output = COALESCE(?, output), error = COALESCE(?, error), finished_at = ?, wall_ms = ? WHERE id = ?")
        .bind(status, j.delayTime ?? null, j.executionTime ?? null, j.workerId ?? null, clipOut(env, j.output), errText, done ? now() : null, done ? now() - rec.submitted_at : null, rec.id)
        .run();
      rec = await env.DB.prepare("SELECT * FROM serverless_jobs WHERE id = ?").bind(rec.id).first<any>();
    }
  }
  const api = String(rec.route).slice("console:".length) as Api;
  const output = parseJson<any>(rec.output, null);
  const fvId = fvIdOf(api, output);
  let store: ConsoleJob["store"] = null;
  if (fvId && env.JOBS_DB) {
    const s = await env.JOBS_DB.prepare("SELECT status, progress, job FROM jobs WHERE protocol = ? AND external_id = ?").bind(api, fvId).first<any>().catch(() => null);
    if (s) store = { status: String(s.status), progress: Number(s.progress || 0), job: parseJson<any>(s.job, {}) };
  }
  return { rec, api, status: String(rec.status || "IN_QUEUE"), output, error: rec.error ?? null, input: parseJson<any>(rec.input, {}), fvId, store };
}

/** Where a console job is, whatever its API: from Runpod's state, the job store and the waiting job's output. */
export interface Phase {
  phase: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  logs: { message: string; level: string; timestamp: string }[];
  queue_position: number | null;
  /** The worker's own reply: the final status / result (done), or the refused submit (failed). */
  reply: { code: number; body: any } | null;
  message: string | null;
}
function runpodError(s: string | null): string {
  const e = parseJson<any>(s, null);
  if (e && typeof e === "object") return String(e.error_message || e.message || e.error_type || s);
  return s || "the queue job failed";
}
export function phaseOf(j: Pick<ConsoleJob, "status" | "output" | "error" | "store">): Phase {
  const o = j.output && typeof j.output === "object" ? j.output : null;
  const st = j.store;
  const logs = Array.isArray(st?.job?.logs) ? st!.job.logs.map((l: any) => ({ message: String(l.message ?? ""), level: String(l.level ?? "info"), timestamp: String(l.timestamp ?? "") })) : [];
  const base = { progress: st ? st.progress : 0, logs, queue_position: st?.job?.queue_position ?? null, reply: null, message: null };
  if (j.status === "COMPLETED" && o && typeof o.status === "number") {
    if (o.submit !== undefined || (o.status >= 200 && o.status < 300)) {
      const w = String(o.body?.status ?? o.body?.data?.status ?? "").toLowerCase();
      const failed = o.status >= 400 || ["failed", "fail", "error"].includes(w);
      const cancelled = w === "cancelled" || w === "canceled";
      return { ...base, progress: 1, phase: cancelled ? "cancelled" : failed ? "failed" : "done", reply: { code: o.status, body: o.body }, message: failed ? errorText(o.body) : null };
    }
    // The worker refused the submit (validation, 4xx): its reply is the answer.
    return { ...base, phase: "failed", reply: { code: o.status, body: o.body }, message: errorText(o.body) };
  }
  if (j.status === "COMPLETED") return { ...base, phase: "failed", message: "the queue job finished without an HTTP reply" };
  if (j.status === "CANCELLED") return { ...base, phase: "cancelled", message: "cancelled" };
  if (j.status === "FAILED" || j.status === "TIMED_OUT") return { ...base, phase: /Cancelled/.test(j.error || "") ? "cancelled" : "failed", message: j.status === "TIMED_OUT" ? "the queue job timed out (the endpoint's execution timeout)" : runpodError(j.error) };
  if (j.status === "GONE") return { ...base, phase: "failed", message: "Runpod no longer has this job (it keeps a finished job about 30 minutes)" };
  // Still running: the store says how far (a finished store row waits for the queue job's final output).
  if (st) return { ...base, phase: st.status === "queued" ? "queued" : "running", progress: Math.min(st.progress, 1) };
  // No store row (no JOBS_DB, or not written yet): Runpod's state and the waiting job's last status word.
  const word = String(o?.state ?? "").toLowerCase();
  const waiting = j.status !== "IN_PROGRESS" || !word || ["queued", "in_queue", "queueing", "preparing"].includes(word);
  return { ...base, phase: waiting ? "queued" : "running" };
}
function errorText(b: any): string {
  if (b && typeof b === "object") {
    if (b.error?.message) return String(b.error.message);
    if (typeof b.error === "string") return b.error;
    if (b.detail !== undefined) return typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail);
    if (b.base_resp?.status_msg) return String(b.base_resp.status_msg);
  }
  return typeof b === "string" && b ? b.slice(0, 300) : "failed";
}

// ---------------------------------------------------------------- per-API views
const iso = (t: number | null | undefined) => (t ? new Date(t).toISOString() : null);
/** The submit reply the page gets right away, in its API's shape, with the Runpod job id as the job's id. */
export function submitReply(api: Api, id: string, base: string, app: string | undefined, body: any, at: number): { code: number; body: any; headers?: Record<string, string> } {
  if (api === "fal") {
    const u = `${base}/${app}/requests/${id}`;
    return { code: 200, body: { request_id: id, response_url: u, status_url: `${u}/status`, cancel_url: `${u}/cancel`, queue_position: 0 }, headers: { "x-fal-request-id": id } };
  }
  if (api === "minimax_v2") return { code: 200, body: { task_id: id, base_resp: { status_code: 0, status_msg: "success" } } };
  if (api === "openai_videos") return { code: 200, body: { id, object: "video", model: body?.model ?? null, prompt: body?.prompt ?? null, status: "queued", progress: 0, created_at: Math.floor(at / 1000), seconds: body?.seconds ?? null, url: null, error: null } };
  return { code: 202, body: { id, object: "fv.job", status: "queued", progress: 0, queue_position: null, model: body?.model ?? null, task: body?.task ?? null, created_at: iso(at), started_at: null, completed_at: null, error: null, output: null, protocol: "native", metrics: {} } };
}
const FAL_LEVEL: Record<string, string> = { debug: "DEBUG", info: "INFO", warn: "WARN", error: "ERROR" };
/** A status / result / job view of a console job in its API's shape. In flight: synthesised from the phase; finished: the worker's own reply with the id swapped. */
export function viewOf(api: Api, view: "status" | "result" | "job", id: string, base: string, app: string | undefined, p: Phase, rec: { submitted_at: number; input?: any }): { code: number; body: any } {
  const body = rec.input?.body && typeof rec.input.body === "object" ? rec.input.body : {};
  if (api === "fal") {
    const u = `${base}/${app}/requests/${id}`;
    if (view === "result") {
      if (p.reply && p.phase !== "running" && p.phase !== "queued") return p.reply;
      if (p.phase === "failed") return { code: 500, body: { detail: p.message || "failed" } };
      if (p.phase === "cancelled") return { code: 400, body: { detail: "the request was cancelled" } };
      return { code: 400, body: { detail: "Request is still in progress" } };
    }
    const s: any = { request_id: id, response_url: u, status_url: `${u}/status`, cancel_url: `${u}/cancel` };
    const logs = p.logs.map((l) => ({ message: l.message, level: FAL_LEVEL[l.level.toLowerCase()] || l.level.toUpperCase(), source: "USER", timestamp: l.timestamp }));
    if (p.phase === "queued") return { code: 202, body: { ...s, status: "IN_QUEUE", queue_position: p.queue_position ?? 0 } };
    if (p.phase === "running") return { code: 202, body: { ...s, status: "IN_PROGRESS", logs } };
    return { code: 200, body: { ...s, status: "COMPLETED", logs, metrics: {}, ...(p.phase === "failed" ? { error: p.message } : p.phase === "cancelled" ? { error_type: "client_cancelled" } : {}) } };
  }
  const swap = (b: any, key: string) => (b && typeof b === "object" && !Array.isArray(b) ? { ...b, [key]: id } : b);
  if (api === "minimax_v2") {
    if (p.phase === "done" && p.reply) return { code: p.reply.code, body: swap(p.reply.body, "task_id") };
    const word = { queued: "Queueing", running: "Processing", done: "Success", failed: "Fail", cancelled: "Fail" }[p.phase];
    if (p.phase === "failed" && p.reply && !p.reply.body?.task_id) return p.reply;
    return { code: 200, body: { task_id: id, status: word, file_id: "", base_resp: p.phase === "failed" || p.phase === "cancelled" ? { status_code: 1027, status_msg: p.message || "failed" } : { status_code: 0, status_msg: "success" } } };
  }
  if (p.reply && (p.phase === "done" || p.phase === "cancelled" || (p.phase === "failed" && p.reply.code < 300))) return { code: p.reply.code, body: swap(p.reply.body, "id") };
  if (api === "openai_videos") {
    const status = { queued: "queued", running: "in_progress", done: "completed", failed: "failed", cancelled: "failed" }[p.phase];
    return {
      code: 200,
      body: { id, object: "video", model: body.model ?? null, prompt: body.prompt ?? null, status, progress: Math.round(p.progress * 100), created_at: Math.floor(rec.submitted_at / 1000), seconds: body.seconds ?? null, url: null, error: p.phase === "failed" || p.phase === "cancelled" ? { code: "generation_failed", message: p.message || "failed" } : null },
    };
  }
  const status = { queued: "queued", running: "running", done: "succeeded", failed: "failed", cancelled: "cancelled" }[p.phase];
  return {
    code: 200,
    body: {
      id,
      object: "fv.job",
      status,
      progress: Math.round(p.progress * 100) / 100,
      queue_position: p.queue_position,
      model: body.model ?? null,
      task: body.task ?? null,
      created_at: iso(rec.submitted_at),
      error: p.phase === "failed" || p.phase === "cancelled" ? { kind: p.phase === "cancelled" ? "cancelled" : "failed", message: p.message || p.phase } : null,
      output: null,
      protocol: "native",
      notes: p.phase === "queued" ? ["waiting for a serverless worker (fv-control): a cold start can take minutes"] : [],
      metrics: {},
    },
  };
}

// ---------------------------------------------------------------- cached GETs
interface CacheEntry {
  at?: number;
  code?: number;
  body?: unknown;
  pending?: { job: string; at: number };
  error?: string;
}
const imageTag = (row: SlsRow) => (row.image || "").slice(-12);
const cacheKey = (row: SlsRow, path: string) => `slsc:${row.id}:${imageTag(row)}:${path}`;
async function readCache(env: Env, key: string): Promise<CacheEntry> {
  const r = await env.DB.prepare("SELECT value FROM settings WHERE key = ?").bind(key).first<{ value: string }>();
  return parseJson<CacheEntry>(r?.value, {});
}
async function writeCache(env: Env, key: string, e: CacheEntry): Promise<void> {
  await env.DB.prepare("INSERT INTO settings (key, value, updated_at, updated_by) VALUES (?, ?, ?, 'serverless-console') ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at")
    .bind(key, JSON.stringify(e), now())
    .run();
}
/** Drops an endpoint's cached replies (the endpoint page's "Refresh console cache"). */
export async function clearConsoleCache(env: Env, row: SlsRow): Promise<number> {
  const r = await env.DB.prepare("DELETE FROM settings WHERE key LIKE ?").bind(`slsc:${row.id}:%`).run();
  return Number(r.meta?.changes ?? 0);
}
/** A queue job's `kind: http` GET reply, once done: {code, body}; null while it runs; throws when it failed. */
async function jobReply(env: SlsEnv, row: SlsRow, job: string): Promise<{ code: number; body: unknown } | null> {
  const j = await sls.status(env, row.endpoint_id!, job);
  const st = String(j?.status || "");
  if (!TERMINAL.has(st)) return null;
  if (st === "COMPLETED" && j.output && typeof j.output.status === "number") return { code: j.output.status, body: j.output.body };
  throw new HttpError(502, `the cache job ${job} ended ${st}: ${runpodError(typeof j?.error === "string" ? j.error : j?.error ? JSON.stringify(j.error) : null)}`);
}
async function startCacheJob(env: SlsEnv, row: SlsRow, path: string, who: string): Promise<string> {
  if (row.status !== "active") throw new HttpError(409, `${row.name} is ${row.status}${row.status === "scaled-down" ? ": scale it up to fill the console's cache" : ""}`);
  await assertFloor(env, "console");
  const input = { kind: "http", method: "GET", path };
  const r = await sls.run(env, row.endpoint_id!, { input, policy: { executionTimeout: 120_000 } });
  const id = String(r?.id || "");
  if (!id) throw new HttpError(502, "runpod: /run returned no job id");
  await env.DB.prepare("INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, input, actor) VALUES (?, ?, 'console-cache', ?, 0, ?, ?, ?)")
    .bind(row.id, id, String(r?.status || "IN_QUEUE"), now(), JSON.stringify(input), who)
    .run();
  return id;
}
async function lbGet(env: SlsEnv, row: SlsRow, path: string): Promise<{ code: number; body: unknown }> {
  const r = await sls.lb(env, row.endpoint_id!, "GET", path);
  return { code: r.status, body: r.body };
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
/** A cached GET: fresh → served; stale → served, and refreshed in the background while a worker is up; missing → one job, waited for a while. */
export async function cachedGet(env: ConsoleEnv, row: SlsRow, path: string, o: { who: string; force?: boolean; waitUntil?: (p: Promise<unknown>) => void; waitMs?: number }): Promise<{ code: number; body: unknown; age_s: number | null; source: string }> {
  const key = cacheKey(row, path);
  let e = await readCache(env, key);
  const settle = async () => {
    if (!e.pending) return;
    try {
      const r = await jobReply(env, row, e.pending.job);
      if (r) {
        e = r.code < 500 ? { at: now(), code: r.code, body: r.body } : { ...e, pending: undefined, error: `HTTP ${r.code}` };
        await writeCache(env, key, e);
      }
    } catch (x) {
      e = { ...e, pending: undefined, error: (x as Error).message };
      await writeCache(env, key, e);
    }
  };
  await settle();
  const age = e.at ? now() - e.at : null;
  if (e.at && e.code !== undefined && !o.force) {
    if (age! > CACHE_TTL_MS && !e.pending && row.mode === "queue" && row.status === "active") {
      // Stale: refresh only while a worker is up (never wake one for it).
      const h = await sls.health(env, row.endpoint_id!).catch(() => null);
      if (h && readyWorkers(h) > 0) {
        const t = startCacheJob(env, row, path, o.who).then((job) => writeCache(env, key, { ...e, pending: { job, at: now() } })).catch(() => {});
        if (o.waitUntil) o.waitUntil(t);
        else await t;
      }
    }
    return { code: e.code, body: e.body, age_s: Math.round(age! / 1000), source: "cache" };
  }
  if (row.mode === "lb") {
    const r = await lbGet(env, row, path);
    if (r.code < 500) await writeCache(env, key, { at: now(), code: r.code, body: r.body });
    return { ...r, age_s: 0, source: "lb" };
  }
  if (!e.pending) {
    const job = await startCacheJob(env, row, path, o.who);
    e = { ...e, pending: { job, at: now() }, error: undefined };
    await writeCache(env, key, e);
  }
  const deadline = now() + (o.waitMs ?? n(env.CONSOLE_CACHE_WAIT_MS, 25_000));
  const every = n(env.CONSOLE_POLL_MS, 1000);
  while (e.pending && now() < deadline) {
    await sleep(every);
    await settle();
  }
  if (e.at && e.code !== undefined && !e.pending) return { code: e.code, body: e.body, age_s: 0, source: "job" };
  if (e.error && !e.pending) throw new HttpError(502, `GET ${path} on the endpoint failed: ${e.error}`);
  throw new HttpError(503, `a worker is starting (cold start): GET ${path} waits as Runpod job ${e.pending!.job}. Reload in a minute.`, { cold_start: true, job: e.pending!.job });
}
/** Starts the cache jobs of a fresh console page (capabilities and the fal catalog) without waiting, so the page's own calls find them running. */
export async function prewarm(env: ConsoleEnv, row: SlsRow, who: string): Promise<void> {
  if (row.mode !== "queue" || row.status !== "active") return;
  for (const path of ["/fv/v1/capabilities", "/fal/schema"]) {
    const key = cacheKey(row, path);
    const e = await readCache(env, key);
    if (e.at || e.pending) continue;
    const job = await startCacheJob(env, row, path, who);
    await writeCache(env, key, { pending: { job, at: now() } });
  }
}
/** The cached capabilities as the console should see them: no key needed (fv-control already authenticated the browser). */
export function capsForConsole(body: any): any {
  if (!body || typeof body !== "object") return body;
  return { ...body, auth: { ...(body.auth || {}), mode: "none" }, served_by: "fv-control serverless console" };
}

// ---------------------------------------------------------------- status
/** fv-serve's serverless pool state from Runpod's worker counts (crates/fastvideo-serve/src/status.rs serverless_state). */
export function serverlessState(counts: Record<string, number>): string {
  const c = (k: string) => Number(counts[k] || 0);
  if (c("idle") + c("ready") > 0) return "ready";
  if (c("running") > 0) return "busy";
  if (c("initializing") > 0) return "loading";
  if (c("unhealthy") > 0) return "unhealthy";
  return "scaled_to_zero";
}
/** The names map of fv-serve's status: every served name, alias and tier alias → model id. */
export function namesOf(caps: any): Record<string, string> {
  const names: Record<string, string> = {};
  for (const m of Array.isArray(caps?.models) ? caps.models : []) {
    const id = m?.caps?.id;
    if (typeof id !== "string") continue;
    names[id] = id;
    for (const s of m.caps.served_names || []) if (typeof s === "string") names[s] = id;
  }
  for (const [a, id] of Object.entries(caps?.aliases || {})) if (typeof id === "string") names[a] = names[id] ?? id;
  for (const t of Array.isArray(caps?.tiers) ? caps.tiers : []) if (t && typeof t.alias === "string" && typeof t.model === "string") names[t.alias] = names[t.model] ?? t.model;
  return names;
}
/** `GET /fv/v1/status` (status.rs shape) for one serverless pool: Runpod's health for a queue endpoint, fv-control's worker count for a load balancer. */
export function statusBody(row: SlsRow, health: any | null, caps: any | null): any {
  const counts: Record<string, number> = {};
  for (const [k, v] of Object.entries(health?.workers || {})) if (typeof v === "number") counts[k] = v;
  let state: string;
  if (row.deleted_at || !["active"].includes(row.status)) state = "down";
  else if (row.mode === "lb") state = (row.workers ?? 0) > 0 ? "ready" : "scaled_to_zero";
  else state = health ? serverlessState(counts) : "down";
  const models = (Array.isArray(caps?.models) ? caps.models : []).map((m: any) => m?.caps?.id).filter((x: unknown): x is string => typeof x === "string");
  const pool = {
    id: row.name,
    kind: "runpod-serverless",
    state,
    available: !["down", "unhealthy", "failed", "draining"].includes(state),
    models,
    queued: Number(health?.jobs?.inQueue ?? 0),
    running: Number(health?.jobs?.inProgress ?? 0),
    last_seen_s: health ? 0 : null,
    workers: [],
    ...(row.mode === "queue" ? { worker_counts: counts } : { worker_counts: { running: row.workers ?? 0 } }),
    mixed_versions: false,
  };
  return {
    object: "fv.status",
    gateway: false,
    state,
    pools: [pool],
    models: Object.fromEntries(models.map((m: string) => [m, { state, pools: [row.name] }])),
    names: namesOf(caps),
    served_by: "fv-control serverless console",
  };
}

// ---------------------------------------------------------------- uploads (R2, a signed public read URL)
async function sign(env: Env, claims: Record<string, unknown>): Promise<string> {
  const p = b64url(new TextEncoder().encode(JSON.stringify(claims)));
  return `${p}.${await hmacB64url(env.SESSION_SECRET, `slsup|${p}`)}`;
}
export async function verifyUpload(env: Env, token: string, mode: "put" | "get"): Promise<{ k: string; ct: string; e: number } | null> {
  const [p, s] = token.split(".");
  if (!p || !s || !safeEqual(s, await hmacB64url(env.SESSION_SECRET, `slsup|${p}`))) return null;
  let c: any;
  try {
    c = JSON.parse(new TextDecoder().decode(unb64(p)));
  } catch {
    return null;
  }
  if (c?.m !== mode || typeof c.k !== "string" || !c.k.startsWith(UPLOAD_PREFIX) || !(c.e > now())) return null;
  return { k: c.k, ct: String(c.ct || "application/octet-stream"), e: c.e };
}
async function uploadInitiate(env: Env, row: SlsRow, req: Request, base: string, publicBase: string): Promise<Response> {
  let b: any = {};
  try {
    b = await req.json();
  } catch {}
  const ct = typeof b.content_type === "string" && /^[\w.+-]+\/[\w.+-]+$/.test(b.content_type) ? b.content_type : "application/octet-stream";
  const name = String(b.file_name || "upload").replace(/[^A-Za-z0-9._-]+/g, "_").slice(-100) || "upload";
  const key = `${UPLOAD_PREFIX}${row.id}/${randomToken("", 12)}/${name}`;
  const put = await sign(env, { m: "put", k: key, ct, e: now() + 15 * 60_000 });
  const get = await sign(env, { m: "get", k: key, ct, e: now() + 24 * 3600_000 });
  return json(200, { upload_url: `${base}/storage/upload/put/${put}`, file_url: `${publicBase}/serverless-uploads/${get}/${encodeURIComponent(name)}` });
}
async function uploadPut(env: Env, req: Request, token: string): Promise<Response> {
  const t = await verifyUpload(env, token, "put");
  if (!t) return err(403, "forbidden", "the upload URL is invalid or expired (15 min)");
  const len = Number(req.headers.get("content-length") || "NaN");
  if (!Number.isFinite(len)) return err(411, "invalid_request", "the upload needs a Content-Length");
  if (len > UPLOAD_MAX) return err(413, "payload_too_large", `uploads through the serverless console are at most ${UPLOAD_MAX >> 20} MB`);
  await env.LOGS.put(t.k, req.body, { httpMetadata: { contentType: req.headers.get("content-type") || t.ct } });
  return new Response(null, { status: 200 });
}
/** GET /serverless-uploads/<token>/<name>: public (the worker fetches it), the signed token is the capability. */
export async function uploadGet(env: Env, token: string): Promise<Response> {
  const t = await verifyUpload(env, token, "get");
  if (!t) return new Response("invalid or expired", { status: 403 });
  const o = await env.LOGS.get(t.k);
  if (!o) return new Response("not found", { status: 404 });
  return new Response(o.body, { headers: { "content-type": o.httpMetadata?.contentType || t.ct, "content-length": String(o.size), "cache-control": "private, max-age=3600", "x-content-type-options": "nosniff" } });
}
/** The tick: uploads older than a day go (their read URLs expire then). */
export async function sweepUploads(env: Env, maxAgeMs = 26 * 3600_000): Promise<number> {
  const l = await env.LOGS.list({ prefix: UPLOAD_PREFIX, limit: 200 });
  const old = l.objects.filter((o) => now() - o.uploaded.getTime() > maxAgeMs).map((o) => o.key);
  if (old.length) await env.LOGS.delete(old);
  return old.length;
}

// ---------------------------------------------------------------- the handler
export interface ConsoleCtx {
  /** The endpoint key as the URL names it (/serverless/<ep>). */
  ep: string;
  who: Actor;
  /** read: GET only (a read-scope API token). */
  readOnly?: boolean;
  waitUntil?: (p: Promise<unknown>) => void;
}
async function falApps(env: Env, row: SlsRow): Promise<string[]> {
  const e = await readCache(env, cacheKey(row, "/fal/schema"));
  const apps = (e.body as any)?.apps;
  return Array.isArray(apps) ? apps.map((a: any) => String(a?.id || "")).filter(Boolean).sort((a, b) => b.length - a.length) : [];
}

/** Serves /serverless/<ep>/<rest> for an authenticated browser (index.ts mounts it behind requireConsoleAuth). */
export async function consoleRequest(env: ConsoleEnv, req: Request, row: SlsRow, c: ConsoleCtx): Promise<Response> {
  const url = new URL(req.url);
  const prefix = `/serverless/${encodeURIComponent(c.ep)}`;
  const rest = url.pathname.slice(url.pathname.indexOf(prefix) + prefix.length) || "/";
  const base = `${url.origin}${prefix}`;
  const publicBase = (env.PUBLIC_URL || url.origin).replace(/\/+$/, "");
  const method = req.method.toUpperCase();
  try {
    if (rest === "/" || rest === "") return Response.redirect(`${base}/console`, 302);
    if (rest === "/console" || rest.startsWith("/console/")) {
      if (method !== "GET" && method !== "HEAD") return err(405, "method_not_allowed", "pages are GET only");
      const r = servePage(rest, prefix, row);
      if (r && (rest === "/console" || rest === "/console/") && c.waitUntil) c.waitUntil(prewarm(env, row, c.who.actor).catch(() => {}));
      return r ?? new Response("not found", { status: 404 });
    }
    if (row.deleted_at || !row.endpoint_id) return err(410, "gone", `${row.name} is ${row.status}: it has no Runpod endpoint`);
    if (c.readOnly && method !== "GET" && method !== "HEAD") return err(403, "forbidden", "read-only token");
    const route = classify(method, rest, url.searchParams, method === "POST" ? await falApps(env, row) : []);
    switch (route.kind) {
      case "cached": {
        const r = await cachedGet(env, row, route.path, { who: c.who.actor, waitUntil: c.waitUntil });
        const body = route.path === "/fv/v1/capabilities" && r.code < 300 ? capsForConsole(r.body) : r.body;
        return json(r.code, body, { "x-fv-console-cache": `${r.source}; age=${r.age_s ?? 0}` });
      }
      case "status": {
        const health = row.mode === "queue" && row.status === "active" ? await sls.health(env, row.endpoint_id).catch(() => null) : null;
        const caps = (await readCache(env, cacheKey(row, "/fv/v1/capabilities"))).body ?? null;
        return json(200, statusBody(row, health, caps));
      }
      case "reactor-schema":
        return err(404, "not_found", "the Reactor runtime is not available through the serverless console");
      case "off":
        return err(404, "not_found", `${route.what}: not available on a serverless endpoint (live sessions and API keys need a pod)`);
      case "upload-initiate":
        return await uploadInitiate(env, row, req, base, publicBase);
      case "upload-put":
        return await uploadPut(env, req, route.token);
    }
    if (row.mode === "lb") return await lbForward(env, row, req, rest + url.search, base);
    switch (route.kind) {
      case "submit":
        return await submit(env, row, req, c, route.api, route.path, route.app, base);
      case "poll": {
        const j = await loadJob(env, row, route.id);
        if (!j || j.api !== route.api) return notFound(route.api, route.id);
        const v = viewOf(route.api, route.view, route.id, base, route.app ?? appOf(j.input), phaseOf(j), { submitted_at: j.rec.submitted_at, input: j.input });
        return json(v.code, v.body);
      }
      case "content":
        return await content(env, row, route.id);
      case "cancel":
        return await cancel(env, row, c, route.api, route.id, route.app);
      case "fallback":
        return await fallback(env, row, c, rest + url.search);
    }
    return err(404, "not_found", `${method} ${rest}: not served by the serverless console`);
  } catch (e) {
    const status = e instanceof HttpError ? e.status : 500;
    const extra: Record<string, string> = e instanceof HttpError && e.extra?.cold_start ? { "retry-after": "10" } : {};
    return err(status, status === 503 ? "cold_start" : status === 402 ? "balance_floor" : "error", scrub(env, (e as Error).message), extra);
  }
}
const appOf = (input: any) => String(input?.path || "").split("/").slice(1, 3).join("/");
function notFound(api: Api, id: string): Response {
  if (api === "fal") return json(404, { detail: `Request ${id} not found` });
  if (api === "minimax_v2") return json(200, { task_id: id, status: "Fail", base_resp: { status_code: 2013, status_msg: "task not found" } });
  return err(404, "not_found", `job \`${id}\` was not found`);
}

async function submit(env: ConsoleEnv, row: SlsRow, req: Request, c: ConsoleCtx, api: Api, path: string, app: string | undefined, base: string): Promise<Response> {
  if (row.status !== "active") return err(409, "unavailable", `${row.name} is ${row.status}${row.status === "scaled-down" ? " (scale it up first)" : ""}`);
  const len = Number(req.headers.get("content-length") || 0);
  if (len > BODY_MAX) return err(413, "payload_too_large", `request bodies through the serverless console are at most ${BODY_MAX >> 20} MB (a Runpod job input is at most 10 MB): upload files and pass their URL`);
  const raw = new Uint8Array(await req.arrayBuffer());
  if (raw.byteLength > BODY_MAX) return err(413, "payload_too_large", `at most ${BODY_MAX >> 20} MB`);
  await assertFloor(env, "console submit");
  const spec = specOf(row);
  const job = wrapSubmit(path, req.headers.get("content-type") || "application/json", raw, spec.execution_timeout_s);
  const health = await sls.health(env, row.endpoint_id!).catch(() => null);
  const cold = health ? readyWorkers(health) === 0 : false;
  const t0 = now();
  const r = await sls.run(env, row.endpoint_id!, { input: job, policy: { executionTimeout: spec.execution_timeout_s * 1000 } });
  const id = String(r?.id || "");
  if (!id) throw new HttpError(502, "runpod: /run returned no job id");
  await env.DB.prepare("INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, input, actor) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
    .bind(row.id, id, `console:${api}`, String(r?.status || "IN_QUEUE"), cold ? 1 : 0, t0, recordedInput(job), c.who.actor)
    .run();
  await audit(env, { actor: c.who.actor, ip: c.who.ip, action: "serverless.console", target: row.name, after: { job: id, api, path, cold } });
  // A warm worker refuses an invalid request within a second or two: give the page fv-serve's own answer then.
  const until = now() + n(env.CONSOLE_SUBMIT_WAIT_MS, cold ? 0 : 3000);
  const every = n(env.CONSOLE_POLL_MS, 500);
  while (now() < until) {
    await sleep(every);
    const j = await loadJob(env, row, id);
    if (!j) break;
    if (j.fvId) break;
    const p = phaseOf(j);
    if (p.phase === "failed" && p.reply && p.reply.code >= 400) return json(p.reply.code, p.reply.body);
    if (p.phase !== "queued" && p.phase !== "running") break;
  }
  const s = submitReply(api, id, base, app, (job as any).body, t0);
  return json(s.code, s.body, s.headers || {});
}

async function content(env: ConsoleEnv, row: SlsRow, id: string): Promise<Response> {
  const j = await loadJob(env, row, id);
  if (!j || j.api !== "openai_videos") return notFound("openai_videos", id);
  const p = phaseOf(j);
  if (p.phase !== "done") return err(409, "not_ready", `video ${id} is ${p.phase}`);
  const u = p.reply?.body?.url;
  if (typeof u !== "string" || !/^https?:\/\//.test(u)) return err(502, "no_url", "the finished video names no URL (the endpoint's workers need R2 artifacts: FV_R2_*)");
  // Fetched here, not redirected: the page fetches the content with fetch(), and a presigned R2 URL sends no CORS headers.
  const r = await fetchWithTimeout(u, { timeoutMs: 120_000 });
  if (!r.ok) return err(502, "upstream", `fetching the video: HTTP ${r.status}`);
  return new Response(r.body, { headers: { "content-type": r.headers.get("content-type") || "video/mp4", ...(r.headers.get("content-length") ? { "content-length": r.headers.get("content-length")! } : {}), "cache-control": "no-store" } });
}

async function cancel(env: ConsoleEnv, row: SlsRow, c: ConsoleCtx, api: Api, id: string, app?: string): Promise<Response> {
  const j = await loadJob(env, row, id);
  if (!j || j.api !== api) return notFound(api, id);
  const p = phaseOf(j);
  if (p.phase !== "queued" && p.phase !== "running") {
    if (api === "fal") return json(400, { status: "ALREADY_COMPLETED" });
    return err(409, "already_completed", `job ${id} is ${p.phase}: nothing to cancel (finished jobs are not deleted through the serverless console)`);
  }
  // fal: the cancel route is a PUT under its app (no cancel_path); cancel.ts sends it as a queue job once the waiting job stopped.
  const fal = api === "fal" && j.fvId && TERMINAL.has(j.status) === false && j.status !== "IN_QUEUE" ? { fv_job: j.fvId, fv_api: "fal" as const, fv_model: app || appOf(j.input) } : {};
  const r = await cancelSlsJob(env, c.who, row, { job: id, ...fal });
  if (api === "fal") return json(202, { status: "CANCELLATION_REQUESTED", note: r.note });
  if (api === "minimax_v2") return json(200, { task_id: id, base_resp: { status_code: 0, status_msg: "success" }, note: r.note });
  if (api === "openai_videos") return json(200, { id, object: "video", status: r.status === "CANCELLED" ? "failed" : "in_progress", note: r.note });
  return json(200, { id, object: "fv.job", status: r.status === "CANCELLED" ? "cancelled" : "running", cancel_requested: true, note: r.note });
}

/** Any other read-only API GET: one queue job, waited for (the reply is fv-serve's own); it needs a worker. */
async function fallback(env: ConsoleEnv, row: SlsRow, c: ConsoleCtx, path: string): Promise<Response> {
  if (row.status !== "active") return err(409, "unavailable", `${row.name} is ${row.status}`);
  await assertFloor(env, "console");
  const input = { kind: "http", method: "GET", path };
  const r = await sls.run(env, row.endpoint_id!, { input, policy: { executionTimeout: 120_000 } });
  const id = String(r?.id || "");
  await env.DB.prepare("INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, input, actor) VALUES (?, ?, 'console-get', ?, 0, ?, ?, ?)").bind(row.id, id, String(r?.status || "IN_QUEUE"), now(), JSON.stringify(input), c.who.actor).run();
  const deadline = now() + n(env.CONSOLE_CACHE_WAIT_MS, 25_000);
  while (now() < deadline) {
    await sleep(n(env.CONSOLE_POLL_MS, 1000));
    const got = await jobReply(env, row, id);
    if (got) return json(got.code, got.body, { "x-fv-console-job": id });
  }
  return err(504, "timeout", `GET ${path} waits as Runpod job ${id} (no worker answered in time); try again`);
}

/** Tracing headers a load-balancer endpoint's requests pass on, and its answers' clock samples (docs/serve/tracing.md). */
export const TRACE_REQ = ["traceparent", "x-fv-trace"] as const;
export const TRACE_RESP = ["traceparent", "x-fv-trace-t", "server-timing"] as const;

/** A load-balancer endpoint: the request goes on to https://<id>.api.runpod.ai with the Runpod key; its own URLs in JSON replies point back here. */
async function lbForward(env: SlsEnv, row: SlsRow, req: Request, pathAndQuery: string, base: string): Promise<Response> {
  const lb = slsBases(env).lb(row.endpoint_id!);
  const method = req.method.toUpperCase();
  const headers: Record<string, string> = { authorization: `Bearer ${env.RUNPOD_API_KEY}`, accept: req.headers.get("accept") || "*/*" };
  const ct = req.headers.get("content-type");
  if (ct) headers["content-type"] = ct;
  // Request tracing (docs/serve/tracing.md): the opt-in and the trace id go on.
  for (const k of TRACE_REQ) {
    const v = req.headers.get(k);
    if (v) headers[k] = v;
  }
  const body = method === "GET" || method === "HEAD" ? undefined : await req.arrayBuffer();
  if (body && body.byteLength > BODY_MAX) return err(413, "payload_too_large", `at most ${BODY_MAX >> 20} MB`);
  const r = await fetchWithTimeout(`${lb}${pathAndQuery}`, { method, headers, body, redirect: "manual", timeoutMs: 150_000 });
  const out = new Headers();
  for (const k of ["content-type", "content-length", "retry-after", "x-fal-request-id", "x-fv-tier", "x-fv-quality", "x-fv-recipe", "x-fv-model", ...TRACE_RESP]) {
    const v = r.headers.get(k);
    if (v) out.set(k, v);
  }
  const loc = r.headers.get("location");
  if (loc) out.set("location", loc.split(lb).join(base));
  if (/json/.test(r.headers.get("content-type") || "")) {
    const text = scrub(env, await r.text()).split(lb).join(base);
    out.delete("content-length");
    return new Response(text, { status: r.status, headers: out });
  }
  return new Response(r.body, { status: r.status, headers: out });
}
