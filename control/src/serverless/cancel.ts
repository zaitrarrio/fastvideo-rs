// Cancelling serverless jobs and purging an endpoint's queue
// (docs/control/serverless.md "Cancel and purge").
//
// A queued job: Runpod's POST /cancel/<job> drops it. A job a worker took:
// the same call makes Runpod list it on the worker's job-stop long poll,
// and the worker cancels it (crates/fastvideo-deploy/src/runpod/worker.rs);
// a `kind: http` job that waits on the fv-serve job it created DELETEs its
// `cancel_path` then (fv-control's invokes set it, ops.ts withCancelPath).
// The fv-serve job outlives its queue job when the http job did not wait
// (the queue job is COMPLETED with the submit reply) or set no
// cancel_path: fv-control then sends the owning API's cancel route as one
// more queue job (`kind: http`, `/run`). fv-serve jobs live in a worker's
// process, so that job must land on the same worker: certain with one
// worker up, not with several (fv-control says so), and pointless with
// none (the job ended with its worker).
import type { Env } from "../env";
import { cancelRoute, fvJobOf, type FvApi } from "../jobapi";
import { audit, HttpError, now, parseJson, scrub } from "../util";
import { clip, pollJob, readyWorkers, TERMINAL, type Actor, type SlsRow } from "./ops";
import { sls, type SlsEnv } from "./runpod-sls";

/** What POST /api/serverless/:id/jobs/:job/cancel takes (schema serverless-cancel). */
export interface SlsCancelIn {
  job: string;
  fv_job?: string;
  fv_api?: FvApi;
  stop_fv_job?: boolean;
}
export interface FvCancel {
  sent: boolean;
  api?: string;
  id?: string;
  route?: string;
  /** The Runpod id of the queue job that carries the cancel, and fv-control's row for it. */
  runpod_job?: string | null;
  job?: number | null;
  reason: string;
}

function live(row: SlsRow): string {
  if (!row.endpoint_id || row.deleted_at || ["deleted", "gone", "deleting"].includes(row.status)) throw new HttpError(409, `${row.name} is ${row.status}: it has no queue`);
  if (row.mode !== "queue") throw new HttpError(409, `${row.name} is a load-balancer endpoint: requests are not queued jobs (nothing to cancel or purge)`);
  return row.endpoint_id;
}
/** Runpod's status of a job, or null when it has none with that id (Runpod keeps a finished job's result about 30 minutes). */
async function statusOrNull(env: SlsEnv, eid: string, job: string): Promise<any | null> {
  try {
    return await sls.status(env, eid, job);
  } catch (e) {
    if (e instanceof HttpError && (e.status === 404 || /not exist|not found/i.test(e.message))) return null;
    throw e;
  }
}

export async function cancelSlsJob(env: SlsEnv, who: Actor, row: SlsRow, x: SlsCancelIn) {
  const eid = live(row);
  // fv-control's own invoke (its row number) or any Runpod job id of the endpoint.
  let rec: any = null;
  if (/^\d{1,9}$/.test(x.job)) {
    rec = await env.DB.prepare("SELECT * FROM serverless_jobs WHERE id = ? AND endpoint = ?").bind(Number(x.job), row.id).first<any>();
    if (!rec) throw new HttpError(404, `no test invoke #${x.job} on ${row.name}`);
    if (!rec.job_id) throw new HttpError(409, `invoke #${x.job} was a load-balancer request: there is no queue job to cancel`);
  } else {
    rec = await env.DB.prepare("SELECT * FROM serverless_jobs WHERE endpoint = ? AND job_id = ? ORDER BY id DESC LIMIT 1").bind(row.id, x.job).first<any>();
  }
  const job = rec?.job_id ?? x.job;
  const before = await statusOrNull(env, eid, job);
  if (!before && !rec) throw new HttpError(404, `Runpod has no job ${job} on ${row.name} (${eid}); a finished job is kept about 30 minutes`);
  const beforeStatus = String(before?.status || rec?.status || "?");
  let cancelled: any = null;
  let cancelError: string | null = null;
  if (before && !TERMINAL.has(beforeStatus)) {
    try {
      cancelled = await sls.cancel(env, eid, job);
    } catch (e) {
      cancelError = scrub(env, (e as Error).message);
    }
  }
  const after = before ? await statusOrNull(env, eid, job) : null;
  const status = String(after?.status || cancelled?.status || beforeStatus);

  // The fv-serve side.
  const input = rec ? parseJson<any>(rec.input, null) : null;
  const found = fvJobOf(input, before?.output ?? (rec ? parseJson<any>(rec.output, rec.output) : null));
  const fv = x.fv_job ? { id: x.fv_job, api: (x.fv_api || "native") as FvApi, done: false } : found;
  const fvCancel = await stopFvJob(env, who, row, eid, { fv, beforeStatus, input, want: x.stop_fv_job !== false, explicit: !!x.fv_job });

  // Record the outcome: fv-control's row, or a row for a job submitted elsewhere (so its status shows and polls).
  let rowId: number | null = rec?.id ?? null;
  if (rec) {
    await env.DB.prepare("UPDATE serverless_jobs SET status = ?, finished_at = COALESCE(finished_at, ?), error = COALESCE(error, ?) WHERE id = ?")
      .bind(status, TERMINAL.has(status) ? now() : null, status === "CANCELLED" ? `cancelled by ${who.actor}` : null, rec.id)
      .run();
  } else {
    const ins = await env.DB.prepare("INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, finished_at, delay_ms, exec_ms, worker_id, input, output, error, actor) VALUES (?, ?, 'external', ?, 0, ?, ?, ?, ?, ?, NULL, ?, ?, ?) RETURNING id")
      .bind(row.id, job, status, now(), TERMINAL.has(status) ? now() : null, after?.delayTime ?? before?.delayTime ?? null, after?.executionTime ?? before?.executionTime ?? null, after?.workerId ?? before?.workerId ?? null, clip(env, after?.output ?? before?.output), status === "CANCELLED" ? `cancelled by ${who.actor}` : null, who.actor)
      .first<{ id: number }>();
    rowId = ins?.id ?? null;
  }
  const note = cancelError
    ? `Runpod refused the cancel: ${cancelError}`
    : TERMINAL.has(beforeStatus)
      ? `the job had already finished (${beforeStatus}): nothing to cancel on Runpod`
      : beforeStatus === "IN_PROGRESS"
        ? "the worker running it stops it at its next job-stop poll (seconds); an fv-serve job stops at its next denoise step"
        : "removed from the queue";
  await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.cancel", target: row.name, ok: !cancelError, detail: cancelError || undefined, before: { job, status: beforeStatus }, after: { job, status, fv_job: fv?.id ?? null, fv_cancel: fvCancel.sent ? fvCancel.runpod_job : fvCancel.reason } });
  if (cancelError && !fvCancel.sent) throw new HttpError(502, `cancel ${job}: ${cancelError}`);
  return { job: rowId, runpod_job: job, before: beforeStatus, status, cancelled: !!cancelled, fv_job: fv ? { id: fv.id, api: fv.api } : null, fv_cancel: fvCancel, note };
}

async function stopFvJob(
  env: SlsEnv,
  who: Actor,
  row: SlsRow,
  eid: string,
  o: { fv: { id: string; api: FvApi; done: boolean } | null; beforeStatus: string; input: any; want: boolean; explicit: boolean },
): Promise<FvCancel> {
  const { fv } = o;
  if (!fv) return { sent: false, reason: o.input && o.input.kind !== "http" && !o.input.path ? `a ${o.input.kind || "?"} job creates no fv-serve job` : "no fv-serve job id is known (give fv_job to cancel one)" };
  const base = { api: fv.api, id: fv.id };
  if (!o.want) return { ...base, sent: false, reason: "not asked (stop_fv_job: false)" };
  if (fv.done && !o.explicit) return { ...base, sent: false, reason: "the fv-serve job had already finished" };
  if (o.beforeStatus === "IN_QUEUE") return { ...base, sent: false, reason: "the queue job never reached a worker" };
  if (o.beforeStatus === "IN_PROGRESS" && o.input?.cancel_path && !o.explicit) return { ...base, sent: false, reason: `the worker DELETEs ${o.input.cancel_path} itself when it stops the queue job (cancel_path)` };
  const route = cancelRoute(fv.api, fv.id);
  if (!route) return { ...base, sent: false, reason: `no cancel route for ${fv.api}` };
  const health = await sls.health(env, eid).catch(() => null);
  if (health && readyWorkers(health) === 0) return { ...base, sent: false, reason: "no worker is up: fv-serve jobs live in the worker's process, so the job ended with it" };
  const input = { kind: "http", method: route.method, path: route.path };
  const r = await sls.run(env, eid, { input, policy: { executionTimeout: 60_000 } });
  const ins = await env.DB.prepare("INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, input, actor) VALUES (?, ?, ?, ?, 0, ?, ?, ?) RETURNING id")
    .bind(row.id, r?.id ?? null, `cancel:${fv.api}`, String(r?.status || "IN_QUEUE"), now(), JSON.stringify(input), who.actor)
    .first<{ id: number }>();
  const workers = health ? readyWorkers(health) : null;
  return {
    ...base,
    sent: true,
    route: `${route.method} ${route.path}`,
    runpod_job: r?.id ?? null,
    job: ins?.id ?? null,
    reason: workers === 1 ? "sent as a queue job to the endpoint's one worker" : `sent as a queue job; with ${workers ?? "several"} workers up it may land on another one (it answers 404 there)`,
  };
}

/** What POST /api/serverless/:id/purge takes (schema serverless-purge). */
export interface SlsPurgeIn {
  confirm: string;
  expected?: number;
}
const queued = (h: any) => Number(h?.jobs?.inQueue ?? 0);
export async function purgeSlsQueue(env: SlsEnv, who: Actor, row: SlsRow, x: SlsPurgeIn) {
  const eid = live(row);
  if (x.confirm !== row.name && x.confirm !== eid) throw new HttpError(400, `confirm: type the endpoint's name (${row.name}) to purge its queue`, { issues: [{ path: ["confirm"], message: `type ${row.name}` }] });
  const h0 = await sls.health(env, eid);
  const before = queued(h0);
  if (x.expected !== undefined && before > x.expected) throw new HttpError(409, `the queue grew to ${before} jobs since you looked (${x.expected}): look again before purging`);
  let r: any;
  try {
    r = await sls.purge(env, eid);
  } catch (e) {
    await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.purge", target: row.name, ok: false, detail: scrub(env, (e as Error).message), before: { queued: before } });
    throw e;
  }
  const h1 = await sls.health(env, eid).catch(() => null);
  // fv-control's recorded jobs that were waiting: their status now.
  const waiting = await env.DB.prepare("SELECT id FROM serverless_jobs WHERE endpoint = ? AND job_id IS NOT NULL AND finished_at IS NULL ORDER BY id DESC LIMIT 20").bind(row.id).all<{ id: number }>();
  for (const j of waiting.results || []) await pollJob(env, row, j.id).catch(() => null);
  const removed = typeof r?.removed === "number" ? r.removed : null;
  await audit(env, { actor: who.actor, ip: who.ip, action: "serverless.purge", target: row.name, before: { queued: before }, after: { removed, queued: h1 ? queued(h1) : null, status: r?.status ?? null } });
  return {
    removed,
    status: r?.status ?? null,
    queued_before: before,
    queued_after: h1 ? queued(h1) : null,
    in_progress: h1 ? Number(h1.jobs?.inProgress ?? 0) : null,
    note: "running jobs are not touched by a purge: cancel them one by one",
  };
}
