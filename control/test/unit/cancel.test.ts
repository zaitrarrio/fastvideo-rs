// Cancelling jobs (docs/control/serverless.md "Cancel and purge",
// docs/control/README.md "Jobs"): the API-to-route map, serverless cancel
// and purge against a mocked Runpod queue API, and the cluster Jobs view
// and cancel against mocked jobs D1s (the edge's, fv-jobs) and pods.
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { apiOfSubmit, cancelRoute, defaultCancelPath, fvJobOf } from "../../src/jobapi";
import { cancelJobRow, cancelQueued, findJob, listJobs } from "../../src/jobs";
import { validate } from "../../src/schemas";
import { cancelSlsJob, purgeSlsQueue } from "../../src/serverless/cancel";
import { getRow, withCancelPath } from "../../src/serverless/ops";
import { normalizeSpec } from "../../src/cluster/spec";
import { insertCluster, newSecrets, secretsOf } from "../../src/cluster/store";
import { d1 } from "./d1shim";

const KEY = "rpa_TESTKEY_0123456789";
const res = (code: number, body: unknown) => new Response(body === undefined ? null : JSON.stringify(body), { status: code, headers: { "content-type": "application/json" } });

describe("which API owns a job id, and its cancel route", () => {
  it("each API's own route; none for LTX and Reactor; fal under its app", () => {
    expect(cancelRoute("native", "fvjob_1")).toEqual({ method: "DELETE", path: "/fv/v1/jobs/fvjob_1" });
    expect(cancelRoute("openai_videos", "video_gen_1")).toEqual({ method: "DELETE", path: "/v1/videos/video_gen_1" });
    expect(cancelRoute("fastwan", "p1")).toEqual({ method: "DELETE", path: "/video/p1" });
    expect(cancelRoute("minimax_v2", "123456789012345678")).toEqual({ method: "DELETE", path: "/v2/video_generation/123456789012345678" });
    expect(cancelRoute("fal", "6d14705e", "minimax/h3/reference-to-video")).toEqual({ method: "PUT", path: "/minimax/h3/reference-to-video/requests/6d14705e/cancel" });
    expect(cancelRoute("fal", "x", null)).toBeNull();
    expect(cancelRoute("fal", "x", "../../etc passwd")).toBeNull();
    expect(cancelRoute("ltx_v2", "x")).toBeNull();
    expect(cancelRoute("reactor", "x")).toBeNull();
    expect(cancelRoute("native", "a/b")!.path).toBe("/fv/v1/jobs/a%2Fb");
  });
  it("a waiting http invoke gets its API's cancel_path; others are left alone", () => {
    expect(apiOfSubmit("/v2/video_generation?x=1")).toBe("minimax_v2");
    expect(defaultCancelPath("/fv/v1/jobs")).toBe("/fv/v1/jobs/{id}");
    expect(defaultCancelPath("/v1/videos")).toBe("/v1/videos/{id}");
    expect(defaultCancelPath("/fal-ai/x")).toBeNull();
    const w = { kind: "http", method: "POST", path: "/fv/v1/jobs", body: {}, wait: true };
    expect(withCancelPath(w).cancel_path).toBe("/fv/v1/jobs/{id}");
    expect(withCancelPath({ ...w, cancel_path: "/mine/{id}" }).cancel_path).toBe("/mine/{id}");
    expect(withCancelPath({ ...w, wait: false }).cancel_path).toBeUndefined();
    expect(withCancelPath({ kind: "info" })).toEqual({ kind: "info" });
    expect(withCancelPath({ ...w, method: "GET" }).cancel_path).toBeUndefined();
    // The worker's envelope takes cancel_path (deny_unknown_fields in crates/fastvideo-deploy).
  });
  it("finds the fv-serve job in a queue job's output: submit reply, wait, progress poll path, MiniMax", () => {
    const http = { kind: "http", method: "POST", path: "/fv/v1/jobs" };
    expect(fvJobOf(http, { status: 202, body: { id: "fvjob_a", status: "queued" } })).toEqual({ id: "fvjob_a", api: "native", done: false });
    expect(fvJobOf(http, { status: 200, body: { id: "fvjob_a", status: "succeeded" }, submit: { id: "fvjob_a", status: "queued" } })).toEqual({ id: "fvjob_a", api: "native", done: true });
    expect(fvJobOf(http, { state: "running", poll_path: "/fv/v1/jobs/fvjob_b" })).toEqual({ id: "fvjob_b", api: "native", done: false });
    expect(fvJobOf({ ...http, path: "/v2/video_generation" }, { state: "processing", poll_path: "/v2/query/video_generation?task_id=123" })).toEqual({ id: "123", api: "minimax_v2", done: false });
    expect(fvJobOf(http, JSON.stringify({ status: 202, body: { id: "fvjob_c" } }))!.id).toBe("fvjob_c");
    expect(fvJobOf({ kind: "info" }, { engine: "fake" })).toBeNull();
    expect(fvJobOf({ ...http, method: "GET" }, { body: { id: "x" } })).toBeNull();
    expect(fvJobOf(http, null)).toBeNull();
  });
  it("the schemas: job ids, the APIs a serverless cancel names, purge, filters", () => {
    expect(validate("serverless-cancel", { job: "c80ffee7-4b3d-4b52-9a39-95d4c1f0a1b2-e1" }).ok).toBe(true);
    expect(validate("serverless-cancel", { job: "x y" }).ok).toBe(false);
    expect(validate("serverless-cancel", { job: "j", fv_api: "fal" }).ok).toBe(false);
    expect(validate("serverless-cancel", { job: "j", bogus: 1 }).ok).toBe(false);
    expect(validate("serverless-purge", { confirm: "h3-max2", expected: 4 }).ok).toBe(true);
    expect(validate("serverless-purge", {}).ok).toBe(false);
    expect(validate("jobs-query", { status: "queued,running", limit: "50" }).ok).toBe(true);
    expect(validate("jobs-query", { status: "queued,bogus" }).ok).toBe(false);
    expect(validate("job-cancel", { job: "fvjob_x", cluster: "tiny" }).ok).toBe(true);
    expect(validate("jobs-cancel-queued", { max: 0 }).ok).toBe(false);
  });
});

// ---------------------------------------------------------------- serverless
function mockQueue() {
  const m = {
    calls: [] as { method: string; path: string; body: any }[],
    jobs: new Map<string, any>(),
    queued: 4,
    running: 0,
    workers: 1,
    ran: [] as any[],
  };
  const fetchMock = async (input: any, init: any = {}) => {
    const u = new URL(String(input));
    const method = (init.method || "GET").toUpperCase();
    const body = init.body ? JSON.parse(init.body) : undefined;
    m.calls.push({ method, path: u.pathname, body });
    expect(init.headers?.authorization).toBe(`Bearer ${KEY}`);
    if (u.host !== "api.runpod.ai") return res(500, { error: "mock: no route" });
    const mm = /^\/v2\/([^/]+)\/(health|run|purge-queue|status\/(.+)|cancel\/(.+))$/.exec(u.pathname);
    if (!mm) return res(404, {});
    if (mm[2] === "health") return res(200, { jobs: { inQueue: m.queued, inProgress: m.running, completed: 9, failed: 0 }, workers: { idle: m.workers, running: 0, ready: 0, initializing: 0 } });
    if (mm[2] === "purge-queue" && method === "POST") {
      const removed = m.queued;
      m.queued = 0;
      for (const j of m.jobs.values()) if (j.status === "IN_QUEUE") j.status = "CANCELLED";
      return res(200, { removed, status: "completed" });
    }
    if (mm[2] === "run" && method === "POST") {
      const id = `run-${m.ran.length + 1}`;
      m.ran.push(body);
      m.jobs.set(id, { id, status: "IN_QUEUE" });
      return res(200, { id, status: "IN_QUEUE" });
    }
    if (mm[3]) {
      const j = m.jobs.get(decodeURIComponent(mm[3]));
      return j ? res(200, j) : res(404, { error: "request does not exist" });
    }
    if (mm[4] && method === "POST") {
      const j = m.jobs.get(decodeURIComponent(mm[4]));
      if (!j) return res(404, { error: "request does not exist" });
      j.status = "CANCELLED";
      return res(200, { id: j.id, status: "CANCELLED" });
    }
    return res(404, {});
  };
  return { m, fetchMock };
}

describe("serverless: cancel a job, purge the queue", () => {
  let env: any;
  let q: ReturnType<typeof mockQueue>;
  const who = { actor: "token:test", ip: "1.2.3.4" };
  const row = async (mode = "queue") => {
    await env.DB.prepare("INSERT INTO serverless_endpoints (id, name, endpoint_id, template_id, mode, spec, status, created_at, created_by, updated_at) VALUES (?, ?, ?, 't', ?, '{}', 'scaled-down', 1, 'x', 1)")
      .bind(`se_${mode}`, `ep-${mode}`, `rp${mode}0000000`, mode)
      .run();
    return getRow(env, `se_${mode}`);
  };
  const invoke = async (r: any, job: string, status: string, input: any, output: any = null) =>
    (await env.DB.prepare("INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, input, output, actor) VALUES (?, ?, 'runsync', ?, 0, 1, ?, ?, 'owner') RETURNING id").bind(r.id, job, status, JSON.stringify(input), output ? JSON.stringify(output) : null).first())!.id as number;
  beforeEach(() => {
    q = mockQueue();
    vi.stubGlobal("fetch", q.fetchMock);
    env = { DB: d1(), RUNPOD_API_KEY: KEY, BALANCE_FLOOR: "8" };
  });
  afterEach(() => vi.unstubAllGlobals());

  it("a queued job pasted by id (not submitted by fv-control): cancelled, recorded so its status shows, audited", async () => {
    const r = await row();
    q.m.jobs.set("ext-1", { id: "ext-1", status: "IN_QUEUE" });
    const out = await cancelSlsJob(env, who, r, { job: "ext-1" });
    expect(out).toMatchObject({ runpod_job: "ext-1", before: "IN_QUEUE", status: "CANCELLED", cancelled: true, fv_job: null, note: "removed from the queue" });
    expect(out.fv_cancel.sent).toBe(false);
    expect(q.m.calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual([`/v2/${r.endpoint_id}/cancel/ext-1`]);
    const rec = await env.DB.prepare("SELECT route, status, finished_at FROM serverless_jobs WHERE id = ?").bind(out.job).first();
    expect(rec).toMatchObject({ route: "external", status: "CANCELLED" });
    expect(rec.finished_at).toBeTruthy();
    const a = await env.DB.prepare("SELECT action, target, ok, ip FROM audit").all();
    expect(a.results).toEqual([{ action: "serverless.cancel", target: "ep-queue", ok: 1, ip: "1.2.3.4" }]);
  });
  it("a running invoke that waits with a cancel_path: Runpod's cancel is enough (the worker DELETEs it on job-stop)", async () => {
    const r = await row();
    const input = withCancelPath({ kind: "http", method: "POST", path: "/fv/v1/jobs", body: { model: "fake-wan" }, wait: true });
    const id = await invoke(r, "job-run", "IN_PROGRESS", input);
    q.m.jobs.set("job-run", { id: "job-run", status: "IN_PROGRESS", output: { state: "running", poll_path: "/fv/v1/jobs/fvjob_r" } });
    const out = await cancelSlsJob(env, who, r, { job: String(id) });
    expect(out).toMatchObject({ job: id, before: "IN_PROGRESS", status: "CANCELLED", fv_job: { id: "fvjob_r", api: "native" } });
    expect(out.fv_cancel).toMatchObject({ sent: false });
    expect(out.fv_cancel.reason).toMatch(/cancel_path/);
    expect(q.m.ran).toEqual([]);
  });
  it("a fire-and-forget submit (COMPLETED, its fv-serve job still running): the native DELETE goes as a queue job to the one worker", async () => {
    const r = await row();
    q.m.jobs.set("job-ff", { id: "job-ff", status: "COMPLETED", output: { status: 202, body: { id: "fvjob_ff", status: "queued" } } });
    const id = await invoke(r, "job-ff", "COMPLETED", { kind: "http", method: "POST", path: "/fv/v1/jobs", body: {} });
    const out = await cancelSlsJob(env, who, r, { job: "job-ff" });
    expect(out.job).toBe(id);
    expect(out.cancelled).toBe(false);
    expect(out.note).toMatch(/already finished/);
    expect(out.fv_cancel).toMatchObject({ sent: true, api: "native", id: "fvjob_ff", route: "DELETE /fv/v1/jobs/fvjob_ff", runpod_job: "run-1" });
    expect(out.fv_cancel.reason).toMatch(/one worker/);
    expect(q.m.ran).toEqual([{ input: { kind: "http", method: "DELETE", path: "/fv/v1/jobs/fvjob_ff" }, policy: { executionTimeout: 60000 } }]);
    const c = await env.DB.prepare("SELECT route, job_id FROM serverless_jobs WHERE id = ?").bind(out.fv_cancel.job).first();
    expect(c).toEqual({ route: "cancel:native", job_id: "run-1" });
  });
  it("an fv-serve job named by hand, another API; none sent with no worker up (the job died with it); stop_fv_job: false", async () => {
    const r = await row();
    q.m.jobs.set("j2", { id: "j2", status: "IN_PROGRESS" });
    const out = await cancelSlsJob(env, who, r, { job: "j2", fv_job: "video_gen_9", fv_api: "openai_videos" });
    expect(out.fv_cancel).toMatchObject({ sent: true, route: "DELETE /v1/videos/video_gen_9" });
    q.m.workers = 0;
    q.m.jobs.set("j3", { id: "j3", status: "COMPLETED", output: { status: 202, body: { id: "fvjob_3" } } });
    await invoke(r, "j3", "COMPLETED", { kind: "http", path: "/fv/v1/jobs" });
    const o3 = await cancelSlsJob(env, who, r, { job: "j3" });
    expect(o3.fv_cancel).toMatchObject({ sent: false, id: "fvjob_3" });
    expect(o3.fv_cancel.reason).toMatch(/no worker is up/);
    q.m.workers = 1;
    expect((await cancelSlsJob(env, who, r, { job: "j3", stop_fv_job: false })).fv_cancel.reason).toMatch(/not asked/);
  });
  it("readable errors: an unknown job, a load-balancer endpoint, a deleted endpoint, an lb invoke", async () => {
    const r = await row();
    await expect(cancelSlsJob(env, who, r, { job: "nope" })).rejects.toThrow(/Runpod has no job nope on ep-queue/);
    await expect(cancelSlsJob(env, who, r, { job: "77" })).rejects.toThrow(/no test invoke #77/);
    const lb = await row("lb");
    await expect(cancelSlsJob(env, who, lb, { job: "x" })).rejects.toMatchObject({ status: 409, message: expect.stringMatching(/load-balancer/) });
    await expect(purgeSlsQueue(env, who, { ...r, status: "deleted", deleted_at: 5 }, { confirm: "ep-queue" })).rejects.toThrow(/has no queue/);
  });
  it("purge: the name to confirm, refused when the queue grew, then every queued job dropped and the recorded ones updated", async () => {
    const r = await row();
    q.m.jobs.set("w1", { id: "w1", status: "IN_QUEUE" });
    const id = await invoke(r, "w1", "IN_QUEUE", { kind: "info" });
    await expect(purgeSlsQueue(env, who, r, { confirm: "wrong" })).rejects.toMatchObject({ status: 400, message: expect.stringMatching(/type the endpoint's name \(ep-queue\)/) });
    await expect(purgeSlsQueue(env, who, r, { confirm: "ep-queue", expected: 2 })).rejects.toMatchObject({ status: 409 });
    expect(q.m.calls.some((c) => c.path.endsWith("/purge-queue"))).toBe(false);
    const out = await purgeSlsQueue(env, who, r, { confirm: "ep-queue", expected: 4 });
    expect(out).toMatchObject({ removed: 4, status: "completed", queued_before: 4, queued_after: 0 });
    const rec = await env.DB.prepare("SELECT status FROM serverless_jobs WHERE id = ?").bind(id).first();
    expect(rec.status).toBe("CANCELLED");
    const a = await env.DB.prepare("SELECT action, after FROM audit WHERE action = 'serverless.purge'").first();
    expect(JSON.parse(a.after)).toMatchObject({ removed: 4, queued: 0 });
  });
});

// ---------------------------------------------------------------- clusters and standalone pods
const JOBS_SCHEMA = `CREATE TABLE jobs (id TEXT PRIMARY KEY NOT NULL, protocol TEXT NOT NULL, external_id TEXT NOT NULL, owner TEXT, status TEXT NOT NULL, model TEXT NOT NULL,
  resolved_model TEXT NOT NULL, task TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, completed_at INTEGER,
  expires_at INTEGER NOT NULL, worker TEXT, version INTEGER NOT NULL DEFAULT 0, job TEXT NOT NULL, UNIQUE (protocol, external_id))`;
function jobsD1() {
  const db = new DatabaseSync(":memory:");
  db.exec(JOBS_SCHEMA);
  const stmt = (sql: string, args: unknown[] = []): any => ({
    bind: (...a: unknown[]) => stmt(sql, a),
    first: async () => (db.prepare(sql).get(...(args as any[])) as any) ?? null,
    all: async () => ({ results: (db.prepare(sql).all(...(args as any[])) as any[]).map((r) => ({ ...r })) }),
    run: async () => db.prepare(sql).run(...(args as any[])),
  });
  const add = (j: { id: string; ext: string; api?: string; status: string; model?: string; worker?: string | null; at: number; started?: string }) =>
    db
      .prepare("INSERT INTO jobs (id, protocol, external_id, owner, status, model, resolved_model, task, progress, created_at, updated_at, expires_at, worker, job) VALUES (?, ?, ?, 'key_1', ?, ?, 'm', 't2v', 0.5, ?, ?, ?, ?, ?)")
      .run(j.id, j.api || "native", j.ext, j.status, j.model || "fasth3", j.at, j.at, j.at + 1e9, j.worker ?? null, JSON.stringify({ started_at: j.started ?? null, cancel_requested: false, request_echo: { url: "https://pod/files/x?sig=SECRET" } }));
  return { db: { prepare: (sql: string) => stmt(sql) } as unknown as D1Database, add };
}

describe("cluster jobs: the view and the cancel route", () => {
  let env: Env;
  let edgeDb: ReturnType<typeof jobsD1>;
  let fvJobs: ReturnType<typeof jobsD1>;
  let calls: { method: string; url: string; headers: any }[];
  let answer: (url: string, method: string) => Response;
  const who = { actor: "owner" };
  const T = 1_791_000_000_000;
  const pod = async (cid: string, podId: string, pool: string, deleted = false) =>
    env.DB.prepare("INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, created_at, deleted_at, status) VALUES (?, ?, 'worker', ?, ?, ?, ?)").bind(podId, cid, pool, T - 3600_000, deleted ? T + 3600_000 : null, deleted ? "deleted" : "ready").run();
  const cluster = async (name: string, over: object = {}, source = "controller") => {
    const { s, ingestHash } = await newSecrets();
    return insertCluster(env, normalizeSpec({ name, template: "tiny-cpu", ...over }), { images: {}, workers: {} }, s, ingestHash, "owner", source, "running", null);
  };
  beforeEach(() => {
    edgeDb = jobsD1();
    fvJobs = jobsD1();
    calls = [];
    answer = (url) => (url.includes("/fv/v1/internal/jobs/") ? res(200, { id: "x", status: "running", cancel_requested: true }) : res(404, {}));
    vi.stubGlobal("fetch", async (u: any, i: any = {}) => {
      calls.push({ method: (i.method || "GET").toUpperCase(), url: String(u), headers: i.headers || {} });
      return answer(String(u), (i.method || "GET").toUpperCase());
    });
    env = {
      DB: d1(),
      EDGE_DB: edgeDb.db,
      JOBS_DB: fvJobs.db,
      CONTROL_KEK: b64(randomBytes(32)),
      SESSION_SECRET: "s".repeat(40),
      RUNPOD_API_KEY: "rpa_x",
      EDGE_URL: "https://edge.test",
      EDGE_INTERNAL_TOKEN: "edge-it",
      EDGE_ADMIN_TOKEN: "fvadm_edge",
      POD_URL_TEMPLATE: "https://pods.test/{pod}",
    } as unknown as Env;
  });
  afterEach(() => vi.unstubAllGlobals());

  it("edge cluster: its pods' jobs and the ones still queued at the edge, newest first, with pools and counts; no job record leaks", async () => {
    const c = await cluster("edgy");
    await pod(c.id, "podaaaaaaaaaa", "fake");
    await pod(c.id, "podbbbbbbbbbbb", "fake2");
    edgeDb.add({ id: "u1", ext: "fvjob_1", status: "running", worker: "podaaaaaaaaaa", at: T + 1, started: "2026-10-07T03:13:25.961108886Z" });
    edgeDb.add({ id: "u2", ext: "a-fal-id", api: "fal", model: "minimax/h3/text-to-video", status: "queued", worker: null, at: T + 2 });
    edgeDb.add({ id: "u3", ext: "fvjob_3", status: "succeeded", worker: "podbbbbbbbbbbb", at: T + 3 });
    edgeDb.add({ id: "u4", ext: "fvjob_4", status: "running", worker: "someoneelse001", at: T + 4 }); // another cluster's
    edgeDb.add({ id: "u5", ext: "fvjob_5", status: "queued", worker: null, at: T - 7200_000 }); // before this cluster ran
    const v = await listJobs(env, c, {});
    expect(v.source).toBe("edge");
    expect(v.jobs.map((j) => j.external_id)).toEqual(["fvjob_3", "a-fal-id", "fvjob_1"]);
    expect(v.counts).toEqual({ queued: 1, running: 1, succeeded: 1 });
    const run = v.jobs.find((j) => j.id === "u1")!;
    expect(run).toMatchObject({ api: "native", pool: "fake", worker: "podaaaaaaaaaa", cancel_route: "DELETE /fv/v1/jobs/fvjob_1", started_at: Date.parse("2026-10-07T03:13:25.961Z") });
    expect(v.jobs.find((j) => j.id === "u2")).toMatchObject({ worker: null, pool: null, cancel_route: "PUT /minimax/h3/text-to-video/requests/a-fal-id/cancel" });
    expect(JSON.stringify(v)).not.toContain("SECRET");
    expect((await listJobs(env, c, { status: "queued,running" })).jobs.map((j) => j.id)).toEqual(["u2", "u1"]);
    expect((await listJobs(env, c, { pool: "fake2" })).jobs.map((j) => j.id)).toEqual(["u3"]);
    await expect(listJobs(env, c, { pod: "nottheirs0000" })).rejects.toThrow(/not one of edgy's/);
  });

  it("cancel on an edge cluster: the holder's internal route with the edge's token; a job queued at the edge through any live front", async () => {
    const c = await cluster("edgy");
    await pod(c.id, "podaaaaaaaaaa", "fake");
    edgeDb.add({ id: "u1", ext: "fvjob_1", status: "running", worker: "podaaaaaaaaaa", at: T + 1 });
    edgeDb.add({ id: "u2", ext: "a-fal-id", api: "fal", model: "minimax/h3/t2v", status: "queued", worker: null, at: T + 2 });
    const { c: c1, row } = await findJob(env, "fvjob_1");
    expect(c1.id).toBe(c.id);
    expect(row.job).toBeUndefined();
    const r = await cancelJobRow(env, who, c1, row);
    expect(r).toMatchObject({ ok: true, via: "internal", pod: "podaaaaaaaaaa", status: "running", cancel_requested: true });
    expect(calls[0]).toMatchObject({ method: "DELETE", url: "https://pods.test/podaaaaaaaaaa/fv/v1/internal/jobs/u1" });
    expect(calls[0]!.headers["x-fv-internal-token"]).toBe("edge-it");
    answer = () => res(200, { id: "u2", status: "cancelled", cancel_requested: false });
    const q = await findJob(env, "u2"); // by the internal id too
    const r2 = await cancelJobRow(env, who, q.c, q.row);
    expect(r2).toMatchObject({ ok: true, via: "internal", pod: "podaaaaaaaaaa", status: "cancelled", note: "cancelled" });
    const a = await env.DB.prepare("SELECT action, target, ok FROM audit WHERE action = 'job.cancel'").all();
    expect(a.results).toHaveLength(2);
  });

  it("falls back to another front, then to the owning API's route through the edge; readable when nothing worked", async () => {
    const c = await cluster("edgy");
    await pod(c.id, "podaaaaaaaaaa", "fake");
    await pod(c.id, "podbbbbbbbbbbb", "fake");
    edgeDb.add({ id: "u1", ext: "video_gen_1", api: "openai_videos", status: "running", worker: "podaaaaaaaaaa", at: T + 1 });
    answer = (url) => (url.includes("podaaaaaaaaaa") ? res(404, { error: { message: "job u1 was not found" } }) : url.includes("podbbbbbbbbbbb") ? res(200, { status: "running", cancel_requested: true }) : res(500, {}));
    const { row } = await findJob(env, "video_gen_1");
    const r = await cancelJobRow(env, who, c, row);
    expect(r).toMatchObject({ ok: true, pod: "podbbbbbbbbbbb" });
    expect(r.note).toMatch(/family object/);
    expect(r.attempts.map((x) => x.status)).toEqual([404, 200]);
    answer = (url) => (url.startsWith("https://edge.test/v1/videos/video_gen_1") ? res(401, { error: { message: "invalid credentials" } }) : res(502, "no pod"));
    const bad = await cancelJobRow(env, who, c, row);
    expect(bad.ok).toBe(false);
    expect(calls.at(-1)).toMatchObject({ method: "DELETE", url: "https://edge.test/v1/videos/video_gen_1" });
    expect(bad.note).toMatch(/api DELETE \/v1\/videos\/video_gen_1: 401 invalid credentials/);
  });

  it("direct clusters and standalone pods: fv-jobs, the cluster's own internal token; a pod that is gone; a finished job", async () => {
    const c = await cluster("solo", { control_plane: "direct" }, "standalone");
    await pod(c.id, "podsolo000001", "pod");
    await pod(c.id, "podgone000001", "pod", true);
    fvJobs.add({ id: "d1", ext: "fvjob_d1", status: "running", worker: "podsolo000001", at: T + 1 });
    fvJobs.add({ id: "d2", ext: "fvjob_d2", status: "running", worker: "podgone000001", at: T + 2 });
    fvJobs.add({ id: "d3", ext: "fvjob_d3", status: "succeeded", worker: "podsolo000001", at: T + 3 });
    const v = await listJobs(env, c, {});
    expect(v.source).toBe("fv-jobs");
    expect(v.cluster).toMatchObject({ kind: "standalone", control_plane: "direct" });
    expect(v.jobs).toHaveLength(3);
    const tok = (await secretsOf(env, c)).internal_token;
    const r = await cancelJobRow(env, who, c, (await findJob(env, "fvjob_d1")).row);
    expect(r.ok).toBe(true);
    expect(calls[0]!.headers["x-fv-internal-token"]).toBe(tok);
    calls = [];
    const gone = await cancelJobRow(env, who, c, (await findJob(env, "fvjob_d2")).row);
    expect(gone).toMatchObject({ ok: false });
    expect(gone.note).toMatch(/podgone000001\) is gone: the job ended with it/);
    expect(calls).toEqual([]);
    const done = await cancelJobRow(env, who, c, (await findJob(env, "fvjob_d3")).row);
    expect(done).toMatchObject({ ok: true, via: null, note: "already succeeded: nothing to cancel" });
  });

  it("findJob: a pod fv-control did not create is refused; unknown ids say where they looked; no binding is a readable 409", async () => {
    fvJobs.add({ id: "f1", ext: "fvjob_f1", status: "running", worker: "foreignpod0001", at: T });
    await expect(findJob(env, "fvjob_f1")).rejects.toMatchObject({ status: 403, message: expect.stringMatching(/did not create/) });
    await expect(findJob(env, "fvjob_none")).rejects.toMatchObject({ status: 404, message: expect.stringMatching(/the edge's D1 or fv-jobs/) });
    const c = await cluster("edgy");
    await expect(listJobs({ ...env, EDGE_DB: undefined } as Env, c, {})).rejects.toMatchObject({ status: 409, message: expect.stringMatching(/EDGE_DB/) });
    // A jobs D1 no worker wrote to yet: no table, no jobs (not an error).
    const empty = { prepare: () => ({ bind: () => ({ all: async () => Promise.reject(new Error("D1_ERROR: no such table: jobs: SQLITE_ERROR")) }) }) } as unknown as D1Database;
    await pod(c.id, "podaaaaaaaaaa", "fake");
    const v = await listJobs({ ...env, EDGE_DB: empty } as Env, c, {});
    expect(v.jobs).toEqual([]);
    expect(v.note).toMatch(/no jobs table yet/);
    await expect(findJob({ ...env, EDGE_DB: empty, JOBS_DB: empty } as Env, "fvjob_x")).rejects.toMatchObject({ status: 404 });
  });

  it("cancel all queued: one cancel each, a summary, one audit row", async () => {
    const c = await cluster("edgy");
    await pod(c.id, "podaaaaaaaaaa", "fake");
    for (let i = 0; i < 3; i++) edgeDb.add({ id: `q${i}`, ext: `fvjob_q${i}`, status: "queued", worker: i ? "podaaaaaaaaaa" : null, at: T + i });
    edgeDb.add({ id: "r1", ext: "fvjob_r1", status: "running", worker: "podaaaaaaaaaa", at: T + 9 });
    answer = (url) => (/\/(fvjob_)?q2$/.test(url) ? res(500, { error: "boom" }) : res(200, { status: "cancelled" }));
    const r = await cancelQueued(env, who, c, {});
    expect(r).toMatchObject({ queued: 3, cancelled: 2 });
    expect(r.failed.map((f) => f.job)).toEqual(["fvjob_q2"]);
    expect(calls.filter((x) => x.url.includes("/r1"))).toEqual([]);
    const a = await env.DB.prepare("SELECT action, ok, after FROM audit").all<any>();
    expect(a.results.map((x: any) => x.action)).toEqual(["jobs.cancel-queued"]);
    expect(JSON.parse(a.results[0].after)).toMatchObject({ queued: 3, cancelled: 2, failed: 1 });
  });
});
