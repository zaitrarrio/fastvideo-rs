// The serverless console (docs/control/serverless.md "Console",
// src/serverless/console.ts): the bundled pages, the route map, submit
// wrapping, status synthesis, store-backed polling, the capability cache,
// cancel and uploads, against a mocked Runpod queue API and a fake job store.
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CONSOLE_ASSETS, CONSOLE_PAGES } from "../../src/serverless/console-assets";
import {
  cachedGet,
  capsForConsole,
  classify,
  consoleHtml,
  consoleRequest,
  fvIdOf,
  namesOf,
  OFF_PAGES,
  phaseOf,
  recordedInput,
  statusBody,
  submitReply,
  verifyUpload,
  viewOf,
  wrapSubmit,
} from "../../src/serverless/console";
import { getRow } from "../../src/serverless/ops";
import { d1 } from "./d1shim";

const KEY = "rpa_TESTKEY_0123456789";
const EID = "rpcons00000001";
const res = (code: number, body: unknown) => new Response(body === undefined ? null : JSON.stringify(body), { status: code, headers: { "content-type": "application/json" } });
const enc = new TextEncoder();

describe("pages: fv-serve's console, unchanged but for the embedding tags", () => {
  it("every page and asset console.rs embeds is bundled", () => {
    expect(Object.keys(CONSOLE_PAGES).sort()).toEqual(["admin", "avatar", "index", "live", "model", "native", "stream"]);
    for (const a of ["common.js", "home.js", "model.js", "form.js", "snippets.js", "native.js", "rtc.js", "console.css"]) expect(CONSOLE_ASSETS[a]?.body.length).toBeGreaterThan(100);
    expect(CONSOLE_ASSETS["console.css"]!.type).toBe("text/css; charset=utf-8");
    expect(CONSOLE_ASSETS["common.js"]!.body).toContain("fv-console-base");
  });
  it("a page gets the prefix, the off pages and the note; its links and assets move under the prefix", () => {
    const html = consoleHtml("model", "/serverless/ep1", 'note <b>"x"</b>')!;
    expect(html).toContain('<meta name="fv-console-base" content="/serverless/ep1">');
    expect(html).toContain(`<meta name="fv-console-off" content="${OFF_PAGES.join(",")}">`);
    expect(html).toContain("&#60;b&#62;&#34;x&#34;&#60;/b&#62;");
    expect(html).toContain('<script type="module" src="/serverless/ep1/console/assets/model.js">');
    expect(html).toContain('href="/serverless/ep1/console/assets/console.css"');
    expect(html).toContain('<a href="/serverless/ep1/console">Models</a>');
    expect(html).not.toMatch(/(href|src)="\/console/);
    // The tags come before the module scripts (deferred: they read them on load).
    expect(html.indexOf("fv-console-base")).toBeLessThan(html.indexOf("<script"));
    expect(consoleHtml("nope", "/x", "")).toBeNull();
  });
});

describe("the route map", () => {
  const q = new URLSearchParams();
  it("catalog calls are cached; status synthesised; live and admin routes off", () => {
    expect(classify("GET", "/fv/v1/capabilities", q)).toEqual({ kind: "cached", path: "/fv/v1/capabilities" });
    expect(classify("GET", "/fal/schema", q).kind).toBe("cached");
    expect(classify("GET", "/fal/schema/fal-ai/wan/v2.2-5b/text-to-video/fast-wan", q).kind).toBe("cached");
    expect(classify("GET", "/fv/v1/status", q).kind).toBe("status");
    expect(classify("GET", "/schema", q).kind).toBe("reactor-schema");
    for (const p of ["/fv/v1/admin/keys", "/fv/v1/streams", "/wma/session", "/sessions/x/transport/webrtc"]) expect(classify("POST", p, q).kind).toBe("off");
  });
  it("submits, polls and cancels of each API", () => {
    expect(classify("POST", "/fv/v1/jobs", q)).toEqual({ kind: "submit", api: "native", path: "/fv/v1/jobs" });
    expect(classify("GET", "/fv/v1/jobs/abc-e1", q)).toEqual({ kind: "poll", api: "native", id: "abc-e1", view: "job" });
    expect(classify("DELETE", "/fv/v1/jobs/abc-e1", q)).toEqual({ kind: "cancel", api: "native", id: "abc-e1" });
    expect(classify("POST", "/v1/videos", q).kind).toBe("submit");
    expect(classify("GET", "/v1/videos/v1/content", q)).toEqual({ kind: "content", id: "v1" });
    expect(classify("DELETE", "/v1/videos/v1", q)).toEqual({ kind: "cancel", api: "openai_videos", id: "v1" });
    expect(classify("POST", "/v2/video_generation", q)).toMatchObject({ kind: "submit", api: "minimax_v2" });
    expect(classify("GET", "/v2/query/video_generation", new URLSearchParams("task_id=j1"))).toEqual({ kind: "poll", api: "minimax_v2", id: "j1", view: "job" });
    expect(classify("POST", "/minimax/h3-max/text-to-video", q)).toEqual({ kind: "submit", api: "fal", path: "/minimax/h3-max/text-to-video", app: "minimax/h3-max" });
    expect(classify("POST", "/fal-ai/wan/v2.2-5b/text-to-video/fast-wan", q, ["fal-ai/wan"])).toMatchObject({ kind: "submit", app: "fal-ai/wan" });
    expect(classify("GET", "/minimax/h3-max/requests/r-1/status", q)).toEqual({ kind: "poll", api: "fal", id: "r-1", app: "minimax/h3-max", view: "status" });
    expect(classify("GET", "/minimax/h3-max/requests/r-1", q)).toMatchObject({ kind: "poll", view: "result" });
    expect(classify("PUT", "/minimax/h3-max/requests/r-1/cancel", q)).toEqual({ kind: "cancel", api: "fal", id: "r-1", app: "minimax/h3-max" });
    expect(classify("GET", "/v1/files/retrieve", new URLSearchParams("file_id=1"))).toEqual({ kind: "fallback", path: "/v1/files/retrieve" });
    expect(classify("POST", "/x", q).kind).toBe("none");
    expect(classify("GET", "/fv/v1/jobs/../x", q).kind).not.toBe("poll");
  });
});

describe("submit wrapping", () => {
  it("a JSON submit becomes a waiting http job; native / OpenAI / MiniMax get their cancel_path, fal none", () => {
    const j = wrapSubmit("/fv/v1/jobs", "application/json", enc.encode('{"model":"fake-wan","prompt":"p"}'), 600);
    expect(j).toEqual({ kind: "http", method: "POST", path: "/fv/v1/jobs", headers: { "content-type": "application/json" }, body: { model: "fake-wan", prompt: "p" }, wait: true, timeout_s: 600, cancel_path: "/fv/v1/jobs/{id}" });
    expect(wrapSubmit("/v1/videos", "application/json", enc.encode("{}"), 60).cancel_path).toBe("/v1/videos/{id}");
    expect(wrapSubmit("/v2/video_generation", "application/json", enc.encode("{}"), 60).cancel_path).toBe("/v2/video_generation/{id}");
    expect(wrapSubmit("/minimax/h3-max/text-to-video", "application/json", enc.encode("{}"), 60).cancel_path).toBeUndefined();
    expect(() => wrapSubmit("/fv/v1/jobs", "application/json", enc.encode("{"), 60)).toThrow(/not valid JSON/);
  });
  it("a non-JSON body goes as body_b64 with its content type; the recorded input elides long strings and stays JSON", () => {
    const j = wrapSubmit("/v1/videos", "multipart/form-data; boundary=x", enc.encode("--x\r\nabc"), 60);
    expect(j.body_b64).toBe(btoa("--x\r\nabc"));
    expect(j.body).toBeUndefined();
    expect((j.headers as any)["content-type"]).toBe("multipart/form-data; boundary=x");
    const big = wrapSubmit("/fv/v1/jobs", "application/json", enc.encode(JSON.stringify({ model: "m", image_url: `data:image/png;base64,${"A".repeat(5000)}` })), 60);
    const rec = JSON.parse(recordedInput(big));
    expect(rec.body.model).toBe("m");
    expect(rec.body.image_url).toMatch(/…\(5022 chars\)$/);
    expect(rec.cancel_path).toBe("/fv/v1/jobs/{id}");
  });
  it("the immediate reply has each API's shape, the Runpod job id as its id", () => {
    expect(submitReply("fal", "rp-1", "https://c/serverless/e", "minimax/h3-max", {}, 0)).toEqual({
      code: 200,
      body: { request_id: "rp-1", response_url: "https://c/serverless/e/minimax/h3-max/requests/rp-1", status_url: "https://c/serverless/e/minimax/h3-max/requests/rp-1/status", cancel_url: "https://c/serverless/e/minimax/h3-max/requests/rp-1/cancel", queue_position: 0 },
      headers: { "x-fal-request-id": "rp-1" },
    });
    expect(submitReply("native", "rp-1", "", undefined, { model: "m", task: "t2v" }, 0)).toMatchObject({ code: 202, body: { id: "rp-1", object: "fv.job", status: "queued", model: "m", task: "t2v" } });
    expect(submitReply("openai_videos", "rp-1", "", undefined, { model: "m" }, 2000).body).toMatchObject({ id: "rp-1", object: "video", status: "queued", created_at: 2 });
    expect(submitReply("minimax_v2", "rp-1", "", undefined, {}, 0).body).toEqual({ task_id: "rp-1", base_resp: { status_code: 0, status_msg: "success" } });
  });
});

describe("status and results: the phase, then each API's view", () => {
  it("the fv-serve id: the submit reply, else the waiting job's progress poll path", () => {
    expect(fvIdOf("native", { state: "running", poll_path: "/fv/v1/jobs/fvjob_1" })).toBe("fvjob_1");
    expect(fvIdOf("fal", { state: "in_progress", poll_path: "/minimax/h3-max/requests/f-1/status" })).toBe("f-1");
    expect(fvIdOf("minimax_v2", { state: "processing", poll_path: "/v2/query/video_generation?task_id=123" })).toBe("123");
    expect(fvIdOf("openai_videos", { status: 200, body: { status: "completed" }, submit: { id: "video_9" } })).toBe("video_9");
    expect(fvIdOf("fal", { status: 200, body: { video: {} }, submit: { request_id: "f-2" } })).toBe("f-2");
    expect(fvIdOf("native", { status: 422, body: { error: { message: "bad" } } })).toBeNull();
    expect(fvIdOf("native", null)).toBeNull();
  });
  const st = (status: string, progress: number, logs: any[] = []) => ({ status, progress, job: { logs, queue_position: status === "queued" ? 2 : null } });
  it("in flight: the store's state and progress, else Runpod's", () => {
    expect(phaseOf({ status: "IN_QUEUE", output: null, error: null, store: null })).toMatchObject({ phase: "queued", progress: 0 });
    expect(phaseOf({ status: "IN_PROGRESS", output: { state: "running", poll_path: "/x/1" }, error: null, store: null }).phase).toBe("running");
    expect(phaseOf({ status: "IN_PROGRESS", output: { state: "queued" }, error: null, store: null }).phase).toBe("queued");
    expect(phaseOf({ status: "IN_PROGRESS", output: null, error: null, store: st("queued", 0) })).toMatchObject({ phase: "queued", queue_position: 2 });
    const p = phaseOf({ status: "IN_PROGRESS", output: null, error: null, store: st("running", 0.4, [{ message: "step 4/10", level: "info", timestamp: "t" }]) });
    expect(p).toMatchObject({ phase: "running", progress: 0.4, logs: [{ message: "step 4/10", level: "info", timestamp: "t" }] });
    // The store already says succeeded: still running until the queue job hands over the worker's own final reply.
    expect(phaseOf({ status: "IN_PROGRESS", output: null, error: null, store: st("succeeded", 1) }).phase).toBe("running");
  });
  it("finished: the worker's reply; refused submits, failures and cancels", () => {
    const done = phaseOf({ status: "COMPLETED", output: { status: 200, body: { id: "fvjob_1", status: "succeeded", output: { url: "https://r2/x.mp4?sig" } }, submit: { id: "fvjob_1" } }, error: null, store: null });
    expect(done).toMatchObject({ phase: "done", reply: { code: 200 } });
    const refused = phaseOf({ status: "COMPLETED", output: { status: 422, headers: {}, body: { error: { message: "prompt is required" } } }, error: null, store: null });
    expect(refused).toMatchObject({ phase: "failed", message: "prompt is required", reply: { code: 422 } });
    expect(phaseOf({ status: "COMPLETED", output: { status: 200, body: { status: "failed", error: { message: "oom" } }, submit: {} }, error: null, store: null }).phase).toBe("failed");
    expect(phaseOf({ status: "FAILED", output: null, error: '{"error_type":"WaitTimeout","error_message":"took too long"}', store: null })).toMatchObject({ phase: "failed", message: "took too long" });
    expect(phaseOf({ status: "FAILED", output: null, error: '{"error_type":"Cancelled","error_message":"cancelled while waiting"}', store: null }).phase).toBe("cancelled");
    expect(phaseOf({ status: "CANCELLED", output: null, error: null, store: null }).phase).toBe("cancelled");
    expect(phaseOf({ status: "TIMED_OUT", output: null, error: null, store: null }).message).toMatch(/timed out/);
  });
  const rec = { submitted_at: 1_000_000, input: { body: { model: "fake-wan", task: "t2v", prompt: "a cat" } } };
  const B = "https://c/serverless/e";
  it("native and OpenAI: synthesised while in flight, the worker's own view (id swapped) once done", () => {
    const running = phaseOf({ status: "IN_PROGRESS", output: null, error: null, store: st("running", 0.5) });
    expect(viewOf("native", "job", "rp-1", B, undefined, running, rec)).toMatchObject({ code: 200, body: { id: "rp-1", object: "fv.job", status: "running", progress: 0.5, model: "fake-wan", task: "t2v", output: null } });
    expect(viewOf("openai_videos", "job", "rp-1", B, undefined, running, rec).body).toMatchObject({ id: "rp-1", object: "video", status: "in_progress", progress: 50, prompt: "a cat" });
    const done = phaseOf({ status: "COMPLETED", output: { status: 200, body: { id: "fvjob_1", status: "succeeded", output: { url: "https://r2/x.mp4" } }, submit: { id: "fvjob_1" } }, error: null, store: null });
    expect(viewOf("native", "job", "rp-1", B, undefined, done, rec)).toEqual({ code: 200, body: { id: "rp-1", status: "succeeded", output: { url: "https://r2/x.mp4" } } });
    const failed = phaseOf({ status: "FAILED", output: null, error: '{"error_message":"worker lost"}', store: null });
    expect(viewOf("native", "job", "rp-1", B, undefined, failed, rec).body).toMatchObject({ status: "failed", error: { message: "worker lost" } });
    expect(viewOf("openai_videos", "job", "rp-1", B, undefined, failed, rec).body).toMatchObject({ status: "failed", error: { code: "generation_failed", message: "worker lost" } });
    const refused = phaseOf({ status: "COMPLETED", output: { status: 422, body: { error: { message: "bad seconds" } } }, error: null, store: null });
    expect(viewOf("native", "job", "rp-1", B, undefined, refused, rec).body).toMatchObject({ status: "failed", error: { message: "bad seconds" } });
  });
  it("fal: IN_QUEUE / IN_PROGRESS with logs / COMPLETED; the result is the worker's own reply", () => {
    const q = viewOf("fal", "status", "rp-1", B, "minimax/h3-max", phaseOf({ status: "IN_QUEUE", output: null, error: null, store: null }), rec);
    expect(q).toEqual({ code: 202, body: { request_id: "rp-1", response_url: `${B}/minimax/h3-max/requests/rp-1`, status_url: `${B}/minimax/h3-max/requests/rp-1/status`, cancel_url: `${B}/minimax/h3-max/requests/rp-1/cancel`, status: "IN_QUEUE", queue_position: 0 } });
    const r = viewOf("fal", "status", "rp-1", B, "minimax/h3-max", phaseOf({ status: "IN_PROGRESS", output: null, error: null, store: st("running", 0.2, [{ message: "denoise", level: "info", timestamp: "2026-10-07T00:00:00Z" }]) }), rec);
    expect(r.body).toMatchObject({ status: "IN_PROGRESS", logs: [{ message: "denoise", level: "INFO", source: "USER", timestamp: "2026-10-07T00:00:00Z" }] });
    expect(viewOf("fal", "result", "rp-1", B, "minimax/h3-max", phaseOf({ status: "IN_PROGRESS", output: null, error: null, store: null }), rec)).toEqual({ code: 400, body: { detail: "Request is still in progress" } });
    const done = phaseOf({ status: "COMPLETED", output: { status: 200, body: { video: { url: "https://r2/v.mp4" }, seed: 7 }, submit: { request_id: "f-1" } }, error: null, store: null });
    expect(viewOf("fal", "status", "rp-1", B, "minimax/h3-max", done, rec).body.status).toBe("COMPLETED");
    expect(viewOf("fal", "result", "rp-1", B, "minimax/h3-max", done, rec)).toEqual({ code: 200, body: { video: { url: "https://r2/v.mp4" }, seed: 7 } });
    const bad = phaseOf({ status: "COMPLETED", output: { status: 422, body: { detail: [{ loc: ["body", "prompt"], msg: "required" }] } }, error: null, store: null });
    expect(viewOf("fal", "status", "rp-1", B, "minimax/h3-max", bad, rec).body).toMatchObject({ status: "COMPLETED", error: expect.any(String) });
    expect(viewOf("fal", "result", "rp-1", B, "minimax/h3-max", bad, rec).code).toBe(422);
  });
  it("MiniMax: Queueing / Processing / the worker's query reply", () => {
    expect(viewOf("minimax_v2", "job", "rp-1", B, undefined, phaseOf({ status: "IN_QUEUE", output: null, error: null, store: null }), rec).body).toMatchObject({ task_id: "rp-1", status: "Queueing" });
    const done = phaseOf({ status: "COMPLETED", output: { status: 200, body: { task_id: "123", status: "Success", file_id: "9" }, submit: { task_id: "123" } }, error: null, store: null });
    expect(viewOf("minimax_v2", "job", "rp-1", B, undefined, done, rec).body).toEqual({ task_id: "rp-1", status: "Success", file_id: "9" });
  });
});

describe("status synthesis", () => {
  const row: any = { name: "h3", mode: "queue", status: "active", deleted_at: null, workers: 0 };
  const caps = { models: [{ caps: { id: "fasth3", served_names: ["fasth3", "MiniMax-H3"] } }], aliases: { "MiniMax-H3-Turbo": "fasth3" }, tiers: [{ alias: "h3-turbo", model: "fasth3" }] };
  const h = (w: any, jobs: any = { inQueue: 1, inProgress: 0 }) => ({ jobs, workers: { idle: 0, running: 0, ready: 0, initializing: 0, throttled: 0, unhealthy: 0, ...w } });
  it("one pool from Runpod's workers: scaled to zero, loading, busy, ready; down when not active", () => {
    const s = statusBody(row, h({}), caps);
    expect(s).toMatchObject({ object: "fv.status", gateway: false, state: "scaled_to_zero", models: { fasth3: { state: "scaled_to_zero", pools: ["h3"] } } });
    expect(s.pools[0]).toMatchObject({ id: "h3", kind: "runpod-serverless", available: true, queued: 1, running: 0, worker_counts: { idle: 0 }, workers: [] });
    expect(statusBody(row, h({ initializing: 1 }), caps).state).toBe("loading");
    expect(statusBody(row, h({ running: 1 }), caps).state).toBe("busy");
    expect(statusBody(row, h({ idle: 1, running: 1 }), caps).state).toBe("ready");
    expect(statusBody(row, h({ unhealthy: 2 }), caps).pools[0].available).toBe(false);
    expect(statusBody({ ...row, status: "scaled-down" }, null, caps).state).toBe("down");
    expect(statusBody({ ...row, mode: "lb", workers: 1 }, null, caps).state).toBe("ready");
    expect(statusBody(row, h({}), null).models).toEqual({});
  });
  it("names: ids, served names, aliases and tier aliases → the model id (the model page's pool badge)", () => {
    expect(namesOf(caps)).toEqual({ fasth3: "fasth3", "MiniMax-H3": "fasth3", "MiniMax-H3-Turbo": "fasth3", "h3-turbo": "fasth3" });
  });
  it("cached capabilities tell the console no key is needed", () => {
    expect(capsForConsole({ object: "fv.capabilities", auth: { mode: "trust-gateway" }, models: [] })).toMatchObject({ auth: { mode: "none" }, models: [] });
  });
});

// ---------------------------------------------------------------- the handler against a mocked queue
function mockQueue() {
  const m = {
    calls: [] as { method: string; path: string; body: any }[],
    jobs: new Map<string, any>(),
    ran: [] as any[],
    workers: { idle: 0, running: 0, ready: 0, initializing: 0, throttled: 0, unhealthy: 0 } as Record<string, number>,
    /** What a new job becomes on its first status poll (the simulated worker). */
    onRun: (_input: any): any => ({ status: "IN_QUEUE" }),
    fetched: [] as string[],
  };
  const fetchMock = async (input: any, init: any = {}) => {
    const u = new URL(String(input));
    const method = (init.method || "GET").toUpperCase();
    const body = init.body ? JSON.parse(init.body) : undefined;
    m.calls.push({ method, path: u.pathname, body });
    if (u.host === "r2.example") {
      m.fetched.push(u.href);
      return new Response("MP4BYTES", { headers: { "content-type": "video/mp4" } });
    }
    if (u.host === "api.runpod.io" && u.pathname === "/graphql") return res(200, { data: { myself: { clientBalance: 50, endpoints: [] } } });
    expect(init.headers?.authorization).toBe(`Bearer ${KEY}`);
    const mm = /^\/v2\/([^/]+)\/(health|run|status\/(.+)|cancel\/(.+))$/.exec(u.pathname);
    if (u.host !== "api.runpod.ai" || !mm) return res(404, { error: "mock: no route" });
    if (mm[2] === "health") return res(200, { jobs: { inQueue: [...m.jobs.values()].filter((j) => j.status === "IN_QUEUE").length, inProgress: 0 }, workers: m.workers });
    if (mm[2] === "run") {
      const id = `rp-${m.ran.length + 1}-e1`;
      m.ran.push(body);
      m.jobs.set(id, { id, status: "IN_QUEUE", next: m.onRun(body.input) });
      return res(200, { id, status: "IN_QUEUE" });
    }
    const j = m.jobs.get(decodeURIComponent(mm[3] || mm[4] || ""));
    if (!j) return res(404, { error: "request does not exist" });
    if (mm[4]) {
      j.status = "CANCELLED";
      return res(200, { id: j.id, status: "CANCELLED" });
    }
    if (j.next) Object.assign(j, j.next, { next: undefined });
    const { next: _n, ...out } = j;
    return res(200, out);
  };
  return { m, fetchMock };
}

describe("the console handler", () => {
  let env: any;
  let q: ReturnType<typeof mockQueue>;
  let jobsDb: DatabaseSync;
  const who = { actor: "owner", ip: "1.2.3.4" };
  const O = "https://fvc.example";
  const P = `${O}/serverless/${EID}`;
  const call = async (method: string, path: string, body?: unknown, headers: Record<string, string> = {}) => {
    const row = await getRow(env, EID);
    const r = await consoleRequest(env, new Request(`${P}${path}`, { method, body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body), headers: { "content-type": "application/json", ...headers } }), row, { ep: EID, who });
    const text = await r.text();
    let j: any = null;
    try {
      j = JSON.parse(text);
    } catch {}
    return { status: r.status, j, text, headers: r.headers };
  };
  const storeRow = (ext: string, api: string, status: string, progress: number, logs: any[] = []) =>
    jobsDb.prepare("INSERT OR REPLACE INTO jobs (id, protocol, external_id, status, progress, job) VALUES (?, ?, ?, ?, ?, ?)").run(`u-${ext}`, api, ext, status, progress, JSON.stringify({ logs, queue_position: null }));
  beforeEach(async () => {
    q = mockQueue();
    vi.stubGlobal("fetch", q.fetchMock);
    jobsDb = new DatabaseSync(":memory:");
    jobsDb.exec("CREATE TABLE jobs (id TEXT PRIMARY KEY, protocol TEXT, external_id TEXT, status TEXT, progress REAL, job TEXT)");
    const jobs = { prepare: (sql: string) => ({ bind: (...a: any[]) => ({ first: async () => jobsDb.prepare(sql).get(...a) ?? null }) }) };
    const r2 = new Map<string, { body: string; type: string }>();
    env = {
      DB: d1(),
      JOBS_DB: jobs,
      RUNPOD_API_KEY: KEY,
      BALANCE_FLOOR: "8",
      SESSION_SECRET: "sess_test_secret_0123456789",
      CONSOLE_SUBMIT_WAIT_MS: "0",
      CONSOLE_CACHE_WAIT_MS: "50",
      CONSOLE_POLL_MS: "5",
      LOGS: {
        put: async (k: string, b: any, o: any) => void r2.set(k, { body: await new Response(b).text(), type: o?.httpMetadata?.contentType }),
        get: async (k: string) => (r2.has(k) ? { body: r2.get(k)!.body, size: r2.get(k)!.body.length, httpMetadata: { contentType: r2.get(k)!.type } } : null),
      },
    };
    const spec = JSON.stringify({ name: "cons", mode: "queue", variant: "cpu", compute: "CPU", workers_min: 0, workers_max: 1, execution_timeout_s: 600 });
    await env.DB.prepare("INSERT INTO serverless_endpoints (id, name, endpoint_id, template_id, mode, spec, image, status, created_at, created_by, updated_at) VALUES ('se_c', 'cons', ?, 't', 'queue', ?, 'ghcr.io/x@sha256:abcdef0123456789', 'active', 1, 'x', 1)")
      .bind(EID, spec)
      .run();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("pages: the console under the prefix; live pages off with a note; assets", async () => {
    const p = await call("GET", "/console");
    expect(p.status).toBe(200);
    expect(p.headers.get("content-security-policy")).toContain("script-src 'self'");
    expect(p.text).toContain(`<meta name="fv-console-base" content="/serverless/${EID}">`);
    expect(p.text).toContain("fvc-cons");
    expect((await call("GET", "/console/models/minimax/h3-max/text-to-video")).text).toContain("model.js");
    for (const off of ["/console/stream", "/console/live", "/console/avatar", "/console/admin", "/console/models/minimax/h3-max/director"]) {
      const r = await call("GET", off);
      expect(r.status, off).toBe(404);
      expect(r.text).toContain("Not available on a serverless endpoint");
    }
    const a = await call("GET", "/console/assets/common.js");
    expect(a.headers.get("content-type")).toBe("text/javascript; charset=utf-8");
    expect((await call("GET", "/console/assets/nope.js")).status).toBe(404);
    expect(q.m.ran).toEqual([]);
  });

  it("capabilities: one queue job on a miss, then the cache (no Runpod call); a cold start answers 503 and keeps the job", async () => {
    const caps = { object: "fv.capabilities", auth: { mode: "trust-gateway" }, models: [{ caps: { id: "fake-wan" } }] };
    q.m.onRun = (input) => ({ status: "COMPLETED", output: { status: 200, headers: {}, body: input.path === "/fv/v1/capabilities" ? caps : { apps: [] } } });
    const a = await call("GET", "/fv/v1/capabilities");
    expect(a.status).toBe(200);
    expect(a.j).toMatchObject({ auth: { mode: "none" }, models: [{ caps: { id: "fake-wan" } }] });
    expect(q.m.ran.map((r) => r.input)).toEqual([{ kind: "http", method: "GET", path: "/fv/v1/capabilities" }]);
    const n = q.m.calls.length;
    const b = await call("GET", "/fv/v1/capabilities");
    expect(b.j.models).toHaveLength(1);
    expect(b.headers.get("x-fv-console-cache")).toMatch(/^cache; age=/);
    expect(q.m.calls.length).toBe(n);
    // A cold endpoint: the job waits in the queue; the page is told, the job stays pending (no second job).
    q.m.onRun = () => ({ status: "IN_QUEUE" });
    const c = await call("GET", "/fal/schema");
    expect(c.status).toBe(503);
    expect(c.j.error.message).toMatch(/cold start/);
    expect(c.headers.get("retry-after")).toBe("10");
    const c2 = await call("GET", "/fal/schema");
    expect(c2.status).toBe(503);
    expect(q.m.ran).toHaveLength(2);
    // The worker answers: the pending job fills the cache.
    q.m.jobs.get("rp-2-e1")!.next = { status: "COMPLETED", output: { status: 200, body: { apps: [{ id: "minimax/h3-max" }] } } };
    expect((await call("GET", "/fal/schema")).j).toEqual({ apps: [{ id: "minimax/h3-max" }] });
    expect(q.m.ran).toHaveLength(2);
  });

  it("a stale entry is served and refreshed only while a worker is up", async () => {
    const row = await getRow(env, EID);
    q.m.onRun = () => ({ status: "COMPLETED", output: { status: 200, body: { v: 1 } } });
    await cachedGet(env, row, "/fal/schema", { who: "t" });
    await env.DB.prepare("UPDATE settings SET value = json_set(value, '$.at', 1) WHERE key LIKE 'slsc:%'").run();
    expect((await cachedGet(env, row, "/fal/schema", { who: "t" })).body).toEqual({ v: 1 });
    expect(q.m.ran).toHaveLength(1); // no worker up: no refresh
    q.m.workers.idle = 1;
    q.m.onRun = () => ({ status: "COMPLETED", output: { status: 200, body: { v: 2 } } });
    expect((await cachedGet(env, row, "/fal/schema", { who: "t" })).body).toEqual({ v: 1 });
    expect(q.m.ran).toHaveLength(2);
    expect((await cachedGet(env, row, "/fal/schema", { who: "t" })).body).toEqual({ v: 2 });
  });

  it("status: synthesised from Runpod's health and the cached capabilities, no job", async () => {
    q.m.workers.initializing = 1;
    const s = await call("GET", "/fv/v1/status");
    expect(s.j).toMatchObject({ object: "fv.status", state: "loading", pools: [{ id: "cons", kind: "runpod-serverless", state: "loading" }] });
    expect(q.m.ran).toEqual([]);
  });

  it("native: submit → the Runpod job id; polls from the store; the worker's final view; cancel through cancel.ts", async () => {
    const sub = await call("POST", "/fv/v1/jobs", { model: "fake-wan", task: "t2v", prompt: "a cat" });
    expect(sub.status).toBe(202);
    expect(sub.j).toMatchObject({ id: "rp-1-e1", object: "fv.job", status: "queued", model: "fake-wan" });
    expect(q.m.ran[0]).toEqual({ input: { kind: "http", method: "POST", path: "/fv/v1/jobs", headers: { "content-type": "application/json" }, body: { model: "fake-wan", task: "t2v", prompt: "a cat" }, wait: true, timeout_s: 600, cancel_path: "/fv/v1/jobs/{id}" }, policy: { executionTimeout: 600_000 } });
    const rec = await env.DB.prepare("SELECT route, status, actor FROM serverless_jobs WHERE job_id = 'rp-1-e1'").first();
    expect(rec).toEqual({ route: "console:native", status: "IN_QUEUE", actor: "owner" });
    expect((await call("GET", "/fv/v1/jobs/rp-1-e1")).j).toMatchObject({ id: "rp-1-e1", status: "queued" });
    // The worker took it: progress names the fv-serve job; the store has its progress.
    q.m.jobs.get("rp-1-e1")!.next = { status: "IN_PROGRESS", output: { state: "running", poll_path: "/fv/v1/jobs/fvjob_7" } };
    storeRow("fvjob_7", "native", "running", 0.42);
    expect((await call("GET", "/fv/v1/jobs/rp-1-e1")).j).toMatchObject({ id: "rp-1-e1", status: "running", progress: 0.42 });
    // Cancel: Runpod's cancel; the worker DELETEs cancel_path itself.
    const c = await call("DELETE", "/fv/v1/jobs/rp-1-e1");
    expect(c.j).toMatchObject({ id: "rp-1-e1", status: "cancelled", cancel_requested: true });
    expect(q.m.calls.some((x) => x.method === "POST" && x.path === `/v2/${EID}/cancel/rp-1-e1`)).toBe(true);
    // A second job runs to the end: the worker's own final view, id swapped.
    q.m.onRun = () => ({ status: "COMPLETED", output: { status: 200, headers: {}, body: { id: "fvjob_8", object: "fv.job", status: "succeeded", output: { url: "https://r2.example/v.mp4?X-Amz-Signature=s" } }, submit: { id: "fvjob_8" }, poll_path: "/fv/v1/jobs/fvjob_8" } });
    await call("POST", "/fv/v1/jobs", { model: "fake-wan", prompt: "b" });
    const done = await call("GET", "/fv/v1/jobs/rp-2-e1");
    expect(done.j).toEqual({ id: "rp-2-e1", object: "fv.job", status: "succeeded", output: { url: "https://r2.example/v.mp4?X-Amz-Signature=s" } });
    expect((await call("DELETE", "/fv/v1/jobs/rp-2-e1")).status).toBe(409);
    expect((await call("GET", "/fv/v1/jobs/rp-nope")).status).toBe(404);
  });

  it("a warm worker's refusal comes back as fv-serve's own reply", async () => {
    env.CONSOLE_SUBMIT_WAIT_MS = "200";
    q.m.workers.idle = 1;
    q.m.onRun = () => ({ status: "COMPLETED", output: { status: 422, headers: {}, body: { error: { kind: "invalid_request", message: "prompt is required" } } } });
    const r = await call("POST", "/fv/v1/jobs", { model: "fake-wan" });
    expect(r.status).toBe(422);
    expect(r.j.error.message).toBe("prompt is required");
  });

  it("fal: submit, IN_QUEUE / IN_PROGRESS with the store's logs, the result; cancel sends the app's PUT as a queue job", async () => {
    const sub = await call("POST", "/minimax/h3-max/text-to-video", { prompt: "a cat" });
    expect(sub.j).toMatchObject({ request_id: "rp-1-e1", status_url: `${P}/minimax/h3-max/requests/rp-1-e1/status` });
    expect(sub.headers.get("x-fal-request-id")).toBe("rp-1-e1");
    expect(q.m.ran[0].input.cancel_path).toBeUndefined();
    expect((await call("GET", "/minimax/h3-max/requests/rp-1-e1/status?logs=1")).j).toMatchObject({ status: "IN_QUEUE", queue_position: 0 });
    q.m.jobs.get("rp-1-e1")!.next = { status: "IN_PROGRESS", output: { state: "in_progress", poll_path: "/minimax/h3-max/requests/f-9/status" } };
    storeRow("f-9", "fal", "running", 0.3, [{ message: "step 3/10", level: "info", timestamp: "2026-10-07T01:02:03Z" }]);
    const st = await call("GET", "/minimax/h3-max/requests/rp-1-e1/status?logs=1");
    expect(st.j).toMatchObject({ status: "IN_PROGRESS", logs: [{ message: "step 3/10", level: "INFO" }] });
    expect((await call("GET", "/minimax/h3-max/requests/rp-1-e1")).status).toBe(400);
    q.m.workers.idle = 1;
    const c = await call("PUT", "/minimax/h3-max/requests/rp-1-e1/cancel");
    expect(c.status).toBe(202);
    expect(c.j.status).toBe("CANCELLATION_REQUESTED");
    expect(q.m.ran[1].input).toEqual({ kind: "http", method: "PUT", path: "/minimax/h3-max/requests/f-9/cancel" });
    // A finished one: the worker's result.
    q.m.onRun = () => ({ status: "COMPLETED", output: { status: 200, headers: {}, body: { video: { url: "https://r2.example/v.mp4" }, seed: 3 }, submit: { request_id: "f-10" } } });
    await call("POST", "/minimax/h3-max/text-to-video", { prompt: "b" });
    expect((await call("GET", "/minimax/h3-max/requests/rp-3-e1/status")).j.status).toBe("COMPLETED");
    expect((await call("GET", "/minimax/h3-max/requests/rp-3-e1")).j).toEqual({ video: { url: "https://r2.example/v.mp4" }, seed: 3 });
  });

  it("OpenAI content: fetched from the finished video's URL (same origin for the page)", async () => {
    q.m.onRun = () => ({ status: "COMPLETED", output: { status: 200, body: { id: "video_1", object: "video", status: "completed", url: "https://r2.example/v.mp4?sig" }, submit: { id: "video_1" } } });
    await call("POST", "/v1/videos", { model: "fake-wan", prompt: "c" });
    const c = await call("GET", "/v1/videos/rp-1-e1/content");
    expect(c.status).toBe(200);
    expect(c.text).toBe("MP4BYTES");
    expect(q.m.fetched).toEqual(["https://r2.example/v.mp4?sig"]);
  });

  it("guards: a scaled-down endpoint takes no submit; the balance floor; live and admin routes are off; read-only tokens", async () => {
    await env.DB.prepare("UPDATE serverless_endpoints SET status = 'scaled-down'").run();
    expect((await call("POST", "/fv/v1/jobs", { model: "m" })).status).toBe(409);
    await env.DB.prepare("UPDATE serverless_endpoints SET status = 'active'").run();
    // The balance below the floor + margin.
    vi.stubGlobal("fetch", async (u: any, i: any) => (String(u).includes("graphql") ? res(200, { data: { myself: { clientBalance: 9, endpoints: [] } } }) : q.fetchMock(u, i)));
    const f = await call("POST", "/fv/v1/jobs", { model: "m" });
    expect(f.status).toBe(402);
    expect(f.j.error.message).toMatch(/below/);
    expect((await call("POST", "/fv/v1/admin/keys", {})).status).toBe(404);
    expect((await call("GET", "/schema")).status).toBe(404);
    const row = await getRow(env, EID);
    const r = await consoleRequest(env, new Request(`${P}/fv/v1/jobs`, { method: "POST", body: "{}" }), row, { ep: EID, who, readOnly: true });
    expect(r.status).toBe(403);
    expect(q.m.ran).toEqual([]);
  });

  it("uploads: an fv-control PUT URL, then a signed public read URL the worker fetches", async () => {
    env.PUBLIC_URL = "https://fvc.example";
    const init = await call("POST", "/storage/upload/initiate?storage_type=fal-cdn-v3", { content_type: "image/png", file_name: "cat pic.png" });
    expect(init.j.upload_url).toMatch(new RegExp(`^${P}/storage/upload/put/`));
    expect(init.j.file_url).toMatch(/^https:\/\/fvc\.example\/serverless-uploads\/[^/]+\/cat_pic\.png$/);
    const put = await call("PUT", init.j.upload_url.slice(P.length), "PNGDATA", { "content-type": "image/png", "content-length": "7" });
    expect(put.status).toBe(200);
    const tok = init.j.file_url.split("/")[4];
    expect(await verifyUpload(env, tok, "get")).toMatchObject({ ct: "image/png" });
    expect(await verifyUpload(env, tok, "put")).toBeNull();
    expect(await verifyUpload(env, tok.replace(/.$/, (c: string) => (c === "A" ? "B" : "A")), "get")).toBeNull();
    const { uploadGet } = await import("../../src/serverless/console");
    const g = await uploadGet(env, tok);
    expect(await g.text()).toBe("PNGDATA");
    expect(g.headers.get("content-type")).toBe("image/png");
    const big = await call("PUT", init.j.upload_url.slice(P.length), "x", { "content-length": String(100 << 20) });
    expect(big.status).toBe(413);
  });
});
