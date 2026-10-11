// Integration test: the Worker under `wrangler dev` (workerd, local D1, R2,
// Durable Objects) against mocked Runpod / GHCR / GitHub / Cloudflare APIs
// and mocked worker pods and edge Worker (test/harness.mjs).
//   node test/integration/run.mjs            (npm run test:integration)
//   FVC_UI=1 …                               also keep it up for test/ui/smoke.mjs
import assert from "node:assert/strict";
import { d1Exec, fakeServe, hashPassphrase, jobInsert, JOBS_TABLE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.cloudriftKey = SECRETS.CLOUDRIFT_API_KEY;
mock.gmiKey = SECRETS.GMI_API_KEY;
mock.brevToken = SECRETS.BREV_API_TOKEN;
mock.brevOrg = "org-test-1";
const w = await startWorker(mock, {
  ...SECRETS,
  // GMI Cloud and NVIDIA Brev (docs/serve/deploy-gmi-brev.md): the owner's config, against the simulated APIs.
  GMI_PRODUCTS: "container.h200.x1,container.b200.x1",
  GMI_DEFAULT_IDC: "us-denver-1",
  GMI_BUDGET_USD: "50",
  BREV_ORG_ID: mock.brevOrg,
  BREV_INSTANCE_TYPES: "g5.xlarge-test,a100-test",
  BREV_PRICES: JSON.stringify({ "g5.xlarge-test": 1.0 }),
  BREV_BUDGET_USD: "2",
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  // The edge stand-in (test/harness.mjs `/edge/*`).
  EDGE_URL: `http://127.0.0.1:${mock.port}/edge`,
  EDGE_INTERNAL_TOKEN: mock.edgeInternal,
  EDGE_ADMIN_TOKEN: mock.edgeAdmin,
  EDGE_D1_DATABASE_ID: "d1-edge-staging",
  // The serverless console: no wait for a warm worker's own submit reply; quick polls.
  CONSOLE_SUBMIT_WAIT_MS: "0",
  CONSOLE_POLL_MS: "100",
  CONSOLE_CACHE_WAIT_MS: "5000",
});
const B = w.url;
const bodies = []; // every response body, checked for secrets at the end
let passed = 0;
const t0 = Date.now();

async function call(path, { method = "GET", body, headers = {}, raw = false } = {}) {
  const r = await fetch(B + path, { method, headers: { ...(body !== undefined ? { "content-type": "application/json" } : {}), ...headers }, body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body) });
  const text = await r.text();
  bodies.push(text);
  let j = null;
  try { j = JSON.parse(text); } catch {}
  return raw ? { status: r.status, text, headers: r.headers } : { status: r.status, j, headers: r.headers };
}
// FVC_ONLY=<regex>: only the steps matching it (plus login and tokens), for iterating on one.
const ONLY = process.env.FVC_ONLY ? new RegExp(process.env.FVC_ONLY) : null;
async function step(name, fn) {
  if (ONLY && !ONLY.test(name) && !/^(public health|login|API tokens)/.test(name)) return;
  const t = Date.now();
  try {
    await fn();
    passed++;
    console.log(`ok   ${name} (${Date.now() - t} ms)`);
  } catch (e) {
    console.log(`FAIL ${name}: ${e.stack || e.message}`);
    console.log(w.output().slice(-4000));
    w.stop();
    mock.close();
    process.exit(1);
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/** After `wrangler d1 execute` on another local D1, workerd can drop the next connection: wait until it answers twice. */
async function settle() {
  let ok = 0;
  for (let i = 0; i < 40 && ok < 2; i++) {
    ok = (await fetch(`${B}/healthz`).then((r) => r.ok).catch(() => false)) ? ok + 1 : 0;
    await sleep(250);
  }
}
let cookie = "";
let csrf = "";
let token = "";
const T = () => ({ authorization: `Bearer ${token}` });
async function waitOp(cluster, kind, timeoutMs = 60000) {
  const t = Date.now();
  while (Date.now() - t < timeoutMs) {
    const r = await call(`/api/clusters/${cluster}/ops`, { headers: T() });
    const op = r.j.operations.find((o) => o.kind === kind && !/is running/.test(o.error || ""));
    if (op && op.status !== "running") return op;
    await sleep(500);
  }
  throw new Error(`operation ${kind} did not finish`);
}

await step("public health; the API needs auth", async () => {
  assert.equal((await call("/healthz")).j.auth, "passphrase");
  assert.equal((await call("/api/overview")).status, 401);
  assert.equal((await call("/api/auth/me")).j.authenticated, false);
  assert.equal((await call("/api/clusters", { method: "POST", body: {} })).status, 401);
});

await step("login: a session cookie, CSRF on mutations, Origin check", async () => {
  assert.equal((await call("/api/auth/login", { method: "POST", body: { passphrase: "wrong" } })).status, 401);
  const r = await fetch(B + "/api/auth/login", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ passphrase: PASSPHRASE }) });
  assert.equal(r.status, 200);
  const sc = r.headers.get("set-cookie");
  assert.match(sc, /__Host-fvc_session=.*HttpOnly; Secure; SameSite=Strict/);
  cookie = sc.split(";")[0];
  csrf = (await r.json()).csrf;
  const me = await call("/api/auth/me", { headers: { cookie } });
  assert.equal(me.j.authenticated, true);
  assert.equal((await call("/api/overview", { headers: { cookie } })).status, 200);
  assert.equal((await call("/api/policies", { method: "PUT", body: {}, headers: { cookie } })).status, 403, "no CSRF token");
  assert.equal((await call("/api/policies", { method: "PUT", body: {}, headers: { cookie, "x-csrf-token": "nope" } })).status, 403);
  assert.equal((await call("/api/policies", { method: "PUT", body: {}, headers: { cookie, "x-csrf-token": csrf, origin: "https://evil.example" } })).status, 403);
  assert.equal((await call("/api/policies", { method: "PUT", body: { idle_min: 20 }, headers: { cookie, "x-csrf-token": csrf } })).status, 200);
  // A forged cookie is refused.
  assert.equal((await call("/api/overview", { headers: { cookie: cookie.replace(/.$/, (c) => (c === "A" ? "B" : "A")) } })).status, 401);
});

await step("API tokens: admin and read scopes", async () => {
  const r = await call("/api/tokens", { method: "POST", body: { name: "ci", scope: "admin" }, headers: { cookie, "x-csrf-token": csrf } });
  assert.equal(r.status, 201);
  token = r.j.token;
  assert.match(token, /^fvc_[0-9a-f]{64}$/);
  const ro = (await call("/api/tokens", { method: "POST", body: { name: "viewer", scope: "read" }, headers: { cookie, "x-csrf-token": csrf } })).j.token;
  assert.equal((await call("/api/clusters", { headers: { authorization: `Bearer ${ro}` } })).status, 200);
  assert.equal((await call("/api/clusters", { method: "POST", body: { name: "x" }, headers: { authorization: `Bearer ${ro}` } })).status, 403);
  assert.equal((await call("/api/tokens", { method: "POST", body: { name: "y" }, headers: T() })).status, 403, "tokens cannot mint tokens");
  assert.equal((await call("/api/clusters", { headers: { authorization: "Bearer fvc_bogus" } })).status, 401);
  const list = await call("/api/tokens", { headers: T() });
  assert.ok(!JSON.stringify(list.j).includes(token));
});

let cid = "";
await step("define a tiny CPU cluster; price check", async () => {
  const r = await call("/api/clusters", { method: "POST", body: { spec: { name: "tiny", template: "tiny-cpu", image: { channel: "stable" }, cap_s: 7200 } }, headers: T() });
  assert.equal(r.status, 201, JSON.stringify(r.j));
  cid = r.j.cluster.id;
  assert.equal((await call("/api/clusters", { method: "POST", body: { spec: { name: "tiny", template: "tiny-cpu" } }, headers: T() })).status, 409);
  assert.equal((await call("/api/clusters", { method: "POST", body: { spec: { name: "Bad!" } }, headers: T() })).status, 400);
  const p = await call(`/api/clusters/${cid}/price`, { method: "POST", body: {}, headers: T() });
  assert.equal(p.j.ok, true);
  assert.equal(p.j.pods.length, 1);
  assert.ok(p.j.cluster_dph < 0.5);
});

await step("the balance floor refuses a start", async () => {
  mock.balance = 9;
  const p = await call(`/api/clusters/${cid}/price`, { method: "POST", body: {}, headers: T() });
  assert.equal(p.j.ok, false);
  assert.match(p.j.reasons.join(" "), /start minimum|floor/);
  const s = await call(`/api/clusters/${cid}/start`, { method: "POST", body: {}, headers: T() });
  assert.equal(s.status, 202);
  const op = await waitOp(cid, "up");
  assert.equal(op.status, "failed");
  assert.equal(mock.pods.size, 0, "nothing created");
  mock.balance = 50;
});

let worker;
await step("start: one front worker behind the edge; registered; ready", async () => {
  await d1Exec(w.dir, "DELETE FROM operations");
  const s = await call(`/api/clusters/${cid}/start`, { method: "POST", body: {}, headers: T() });
  assert.equal(s.status, 202);
  assert.equal((await call(`/api/clusters/${cid}/start`, { method: "POST", body: {}, headers: T() })).status, 409, "one operation at a time");
  const op = await waitOp(cid, "up");
  assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-5)));
  assert.ok(op.log.some((l) => /ready = a ready front at the edge\) ready 1\/1: fake/.test(l.msg)), JSON.stringify(op.log.slice(-4)));
  const pods = [...mock.pods.values()];
  assert.equal(pods.length, 1, "no gateway pod");
  worker = pods[0];
  const edge = `http://127.0.0.1:${mock.port}/edge`;
  // The worker: CPU, the boot command (watchdog), fake engine config inline, a front behind the edge.
  assert.equal(worker.payload.computeType, "CPU");
  assert.match(worker.payload.dockerStartCmd[0], /\[watchdog\]/);
  assert.equal(worker.env.FV_SERVE_ROLE, "worker");
  assert.equal(worker.env.FV_DISPATCH_FRONT, "1");
  assert.equal(worker.env.FV_INTERNAL_TOKEN, mock.edgeInternal);
  assert.equal(worker.env.FV_PUBLIC_BASE_URL, edge);
  assert.equal(worker.env.FV_ADMIN_TOKEN, undefined);
  assert.equal(worker.env.FV_GATEWAY_TOML_B64, undefined);
  assert.equal(worker.env.FV_GITHUB_TOKEN, undefined);
  assert.match(worker.env.FV_LOG_SHIP_URL, /\/ingest\/v1\/logs$/);
  assert.ok(Number(worker.env.FV_CLUSTER_DEADLINE) > Date.now() / 1000);
  assert.match(Buffer.from(worker.env.FV_WORKER_TOML_B64, "base64").toString(), /backend = "fake"/);
  assert.match(worker.image, /@sha256:/);
  const c = await call(`/api/clusters/${cid}`, { headers: T() });
  assert.equal(c.j.cluster.status, "running");
  assert.equal(c.j.pods.filter((p) => p.status === "ready").length, 1);
});

await step("jobs: the Jobs view from the edge's D1, cancel by id at the worker that holds it, cancel all queued; admin only", async () => {
  const now = Date.now();
  const pid = worker.id;
  const U = (n) => `0000000${n}-aaaa-4bbb-8ccc-dddddddddddd`;
  for (const db of ["fv-edge", "fv-jobs"]) d1Exec(w.dir, JOBS_TABLE, db);
  d1Exec(w.dir, [
    jobInsert({ id: U(1), ext: "fvjob_run1", status: "running", worker: pid, at: now - 60_000, started: new Date(now - 50_000).toISOString(), progress: 0.4 }),
    jobInsert({ id: U(2), ext: "video_gen_q1", api: "openai_videos", status: "queued", worker: null, at: now - 15_000 }),
    jobInsert({ id: U(3), ext: "fvjob_q2", status: "queued", worker: pid, at: now - 5_000 }),
    jobInsert({ id: U(4), ext: "fvjob_done", status: "succeeded", worker: pid, at: now - 120_000 }),
    jobInsert({ id: U(5), ext: "fvjob_foreign", status: "running", worker: "foreignpod0001", at: now - 10_000 }),
  ].join(";\n"), "fv-edge");
  await settle();
  const ro = (await call("/api/tokens", { method: "POST", body: { name: "jobs-viewer", scope: "read" }, headers: { cookie, "x-csrf-token": csrf } })).j.token;
  const v = await call(`/api/clusters/${cid}/jobs`, { headers: { authorization: `Bearer ${ro}` } });
  assert.equal(v.status, 200, JSON.stringify(v.j));
  assert.equal(v.j.source, "edge");
  assert.deepEqual(v.j.jobs.map((j) => j.external_id), ["fvjob_q2", "video_gen_q1", "fvjob_run1", "fvjob_done"]);
  assert.deepEqual(v.j.counts, { queued: 2, running: 1, succeeded: 1 });
  const run = v.j.jobs.find((j) => j.external_id === "fvjob_run1");
  assert.equal(run.pool, "fake");
  assert.equal(run.api, "native");
  assert.ok(run.started_at > now - 60_000, "started time from the job record");
  assert.equal(v.j.jobs.find((j) => j.external_id === "video_gen_q1").worker, null, "queued at the edge");
  assert.ok(!JSON.stringify(v.j).includes("SIGNED_URL_SECRET"), "the job record (signed URLs) never leaves fv-control");
  assert.deepEqual((await call(`/api/clusters/${cid}/jobs?status=queued`, { headers: T() })).j.jobs.map((j) => j.external_id), ["fvjob_q2", "video_gen_q1"]);
  const bad = await call(`/api/clusters/${cid}/jobs?status=queued,stuck`, { headers: T() });
  assert.equal(bad.status, 400);
  assert.match(bad.j.error, /statuses: queued, running/);
  // Cancel by id: admin only, CSRF on a cookie session.
  assert.equal((await call("/api/jobs/fvjob_run1/cancel", { method: "POST", body: {}, headers: { authorization: `Bearer ${ro}` } })).status, 403);
  assert.equal((await call("/api/jobs/fvjob_run1/cancel", { method: "POST", body: {}, headers: { cookie } })).status, 403);
  mock.internalJobs[U(1)] = [200, { id: U(1), status: "running", cancel_requested: true }];
  const c1 = await call("/api/jobs/fvjob_run1/cancel", { method: "POST", body: {}, headers: { cookie, "x-csrf-token": csrf } });
  assert.equal(c1.status, 200, JSON.stringify(c1.j));
  assert.equal(c1.j.via, "internal");
  assert.equal(c1.j.pod, pid);
  assert.equal(c1.j.cancel_requested, true);
  assert.deepEqual(mock.internalCancels.at(-1), { pod: pid, id: U(1) }, "straight to the worker that holds it, by the internal id");
  assert.equal((await call("/api/jobs/fvjob_done/cancel", { method: "POST", body: {}, headers: T() })).j.note, "already succeeded: nothing to cancel");
  const foreign = await call("/api/jobs/fvjob_foreign/cancel", { method: "POST", body: {}, headers: T() });
  assert.equal(foreign.status, 403);
  assert.match(foreign.j.error, /did not create/);
  assert.equal((await call("/api/jobs/fvjob_nope/cancel", { method: "POST", body: {}, headers: T() })).status, 404);
  assert.equal((await call(`/api/jobs/${encodeURIComponent("bad id")}/cancel`, { method: "POST", body: {}, headers: T() })).status, 400);
  // A worker that does not know the job and no other route: a readable 502.
  const miss = await call("/api/jobs/fvjob_q2/cancel", { method: "POST", body: {}, headers: T() });
  assert.equal(miss.status, 502);
  assert.match(miss.j.error, /fvjob_q2 \(native, tiny\): internal .*404 job `.*` was not found/);
  // Cancel all queued: the one on the worker and the one still queued at the edge (through the front).
  mock.internalJobs[U(2)] = [200, { id: U(2), status: "cancelled" }];
  mock.internalJobs[U(3)] = [200, { id: U(3), status: "cancelled" }];
  const cq = await call(`/api/clusters/${cid}/jobs/cancel-queued`, { method: "POST", body: { max: 10 }, headers: T() });
  assert.equal(cq.status, 200, JSON.stringify(cq.j));
  assert.equal(cq.j.queued, 2);
  assert.equal(cq.j.cancelled, 2);
  assert.ok(mock.internalCancels.some((x) => x.id === U(2) && x.pod === pid), "the edge-queued job went through a live front");
  assert.equal((await call(`/api/clusters/${cid}/jobs/cancel-queued`, { method: "POST", body: { max: 0 }, headers: T() })).status, 400);
  const one = await call(`/api/jobs/${U(1)}`, { headers: T() });
  assert.equal(one.j.job.external_id, "fvjob_run1");
  const a = (await call("/api/audit?limit=50", { headers: T() })).j.audit;
  assert.ok(a.some((x) => x.action === "job.cancel" && x.target === "tiny" && x.ok === 1));
  assert.ok(a.some((x) => x.action === "jobs.cancel-queued"));
});

await step("serverless: cancel a pasted job id and a fire-and-forget submit's fv-serve job; purge with the name and the queued count", async () => {
  const now = Date.now();
  const spec = JSON.stringify({ name: "qmock", mode: "queue", variant: "cpu", compute: "CPU", workers_min: 0, workers_max: 1 });
  d1Exec(w.dir, `INSERT INTO serverless_endpoints (id, name, endpoint_id, template_id, mode, spec, status, created_at, created_by, updated_at, deleted_at) VALUES ('se_qmock', 'qmock', 'rpqmock000001', 'tpl', 'queue', '${spec}', 'active', ${now}, 'owner', ${now}, NULL)`);
  const q = mock.queue;
  q.jobs.set("rp-queued-1", { id: "rp-queued-1", status: "IN_QUEUE" });
  q.jobs.set("rp-ff-1", { id: "rp-ff-1", status: "COMPLETED", output: { status: 202, body: { id: "fvjob_ff", status: "queued" } } });
  q.queued = 4;
  q.workers = 1;
  await settle();
  d1Exec(w.dir, `INSERT INTO serverless_jobs (endpoint, job_id, route, status, cold, submitted_at, finished_at, input, output, actor) VALUES ('se_qmock', 'rp-ff-1', 'run', 'COMPLETED', 0, ${now}, ${now}, '{"kind":"http","method":"POST","path":"/fv/v1/jobs","body":{"model":"fake-wan"}}', '{"status":202,"body":{"id":"fvjob_ff","status":"queued"}}', 'owner')`);
  await settle();
  const ro = (await call("/api/tokens", { method: "POST", body: { name: "sls-viewer", scope: "read" }, headers: { cookie, "x-csrf-token": csrf } })).j.token;
  assert.equal((await call("/api/serverless/qmock/jobs/rp-queued-1/cancel", { method: "POST", body: {}, headers: { authorization: `Bearer ${ro}` } })).status, 403);
  const c1 = await call("/api/serverless/qmock/jobs/rp-queued-1/cancel", { method: "POST", body: {}, headers: { cookie, "x-csrf-token": csrf } });
  assert.equal(c1.status, 200, JSON.stringify(c1.j));
  assert.equal(c1.j.before, "IN_QUEUE");
  assert.equal(c1.j.status, "CANCELLED");
  // Its status polls like an invoke's.
  assert.equal((await call(`/api/serverless/qmock/jobs/${c1.j.job}`, { headers: T() })).j.job.status, "CANCELLED");
  const c2 = await call("/api/serverless/qmock/jobs/rp-ff-1/cancel", { method: "POST", body: {}, headers: T() });
  assert.equal(c2.status, 200, JSON.stringify(c2.j));
  assert.equal(c2.j.fv_cancel.sent, true);
  assert.equal(c2.j.fv_cancel.route, "DELETE /fv/v1/jobs/fvjob_ff");
  assert.deepEqual(q.ran.at(-1).input, { kind: "http", method: "DELETE", path: "/fv/v1/jobs/fvjob_ff" });
  const bad = await call("/api/serverless/qmock/jobs/rp-nope/cancel", { method: "POST", body: {}, headers: T() });
  assert.equal(bad.status, 404);
  assert.match(bad.j.error, /Runpod has no job rp-nope on qmock/);
  assert.equal((await call("/api/serverless/qmock/jobs/rp-nope/cancel", { method: "POST", body: { fv_api: "ltx" }, headers: T() })).status, 400);
  // Purge: the queue first, the name to confirm, the count seen.
  const qq = await call("/api/serverless/qmock/queue", { headers: T() });
  assert.equal(qq.j.queued, 3);
  const wrong = await call("/api/serverless/qmock/purge", { method: "POST", body: { confirm: "nope" }, headers: T() });
  assert.equal(wrong.status, 400);
  assert.match(wrong.j.error, /type the endpoint's name \(qmock\)/);
  assert.equal((await call("/api/serverless/qmock/purge", { method: "POST", body: { confirm: "qmock", expected: 1 }, headers: T() })).status, 409);
  assert.equal(q.purges, 0);
  const pg = await call("/api/serverless/qmock/purge", { method: "POST", body: { confirm: "qmock", expected: 3 }, headers: T() });
  assert.equal(pg.status, 200, JSON.stringify(pg.j));
  assert.equal(pg.j.removed, 3);
  assert.equal(pg.j.queued_after, 0);
  const a = (await call("/api/audit?limit=50", { headers: T() })).j.audit;
  assert.ok(a.some((x) => x.action === "serverless.purge" && x.target === "qmock"));
  assert.ok(a.filter((x) => x.action === "serverless.cancel").length >= 2);
});

await step("serverless presets: an h3-ref2v endpoint (queue and lb) carries the ref2v config inline; what it serves, its examples; a 5090 and a custom config not in the image are refused", async () => {
  const ref2v = new URL("../../../configs/serve/runpod-h3-ref2v.toml", import.meta.url);
  const { readFileSync } = await import("node:fs");
  const want = readFileSync(ref2v, "utf8");
  // The catalog: every preset with what it serves.
  const cat = await call("/api/serverless/presets", { headers: T() });
  assert.equal(cat.status, 200);
  const pr = cat.j.presets.find((p) => p.id === "h3-ref2v");
  assert.deepEqual(pr.serves.models.map((m) => m.id), ["h3-ref2v-turbo", "h3-ref2v-max"]);
  assert.ok(pr.inline_config && pr.min_vram_gb === 80);
  assert.ok(cat.j.presets.some((p) => p.id === "cpu"));
  // The validator: field-level refusals, and what it serves (live in the form).
  const v5090 = await call("/api/serverless/validate", { method: "POST", body: { spec: { name: "r2v-x", preset: "h3-ref2v", gpu_types: ["NVIDIA GeForce RTX 5090"] } }, headers: T() });
  assert.equal(v5090.j.ok, false);
  assert.ok(v5090.j.issues.some((i) => i.path.join(".") === "gpu_types.0" && /32 GB.*at least 80 GB/.test(i.message)), JSON.stringify(v5090.j.issues));
  const vimg = await call("/api/serverless/validate", { method: "POST", body: { spec: { name: "r2v-x", variant: "h3-max", config: "/etc/fv/runpod-h3-ref2v.toml" } }, headers: T() });
  assert.ok(vimg.j.issues.some((i) => i.path[0] === "config" && /not in it \(preset h3-ref2v sends that config inline\)/.test(i.message)), JSON.stringify(vimg.j.issues));
  const vok = await call("/api/serverless/validate", { method: "POST", body: { spec: { name: "r2v-x", preset: "h3-ref2v", mode: "lb" } }, headers: T() });
  assert.equal(vok.j.ok, true, JSON.stringify(vok.j));
  assert.ok(vok.j.serving.serves.aliases.some((a) => a.name === "MiniMax-H3-Turbo" && a.model === "h3-ref2v-turbo"));
  // The incident's spec (h3-max, no config): the preset is inferred, and MiniMax-H3-Turbo is not among its names.
  const vmax = await call("/api/serverless/validate", { method: "POST", body: { spec: { name: "r2v-x", variant: "h3-max", mode: "lb" } }, headers: T() });
  assert.equal(vmax.j.serving.preset, "h3-max");
  assert.equal(vmax.j.serving.preset_inferred, true);
  assert.ok(!vmax.j.serving.serves.aliases.some((a) => a.name === "MiniMax-H3-Turbo"));
  // Create: queue (REST v1 template + endpoint) and lb (REST v2, then the template gets the boot).
  const tplOf = (id) => [...mock.sls.templates.values()].find((t) => t.id === id);
  const q = await call("/api/serverless", { method: "POST", body: { spec: { name: "r2v-q", preset: "h3-ref2v" } }, headers: T() });
  assert.equal(q.status, 201, JSON.stringify(q.j));
  assert.equal(q.j.endpoint.spec.preset, "h3-ref2v");
  const qt = tplOf(q.j.endpoint.template_id);
  assert.equal(Buffer.from(qt.env.FV_WORKER_TOML_B64, "base64").toString(), want, "the queue template carries the ref2v config");
  assert.deepEqual(qt.dockerEntrypoint.slice(0, 2), ["/bin/sh", "-c"]);
  assert.equal(qt.env.FV_SERVE_MODE, "runpod-queue");
  const lb = await call("/api/serverless", { method: "POST", body: { spec: { name: "r2v-lb", preset: "h3-ref2v", mode: "lb" } }, headers: T() });
  assert.equal(lb.status, 201, JSON.stringify(lb.j));
  const lt = tplOf(lb.j.endpoint.template_id);
  assert.equal(Buffer.from(lt.env.FV_WORKER_TOML_B64, "base64").toString(), want, "the lb template carries the ref2v config");
  assert.ok(lt.patches >= 1 && lt.dockerEntrypoint[2].includes("FV_WORKER_TOML_B64"), "the lb template got the boot that decodes it");
  assert.equal(lt.env.FV_SERVE_MODE, "http");
  // The endpoint page: what it serves and ready-made invokes in the endpoint's mode.
  const page = await call(`/api/serverless/${lb.j.endpoint.id}`, { headers: T() });
  const ex = page.j.serving.examples.find((e) => e.api === "minimax" && e.model === "h3-ref2v-turbo");
  assert.equal(ex.invoke.path, "/v2/video_generation");
  assert.equal(ex.invoke.body.model, "MiniMax-H3-Turbo");
  assert.deepEqual(ex.needs, ["image_url"]);
  const qpage = await call(`/api/serverless/${q.j.endpoint.id}/serves`, { headers: T() });
  assert.equal(qpage.j.examples.find((e) => e.api === "minimax" && e.model === "h3-ref2v-turbo").invoke.input.kind, "http");
  // A preset switch on update drops the old preset's config; a 5090 on update is refused.
  assert.equal((await call(`/api/serverless/${q.j.endpoint.id}`, { method: "PUT", body: { spec: { gpu_types: ["NVIDIA GeForce RTX 5090"] } }, headers: T() })).status, 400);
  const sw = await call(`/api/serverless/${q.j.endpoint.id}`, { method: "PUT", body: { spec: { preset: "h3-max" } }, headers: T() });
  assert.equal(sw.status, 200, JSON.stringify(sw.j));
  assert.equal(sw.j.endpoint.spec.config_toml, undefined);
  assert.equal(tplOf(q.j.endpoint.template_id).env.FV_WORKER_TOML_B64, undefined);
  // The CLI: the preset catalog.
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const dir = mkdtempSync(`${tmpdir()}/fvc-cli-`);
  writeFileSync(`${dir}/token`, token, { mode: 0o600 });
  const cli = async (...args) => (await promisify(execFile)("bash", [new URL("../../../scripts/serve/fv-control.sh", import.meta.url).pathname, ...args], { env: { ...process.env, FV_CONTROL_URL: B, FV_CONTROL_TOKEN_FILE: `${dir}/token`, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" } })).stdout;
  assert.match(await cli("endpoint", "presets"), /^h3-ref2v\th3-max \+ inline config\t>= 80 GB\tweights: h3-base,h3-ref2va\n  serves: h3-ref2v-turbo \[ref2v\]/m);
  assert.match(await cli("endpoint", "serves", "r2v-lb"), /"MiniMax-H3-Turbo"/);
  for (const e of [q.j.endpoint, lb.j.endpoint]) assert.equal((await call(`/api/serverless/${e.id}`, { method: "DELETE", headers: T() })).status, 200);
});

await step("serverless console: pages behind the login, cached capabilities, synthesised status, fal and native jobs as queue jobs, store-backed polls, cancel, uploads", async () => {
  const now = Date.now();
  const EID = "rpcons0000001";
  const spec = JSON.stringify({ name: "cons", mode: "queue", variant: "cpu", compute: "CPU", workers_min: 0, workers_max: 1, execution_timeout_s: 600 });
  d1Exec(w.dir, `INSERT INTO serverless_endpoints (id, name, endpoint_id, template_id, mode, spec, image, status, created_at, created_by, updated_at) VALUES ('se_cons', 'cons', '${EID}', 'tpl', 'queue', '${spec}', 'ghcr.io/x/serve@sha256:0123456789abcdef', 'active', ${now}, 'owner', ${now})`);
  d1Exec(w.dir, JOBS_TABLE, "fv-jobs");
  await settle();
  const q = mock.queue;
  q.onRun = fakeServe(mock, { steps: 1 });
  q.workers = 1;
  const ran0 = q.ran.length;
  const S = `/serverless/${EID}`;
  const own = { cookie, origin: B };
  // Login first: a page goes to the dashboard's login, an API call is refused.
  const anon = await fetch(`${B}${S}/console`, { redirect: "manual" });
  assert.equal(anon.status, 302);
  assert.match(anon.headers.get("location"), /#\/serverless$/);
  assert.equal((await call(`${S}/fv/v1/status`)).status, 401);
  assert.equal((await call("/serverless/nope-endpoint/console", { headers: own })).status, 404);
  // The page: fv-serve's console under the prefix (its CSP, the embedding tags), the assets unchanged.
  const page = await call(`${S}/console`, { headers: own, raw: true });
  assert.equal(page.status, 200);
  assert.match(page.headers.get("content-security-policy"), /script-src 'self'/);
  assert.ok(page.text.includes(`<meta name="fv-console-base" content="${S}">`));
  assert.ok(page.text.includes(`src="${S}/console/assets/home.js"`));
  const js = await call(`${S}/console/assets/common.js`, { headers: own, raw: true });
  assert.equal(js.headers.get("content-type"), "text/javascript; charset=utf-8");
  assert.equal((await call(`${S}/console/stream`, { headers: own, raw: true })).status, 404);
  // Capabilities: the page load started the cache jobs; the call finds them (no extra job), then the cache.
  const caps = await call(`${S}/fv/v1/capabilities`, { headers: own });
  assert.equal(caps.status, 200, JSON.stringify(caps.j));
  assert.equal(caps.j.auth.mode, "none");
  assert.equal(caps.j.models[0].caps.id, "fake-wan");
  const cat = await call(`${S}/fal/schema`, { headers: own });
  assert.equal(cat.j.apps[0].id, "fastvideo/fake-wan");
  const capsJobs = () => q.ran.slice(ran0).filter((r) => r.input.method === "GET" && r.input.path === "/fv/v1/capabilities").length;
  assert.equal(capsJobs(), 1);
  assert.match((await call(`${S}/fv/v1/capabilities`, { headers: own })).headers.get("x-fv-console-cache"), /^cache; age=/);
  assert.equal(capsJobs(), 1);
  // Status: one pool from Runpod's health, the model names from the cached capabilities.
  const st = await call(`${S}/fv/v1/status`, { headers: own });
  assert.equal(st.j.object, "fv.status");
  assert.equal(st.j.pools[0].kind, "runpod-serverless");
  assert.equal(st.j.state, "ready");
  assert.equal(st.j.names["fake-wan-alias"], "fake-wan");
  q.workers = 0;
  assert.equal((await call(`${S}/fv/v1/status`, { headers: own })).j.state, "scaled_to_zero");
  // fal: a submit needs this origin (the console sends no CSRF token).
  const falBody = { prompt: "a red fox", seed: 7 };
  assert.equal((await call(`${S}/fastvideo/fake-wan/text-to-video`, { method: "POST", body: falBody, headers: { cookie } })).status, 403);
  assert.equal((await call(`${S}/fastvideo/fake-wan/text-to-video`, { method: "POST", body: falBody, headers: { cookie, origin: "https://evil.example" } })).status, 403);
  const sub = await call(`${S}/fastvideo/fake-wan/text-to-video`, { method: "POST", body: falBody, headers: own });
  assert.equal(sub.status, 200, JSON.stringify(sub.j));
  const rid = sub.j.request_id;
  assert.match(rid, /^mock-run-/);
  assert.equal(sub.j.status_url, `${B}${S}/fastvideo/fake-wan/requests/${rid}/status`);
  const env = q.ran.at(-1);
  assert.deepEqual(env.input, { kind: "http", method: "POST", path: "/fastvideo/fake-wan/text-to-video", headers: { "content-type": "application/json" }, body: falBody, wait: true, timeout_s: 600 });
  assert.equal(env.policy.executionTimeout, 600_000);
  // In flight: the worker took it (its progress names the fal request); the shared job store has its logs.
  const s1 = await call(`${S}/fastvideo/fake-wan/requests/${rid}/status?logs=1`, { headers: own });
  assert.equal(s1.j.status, "IN_QUEUE");
  d1Exec(w.dir, jobInsert({ id: "00000000-cccc-4bbb-8ccc-000000000001", ext: "f-sim-1", api: "fal", status: "running", at: now, progress: 0.5 }), "fv-jobs");
  await settle();
  const s2 = await call(`${S}/fastvideo/fake-wan/requests/${rid}/status?logs=1`, { headers: own });
  assert.equal(s2.j.status, "IN_PROGRESS", JSON.stringify(s2.j));
  const s3 = await call(`${S}/fastvideo/fake-wan/requests/${rid}/status?logs=1`, { headers: own });
  assert.equal(s3.j.status, "COMPLETED");
  const result = await call(`${S}/fastvideo/fake-wan/requests/${rid}`, { headers: own });
  assert.equal(result.status, 200);
  assert.equal(result.j.video.url, `http://127.0.0.1:${mock.port}/media/f-sim-1.mp4`);
  assert.equal(result.j.seed, 42);
  // The endpoint page lists it as a console job.
  const jobs = (await call("/api/serverless/cons/jobs", { headers: T() })).j.jobs;
  assert.ok(jobs.some((j) => j.job_id === rid && j.route === "console:fal" && j.status === "COMPLETED"));
  // Native: submit, poll to the worker's own final view; a refused one; cancel.
  const nat = await call(`${S}/fv/v1/jobs`, { method: "POST", body: { model: "fake-wan", task: "t2v", prompt: "a cat" }, headers: own });
  assert.equal(nat.status, 202);
  assert.equal(nat.j.status, "queued");
  let view = null;
  for (let i = 0; i < 6 && view?.status !== "succeeded"; i++) view = (await call(`${S}/fv/v1/jobs/${nat.j.id}`, { headers: own })).j;
  assert.equal(view.status, "succeeded");
  assert.equal(view.id, nat.j.id);
  assert.match(view.output.url, /\/media\/fvjob_sim/);
  const refused = await call(`${S}/fv/v1/jobs`, { method: "POST", body: { model: "fake-wan" }, headers: own });
  const rv = (await call(`${S}/fv/v1/jobs/${refused.j.id}`, { headers: own })).j;
  assert.equal(rv.status, "failed");
  assert.equal(rv.error.message, "prompt is required");
  q.onRun = fakeServe(mock, { steps: 50 });
  const long = await call(`${S}/fv/v1/jobs`, { method: "POST", body: { model: "fake-wan", prompt: "long" }, headers: own });
  assert.equal((await call(`${S}/fv/v1/jobs/${long.j.id}`, { headers: own })).j.status, "queued");
  const del = await call(`${S}/fv/v1/jobs/${long.j.id}`, { method: "DELETE", headers: own });
  assert.equal(del.status, 200, JSON.stringify(del.j));
  assert.equal(del.j.status, "cancelled");
  assert.equal((await call(`${S}/fv/v1/jobs/${long.j.id}`, { headers: own })).j.status, "cancelled");
  // Uploads: an fv-control PUT URL; the worker reads it back through a signed public URL.
  const init = await call(`${S}/storage/upload/initiate?storage_type=fal-cdn-v3`, { method: "POST", body: { content_type: "image/png", file_name: "cat.png" }, headers: own });
  assert.equal(init.status, 200, JSON.stringify(init.j));
  const put = await fetch(init.j.upload_url, { method: "PUT", headers: { "content-type": "image/png", cookie, origin: B }, body: "PNGBYTES" });
  assert.equal(put.status, 200);
  const got = await fetch(init.j.file_url);
  assert.equal(got.status, 200);
  assert.equal(got.headers.get("content-type"), "image/png");
  assert.equal(await got.text(), "PNGBYTES");
  assert.equal((await fetch(init.j.file_url.replace(/\/serverless-uploads\/[^.]+/, "/serverless-uploads/eyJ4IjoxfQ"))).status, 403);
  // A read-only token sees the status but cannot submit; live and admin routes are off.
  const ro = (await call("/api/tokens", { method: "POST", body: { name: "console-viewer", scope: "read" }, headers: { cookie, "x-csrf-token": csrf } })).j.token;
  assert.equal((await call(`${S}/fv/v1/status`, { headers: { authorization: `Bearer ${ro}` } })).status, 200);
  assert.equal((await call(`${S}/fv/v1/jobs`, { method: "POST", body: { prompt: "x" }, headers: { authorization: `Bearer ${ro}` } })).status, 403);
  assert.equal((await call(`${S}/fv/v1/admin/keys`, { headers: own })).status, 404);
  // The endpoint page's "Refresh console cache".
  const cc = await call("/api/serverless/cons/console-cache", { method: "DELETE", headers: T() });
  assert.equal(cc.status, 200);
  assert.ok(cc.j.dropped >= 2);
  q.onRun = undefined;
  const a = (await call("/api/audit?limit=80", { headers: T() })).j.audit;
  assert.ok(a.some((x) => x.action === "serverless.console" && x.target === "cons"));
});

await step("CLI: fv-control.sh jobs, job-cancel, jobs-cancel-queued, endpoint cancel and purge", async () => {
  // Async: the mocks run in this process, so a blocking exec would starve them.
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const dir = mkdtempSync(`${tmpdir()}/fvc-cli-`);
  writeFileSync(`${dir}/token`, token, { mode: 0o600 });
  const cli = async (...args) => (await promisify(execFile)("bash", [new URL("../../../scripts/serve/fv-control.sh", import.meta.url).pathname, ...args], { env: { ...process.env, FV_CONTROL_URL: B, FV_CONTROL_TOKEN_FILE: `${dir}/token`, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" } })).stdout;
  const list = await cli("jobs", "tiny", "--status", "running,succeeded");
  assert.match(list, /^JOB\tAPI\tSTATUS/m);
  assert.match(list, /fvjob_run1\tnative\trunning\tfake-wan\tfake\t/);
  assert.match(await cli("jobs", worker.id), /fvjob_done/, "a pod id narrows to that pod");
  mock.internalJobs["0000000" + "1-aaaa-4bbb-8ccc-dddddddddddd"] = [200, { status: "running", cancel_requested: true }];
  assert.match(await cli("job-cancel", "fvjob_run1"), /fvjob_run1 \(native, tiny\): running, cancel requested via internal on /);
  assert.match(await cli("jobs-cancel-queued", "tiny", "--yes"), /^$|cancelled/);
  mock.queue.jobs.set("rp-cli-1", { id: "rp-cli-1", status: "IN_QUEUE" });
  mock.queue.queued = 2;
  assert.match(await cli("endpoint", "cancel", "qmock", "rp-cli-1"), /rp-cli-1: IN_QUEUE -> CANCELLED \(cancelled\); removed from the queue/);
  assert.match(await cli("endpoint", "purge", "qmock", "--yes"), /removed 1 \(Runpod: completed\); queued 1 -> 0/);
});

await step("the collector: pods, owners, costs, samples, external attribution, alerts", async () => {
  const r = await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.equal(r.status, 200, JSON.stringify(r.j));
  assert.equal(r.j.balance, 50);
  const pods = (await call("/api/pods", { headers: T() })).j.pods;
  const by = Object.fromEntries(pods.map((p) => [p.pod_id, p]));
  assert.equal(by[worker.id].owner, "cluster:tiny");
  assert.equal(by[worker.id].health, "ready");
  assert.equal(by[worker.id].jobs_running, 0);
  assert.equal(by.extbuild0001.owner, "external:build-pod");
  assert.equal(by.extgpu00001.owner, "external:b200-bench");
  assert.ok(by.extgpu00001.idle_since, "a GPU pod at 2% with no jobs is idle");
  const costs = (await call("/api/costs?days=1", { headers: T() })).j;
  assert.ok(costs.by_owner.find((o) => o.owner === "external:b200-bench").usd > 0.1);
  assert.ok(costs.by_cluster.find((x) => x.name === "tiny"));
  const series = (await call("/api/metrics/series?hours=1", { headers: T() })).j;
  assert.equal(series.source, "d1");
  assert.ok(series.points.some((p) => p.pod === "extgpu00001" && p.gpu === 2));
  const ov = (await call("/api/overview", { headers: T() })).j;
  assert.equal(ov.balance, 50);
  assert.ok(ov.hours_to_floor > 0);
  assert.equal(ov.clusters[0].name, "tiny");
  // Idle alert once the pod has been idle long enough (policy idle_min 20 from the step above).
  await d1Exec(w.dir, `UPDATE pods SET idle_since = ${Date.now() - 3600_000} WHERE pod_id = 'extgpu00001'`);
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  const al = (await call("/api/alerts", { headers: T() })).j.alerts;
  const idleAlert = al.find((a) => a.kind === "pod_idle" && a.target === "extgpu00001");
  assert.ok(idleAlert, JSON.stringify(al));
  // Resolve: closed now, open again at the next pass while the pod is still idle.
  assert.equal((await call(`/api/alerts/${idleAlert.id}/resolve`, { method: "POST", body: {}, headers: T() })).status, 200);
  assert.ok(!(await call("/api/alerts", { headers: T() })).j.alerts.some((a) => a.id === idleAlert.id));
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok((await call("/api/alerts", { headers: T() })).j.alerts.some((a) => a.kind === "pod_idle" && a.target === "extgpu00001"), "re-opened");
  assert.ok(mock.pods.has(worker.id) && !mock.log.some((l) => l.method === "DELETE" && l.path.includes("extgpu")), "external pods are never touched");
});

await step("CloudRift: rentals collected, deadline backstop, balance, terminate only ours", async () => {
  // The collector step above already ran the cron once.
  const cr = mock.cloudrift;
  assert.equal(cr.instances.find((i) => i.id === "cr-ours-1").status, "Inactive", "past its fv-deadline tag: terminated");
  assert.equal(cr.instances.find((i) => i.id === "cr-ours-2").status, "Active");
  assert.equal(cr.instances.find((i) => i.id === "cr-foreign").status, "Active", "a foreign rental is never touched (even on a type we refuse)");
  assert.ok(cr.calls.filter((c) => c.path !== "instance-types/list").every((c) => c.key && !c.bearer && c.version === "2026-09-08"));
  const pods = (await call("/api/pods", { headers: T() })).j.pods;
  const by = Object.fromEntries(pods.map((p) => [p.pod_id, p]));
  assert.equal(by["cr-ours-2"].provider, "cloudrift");
  assert.equal(by["cr-ours-2"].owner, "cloudrift:serve-h3-turbo");
  assert.equal(by["cr-foreign"].owner, "external:cloudrift");
  assert.equal(by[worker.id].provider, "runpod");
  // The deadline alert resolved itself once the rental was gone (the next cron); the audit keeps the action.
  const aud = (await call("/api/audit?limit=500", { headers: T() })).j.audit;
  assert.ok(aud.some((x) => x.action === "cloudrift.terminate" && x.target === "cr-ours-1" && x.actor === "policy:cloudrift_deadline"), JSON.stringify(aud.slice(0, 3)));
  const ov = (await call("/api/overview", { headers: T() })).j;
  assert.equal(ov.cloudrift.balance, 30, "account/info's 3000 cents");
  assert.equal(ov.balance, 50, "the Runpod balance is separate");
  const prov = (await call("/api/providers", { headers: T() })).j.providers;
  assert.deepEqual(prov.map((p) => [p.id, p.enabled]), [["runpod", true], ["cloudrift", true], ["gmi", true], ["brev", true]]);
  const price = (await call("/api/providers/cloudrift/price?gpu=RTX%20PRO%206000", { headers: T() })).j;
  assert.equal(price.offers[0].usd_per_hr, 1.3936);
  // Owner rule: only RTX PRO 6000 and RTX 5090 on CloudRift.
  assert.equal((await call("/api/providers/cloudrift/price?gpu=V100%20SXM2", { headers: T() })).status, 400);
  assert.equal((await call("/api/providers/cloudrift/instances/cr-foreign/terminate", { method: "POST", body: {}, headers: T() })).status, 403);
  const r = await call("/api/providers/cloudrift/instances/cr-ours-2/terminate", { method: "POST", body: {}, headers: T() });
  assert.equal(r.status, 200, JSON.stringify(r.j));
  assert.equal(cr.instances.find((i) => i.id === "cr-ours-2").status, "Inactive");
  // A second run of the cron: the rows go, the Runpod pods stay.
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  const after = Object.fromEntries((await call("/api/pods", { headers: T() })).j.pods.map((p) => [p.pod_id, p]));
  assert.ok(!after["cr-ours-2"], "terminated rental marked gone");
  assert.ok(after[worker.id], "Runpod pods are not marked gone by the CloudRift half");
  // The balance floor: below it, our live rentals are terminated (none left), alert critical.
  cr.balance = 500; // cents: $5
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok((await call("/api/alerts", { headers: T() })).j.alerts.some((a) => a.kind === "cloudrift_balance_floor"));
  assert.equal(cr.instances.find((i) => i.id === "cr-foreign").status, "Active");
  cr.balance = 3000;
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok(!(await call("/api/alerts", { headers: T() })).j.alerts.some((a) => a.kind === "cloudrift_balance_floor"), "resolved");
});

await step("env at three levels: masked view, restart needed, rolling restart", async () => {
  assert.equal((await call("/api/env/account/RUST_LOG", { method: "PUT", body: { value: "warn" }, headers: T() })).status, 200);
  assert.equal((await call(`/api/env/cluster/${cid}/HF_TOKEN`, { method: "PUT", body: { value: "hf_supersecret", secret: true }, headers: T() })).status, 200);
  assert.equal((await call(`/api/env/pod/${worker.id}/RUST_LOG`, { method: "PUT", body: { value: "debug" }, headers: T() })).status, 200);
  assert.equal((await call(`/api/env/cluster/${cid}/FV_INTERNAL_TOKEN`, { method: "PUT", body: { value: "x" }, headers: T() })).status, 400);
  const e = (await call(`/api/clusters/${cid}/env`, { headers: T() })).j;
  assert.deepEqual(e.needs_restart, [worker.id]);
  const wenv = Object.fromEntries(e.pods.find((p) => p.pod_id === worker.id).env.map((v) => [v.key, v]));
  assert.equal(wenv.RUST_LOG.value, "debug");
  assert.equal(wenv.RUST_LOG.source, "pod");
  assert.equal(wenv.HF_TOKEN.value, "••••••••");
  assert.equal(wenv.FV_INTERNAL_TOKEN.secret, true);
  const s = await call(`/api/clusters/${cid}/restart`, { method: "POST", body: {}, headers: T() });
  assert.equal(s.status, 202);
  const op = await waitOp(cid, "restart");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.get(worker.id).env.RUST_LOG, "debug");
  assert.equal(mock.pods.get(worker.id).env.HF_TOKEN, "hf_supersecret");
  const patches = mock.log.filter((l) => l.method === "PATCH").slice(-1).map((l) => l.path.split("/").pop());
  assert.deepEqual(patches, [worker.id]);
  assert.deepEqual((await call(`/api/clusters/${cid}/env`, { headers: T() })).j.needs_restart, []);
});

await step("pool env: every worker of the pool; restart by pool or pod", async () => {
  assert.equal((await call(`/api/env/pool/${cid}:fake/FASTVIDEO_ATTN_SAGE`, { method: "PUT", body: { value: "0" }, headers: T() })).status, 200);
  // By cluster name too; an unknown pool is refused; reserved keys too.
  assert.equal((await call(`/api/env/pool/tiny:fake/POOL_SECRET`, { method: "PUT", body: { value: "pool_s3cret_value", secret: true }, headers: T() })).status, 200);
  assert.equal((await call(`/api/env/pool/${cid}:nope/X`, { method: "PUT", body: { value: "1" }, headers: T() })).status, 404);
  assert.equal((await call(`/api/env/pool/${cid}:fake/FV_INTERNAL_TOKEN`, { method: "PUT", body: { value: "x" }, headers: T() })).status, 400);
  const keys = (await call(`/api/env/pool/${cid}:fake`, { headers: T() })).j.vars.map((v) => v.key);
  assert.deepEqual(keys.sort(), ["FASTVIDEO_ATTN_SAGE", "POOL_SECRET"]);
  const doc = (await call(`/api/docs/env/pool:${cid}:fake`, { headers: T() })).j;
  assert.deepEqual(doc.doc.POOL_SECRET, { value: null, secret: true });
  const e = (await call(`/api/clusters/${cid}/env`, { headers: T() })).j;
  assert.deepEqual(e.needs_restart, [worker.id]);
  const wenv = Object.fromEntries(e.pods.find((p) => p.pod_id === worker.id).env.map((v) => [v.key, v]));
  assert.equal(wenv.FASTVIDEO_ATTN_SAGE.source, "pool");
  assert.equal(wenv.POOL_SECRET.value, "••••••••");
  assert.ok(e.preview.fake.some((v) => v.key === "FASTVIDEO_ATTN_SAGE" && v.source === "pool"), "a new worker of the pool gets it");
  await d1Exec(w.dir, "DELETE FROM operations WHERE kind = 'restart'");
  assert.equal((await call(`/api/clusters/${cid}/restart`, { method: "POST", body: { pools: ["fake"] }, headers: T() })).status, 202);
  let op = await waitOp(cid, "restart");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.get(worker.id).env.FASTVIDEO_ATTN_SAGE, "0");
  assert.equal(mock.pods.get(worker.id).env.POOL_SECRET, "pool_s3cret_value");
  // A chosen pod restarts even when its env did not change.
  const p0 = mock.pods.get(worker.id).patches;
  await d1Exec(w.dir, "DELETE FROM operations WHERE kind = 'restart'");
  assert.equal((await call(`/api/clusters/${cid}/restart`, { method: "POST", body: { pods: [worker.id] }, headers: T() })).status, 202);
  op = await waitOp(cid, "restart");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.get(worker.id).patches, p0 + 1);
  await d1Exec(w.dir, "DELETE FROM operations WHERE kind = 'restart'");
});

await step("templates and pool presets; a legacy gateway block migrates", async () => {
  const tp = (await call("/api/templates", { headers: T() })).j;
  assert.deepEqual(tp.templates.map((x) => x.id), ["standard", "tiny-cpu", "ltx", "h3", "wan", "longlive"]);
  for (const id of ["ltx-pro", "ltx-a2v", "ltx-ref2v", "h3-ref2v", "fastwan21", "sfwan", "longlive"]) assert.ok(tp.pool_presets.some((p) => p.id === id), id);
  assert.match(tp.pool_presets.find((p) => p.id === "longlive").licence, /non-commercial/i);
  assert.equal(tp.ltx.pools.length, 4);
  const r = await call("/api/clusters", { method: "POST", body: { spec: { name: "ltxs", template: "ltx", gateway: { fal_apps: ["lightricks/ltx-2.5", "fal-ai/ltx-2.3"], protocols: { fastwan: true }, reactor_model: null } } }, headers: T() });
  assert.equal(r.status, 201, JSON.stringify(r.j));
  const s = r.j.cluster.spec;
  assert.deepEqual(s.pools.map((p) => p.id), ["ltx", "ltx-pro", "ltx-a2v", "ltx-ref2v"]);
  assert.match(s.pools[1].config_toml, /recipe = "ltx-pro"/);
  assert.equal(s.gateway, undefined, "the retired gateway block is dropped");
  assert.equal(s.control_plane, "edge");
  assert.equal(s.auth, "keys");
  // A bare preset id fills in; a Plug recipe is refused.
  const ok = await call(`/api/clusters/${r.j.cluster.id}/spec`, { method: "PUT", body: { spec: { ...s, pools: [...s.pools, { id: "longlive", count: 0 }] } }, headers: T() });
  assert.equal(ok.status, 200, JSON.stringify(ok.j));
  assert.equal(ok.j.cluster.spec.pools[4].variant, "sfwan");
  const bad = await call(`/api/clusters/${r.j.cluster.id}/spec`, { method: "PUT", body: { spec: { ...s, pools: [{ ...s.pools[0], models: [{ id: "x", family: "h3", recipe: "h3-plug-4step" }] }] } }, headers: T() });
  assert.equal(bad.status, 400);
  assert.match(bad.j.error, /not in the fv-serve catalog/);
  // Its pool env and suggestions.
  assert.equal((await call(`/api/env/pool/ltxs:ltx-pro/FASTVIDEO_ATTN_SAGE`, { method: "PUT", body: { value: "0" }, headers: T() })).status, 200);
  const dyn = (await call(`/api/schemas/dynamic?cluster=${r.j.cluster.id}`, { headers: T() })).j;
  assert.ok(dyn.recipes.some((x) => x.id === "wan5b-plug-4step" && /NOT servable/.test(x.detail)));
  assert.ok(dyn.fal_apps.some((x) => x.id === "fal-ai/ltx-2.3-quality"));
  assert.ok(dyn.model_ids.some((x) => x.id === "ltx25-distill-dense"));
  assert.match(dyn.env_keys.find((k) => k.id === "FASTVIDEO_ATTN_SAGE").detail, /in use: (cluster|pool)/);
  assert.equal(dyn.variants.find((v) => v.id === "ltx").detail, "presets: ltx, ltx-pro, ltx-a2v, ltx-ref2v");
  // Deleting the definition drops its pool env.
  assert.equal((await call(`/api/clusters/${r.j.cluster.id}`, { method: "DELETE", headers: T() })).status, 200);
  assert.equal(JSON.parse(d1Exec(w.dir, `SELECT COUNT(*) AS n FROM env_vars WHERE scope = 'pool' AND scope_id LIKE '${r.j.cluster.id}:%'`))[0].results[0].n, 0);
});

await step("the build pod card: its /healthz timers, last self-stop, jobs, the backstop's distance", async () => {
  mock.buildHealth = { extbuild0001: { ok: true, ready: true, phase: "ready", boot: 1, uptime_s: 3600, idle_s: 0, idle_stop_in_s: null, max_stop_in_s: 25200, idle_stop_s: 1200, max_s: 28800, max_grace_s: 1800, jobs_active: 1, self_stop: { attempts: 1, next_at: null, reason: "idle 20 min", at: 1000, ok: null, error: `REST stop: HTTP 403 ${SECRETS.RUNPOD_API_KEY}` }, jobs: [{ id: "1002-abc", agent: "wt-ui-dash", state: "running", seconds: 42 }] } };
  const r = (await call("/api/buildpod", { headers: T() })).j;
  assert.equal(r.pods.length, 1);
  const p = r.pods[0];
  assert.equal(p.pod_id, "extbuild0001");
  assert.equal(p.health.jobs[0].agent, "wt-ui-dash");
  assert.equal(p.backstop.cap_in_s, 9 * 3600 - 3600);
  assert.equal(p.backstop.idle_in_s, null, "jobs running: no idle countdown");
  assert.equal(p.backstop.verdict, null);
  assert.ok(!JSON.stringify(r).includes(SECRETS.RUNPOD_API_KEY), "the self-stop error is scrubbed");
  assert.equal(r.policy.max_h, 9);
  assert.ok(!mock.log.some((l) => l.path.includes("extbuild0001") && l.method !== "GET"), "read only");
});

await step("log shipping: ingest, search, level filter, live tail, download, auth", async () => {
  const ingestTok = mock.pods.get(worker.id).env.FV_LOG_SHIP_TOKEN;
  assert.match(ingestTok, /^fvi_/);
  const ws = new WebSocket(`${B.replace("http", "ws")}/api/logs/tail?pod=${worker.id}`, { headers: T() });
  const got = [];
  ws.onmessage = (ev) => got.push(JSON.parse(ev.data));
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  const lines = [
    { ts: Date.now(), level: "INFO", target: "fv_serve::jobs", fields: { message: "job started", job_id: "job_abc" } },
    { ts: Date.now(), level: "WARN", target: "fv_serve::engine", fields: { message: "slow step", job_id: "job_abc" } },
    { ts: Date.now(), level: "DEBUG", target: "x", fields: { message: `oops ${SECRETS.RUNPOD_API_KEY}` } },
  ];
  assert.equal((await call("/ingest/v1/logs", { method: "POST", body: { pod: worker.id, lines } })).status, 401);
  assert.equal((await call("/ingest/v1/logs", { method: "POST", body: { pod: worker.id, lines }, headers: { authorization: "Bearer fvi_wrong" } })).status, 401);
  assert.equal((await call("/ingest/v1/logs", { method: "POST", body: { pod: "notmypod1234", lines }, headers: { authorization: `Bearer ${ingestTok}` } })).status, 403);
  const r = await call("/ingest/v1/logs", { method: "POST", body: { pod: worker.id, lines }, headers: { authorization: `Bearer ${ingestTok}` } });
  assert.equal(r.j.accepted, 3);
  const all = (await call(`/api/logs?pod=${worker.id}&level=trace&source=serve`, { headers: T() })).j.lines;
  assert.equal(all.length, 3);
  assert.ok(!JSON.stringify(all).includes(SECRETS.RUNPOD_API_KEY), "secrets scrubbed from logs");
  assert.equal((await call(`/api/logs?pod=${worker.id}&level=warn&source=serve`, { headers: T() })).j.lines.length, 1);
  assert.equal((await call(`/api/logs?pod=${worker.id}&level=trace&q=job_abc`, { headers: T() })).j.lines.length, 2);
  // Logs from boot: the Runpod container and system tail the controller copied (the `up` wait and the cron), once each, scrubbed.
  const rp = (await call(`/api/pods/${worker.id}/logs?source=runpod`, { headers: T() })).j.lines;
  // (The mock's container lines are dated 2026-09-29: the cron's 24 h retention drops them; its system line has no date.)
  assert.ok(rp.some((l) => l.target === "runpod.system" && l.msg === "pulling image"), JSON.stringify(rp));
  assert.ok(!JSON.stringify(rp).includes(SECRETS.RUNPOD_API_KEY), "secrets scrubbed from captured Runpod logs");
  const st = (await call(`/api/pods/${worker.id}/status`, { headers: T() })).j;
  assert.equal(st.kind, "cluster");
  assert.ok(st.logs.lines >= 4, JSON.stringify(st.logs));
  for (let i = 0; i < 20 && !got.length; i++) await sleep(100);
  assert.equal(got[0]?.lines?.length, 3, "live tail got the batch");
  ws.close();
  const d = await call(`/api/logs/download?pod=${worker.id}`, { headers: T(), raw: true });
  assert.equal(d.text.trim().split("\n").filter((l) => !JSON.parse(l).target?.startsWith("runpod.")).length, 3);
  assert.match(d.headers.get("content-disposition"), /attachment/);
  const nd = await call(`/ingest/v1/logs?pod=${worker.id}`, { method: "POST", body: lines.map((l) => JSON.stringify(l)).join("\n"), headers: { authorization: `Bearer ${ingestTok}`, "content-type": "application/x-ndjson" } });
  assert.equal(nd.j.accepted, 3);
  const rl = (await call(`/api/pods/${worker.id}/runpod-logs`, { headers: T() })).j;
  assert.equal(rl.container.length, 2);
  assert.ok(!JSON.stringify(rl).includes(SECRETS.RUNPOD_API_KEY));
});

await step("schemas, dynamic values and the document API (versions, validation, plan, history, restore)", async () => {
  const sc = (await call("/api/schemas", { headers: T() })).j;
  assert.ok(sc.schemas["cluster-spec"].properties.pools);
  assert.equal(sc.documents.env, "env");
  const dyn = (await call(`/api/schemas/dynamic?cluster=${cid}`, { headers: T() })).j;
  assert.equal(dyn.gpu_types.length, 3);
  assert.equal(dyn.gpu_types[0].secure_price, 2.09, "sorted by price");
  assert.deepEqual(dyn.pools, ["fake"]);
  assert.ok(dyn.env_keys.some((k) => k.id === "HF_TOKEN" && /in use/.test(k.detail)));
  assert.ok(dyn.env_keys.some((k) => k.id === "FASTVIDEO_WAN_AUDIO" && /mmaudio/.test(k.detail)));
  assert.ok(dyn.regions.find((r) => r.id === "eu").volume === "jg48s6o1w0");
  assert.ok(!dyn.regions.find((r) => r.id === "us"), "us is not offered (its weights volume is gone)");
  assert.deepEqual(sc.schemas["cluster-spec"].properties.regions.items.enum, ["eu"]);
  const d = (await call(`/api/docs/cluster-spec/${cid}`, { headers: T() })).j;
  assert.equal(d.schema, "cluster-spec");
  const v0 = d.version;
  const bad = (await call(`/api/docs/cluster-spec/${cid}/validate`, { method: "POST", body: { doc: { ...d.doc, cap_s: 5, image: { channel: "stable", sha: "abcdef1" } } }, headers: T() })).j;
  assert.equal(bad.ok, false);
  assert.ok(bad.issues.some((i) => i.path.join(".") === "cap_s"));
  assert.ok(bad.issues.some((i) => /exactly one/.test(i.message)));
  const put400 = await call(`/api/docs/cluster-spec/${cid}`, { method: "PUT", body: { doc: { ...d.doc, cap_s: 5 }, version: v0 }, headers: T() });
  assert.equal(put400.status, 400);
  assert.ok(put400.j.issues.length);
  const putUs = await call(`/api/clusters/${cid}/spec`, { method: "PUT", body: { spec: { ...d.doc, regions: ["eu", "us"] } }, headers: T() });
  assert.equal(putUs.status, 400, "a spec naming us is rejected");
  assert.match(JSON.stringify(putUs.j), /US weights volume deleted 2026-10; EU only, see docs\/ops\/runpod-volumes.md/);
  const plan = (await call(`/api/docs/cluster-spec/${cid}/plan`, { method: "POST", body: { doc: { ...d.doc, pools: [{ ...d.doc.pools[0], count: 3 }] } }, headers: T() })).j;
  assert.equal(plan.ok, true);
  assert.ok(plan.projection.cluster_dph > 0);
  const ok = await call(`/api/docs/cluster-spec/${cid}`, { method: "PUT", body: { doc: { ...d.doc, max_gpu_dph: 3 }, version: v0 }, headers: T() });
  assert.equal(ok.status, 200, JSON.stringify(ok.j));
  assert.equal(ok.j.version, v0 + 1);
  const stale = await call(`/api/docs/cluster-spec/${cid}`, { method: "PUT", body: { doc: { ...d.doc, max_gpu_dph: 4 }, version: v0 }, headers: T() });
  assert.equal(stale.status, 409);
  assert.equal(stale.j.current_version, v0 + 1);
  assert.equal((await call(`/api/docs/cluster-spec/${cid}`, { method: "PUT", body: { doc: d.doc }, headers: T() })).status, 400, "a version is required");
  const hist = (await call(`/api/docs/cluster-spec/${cid}/history`, { headers: T() })).j.history;
  assert.equal(hist[0].after.max_gpu_dph, 3);
  const rs = await call(`/api/docs/cluster-spec/${cid}/restore`, { method: "POST", body: { audit_id: hist[0].audit_id, which: "before", version: v0 + 1 }, headers: T() });
  assert.equal(rs.status, 200, JSON.stringify(rs.j));
  assert.equal(rs.j.doc.max_gpu_dph, d.doc.max_gpu_dph);
  // Env document: secrets never come back.
  const e = (await call(`/api/docs/env/cluster:${cid}`, { headers: T() })).j;
  assert.deepEqual(e.doc.HF_TOKEN, { value: null, secret: true });
  const e2 = await call(`/api/docs/env/cluster:${cid}`, { method: "PUT", body: { doc: { ...e.doc, NEW_SECRET: { value: null, secret: true, set: "s3cr3t_new_value" } }, version: e.version }, headers: T() });
  assert.equal(e2.status, 200, JSON.stringify(e2.j));
  assert.deepEqual(e2.j.doc.NEW_SECRET, { value: null, secret: true });
  assert.ok(!bodies.join("").includes("s3cr3t_new_value"));
  // The legacy per-key route bumps the same version.
  await call(`/api/env/cluster/${cid}/OTHER`, { method: "PUT", body: { value: "1" }, headers: T() });
  assert.equal((await call(`/api/docs/env/cluster:${cid}`, { headers: T() })).j.version, e2.j.version + 1);
  await call(`/api/env/cluster/${cid}/OTHER`, { method: "DELETE", headers: T() });
  await call(`/api/env/cluster/${cid}/NEW_SECRET`, { method: "DELETE", headers: T() });
  assert.equal((await call("/api/tokens", { method: "POST", body: { name: "x", scope: "root" }, headers: { cookie, "x-csrf-token": csrf } })).status, 400, "token scope validated by schema");
});

await step("scale a pool up and down (drain first)", async () => {
  assert.equal((await call(`/api/clusters/${cid}/scale`, { method: "POST", body: { pool: "fake", count: 2 }, headers: T() })).status, 202);
  let op = await waitOp(cid, "scale");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.size, 2);
  const grown = [...mock.pods.values()].find((p) => p.id !== worker.id);
  assert.equal(grown.env.FV_DISPATCH_FRONT, "1");
  assert.equal(grown.env.FASTVIDEO_ATTN_SAGE, "0", "a scale-up worker gets the pool env");
  await d1Exec(w.dir, "DELETE FROM operations WHERE kind = 'scale'");
  assert.equal((await call(`/api/clusters/${cid}/scale`, { method: "POST", body: { pool: "fake", count: 1 }, headers: T() })).status, 202);
  op = await waitOp(cid, "scale", 90000);
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.size, 1);
  assert.equal(mock.drained.length, 1);
  worker = [...mock.pods.values()][0];
});

await step("rolling redeploy to a commit", async () => {
  const old = worker.id;
  assert.equal((await call(`/api/clusters/${cid}/roll`, { method: "POST", body: { target: "abcdef1" }, headers: T() })).status, 202);
  const op = await waitOp(cid, "roll", 120000);
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.ok(!mock.pods.has(old), "old worker deleted");
  assert.ok(mock.drained.includes(old), "old worker drained first");
  assert.equal(mock.pods.size, 1);
  const nw = [...mock.pods.values()][0];
  assert.match(nw.image, /@sha256:/);
  assert.equal(nw.env.FV_IMAGE_DIGEST, "sha256:" + Buffer.from("cpu-sha-abcdef1").toString("hex").padEnd(64, "0").slice(0, 64));
  assert.equal(nw.env.FASTVIDEO_ATTN_SAGE, "0", "a rolled worker keeps the pool env");
  worker = nw;
});

await step("extend; the edge's admin token; keys minted, listed and revoked at the edge", async () => {
  const before = (await call(`/api/clusters/${cid}`, { headers: T() })).j.cluster.deadline;
  const wdl = mock.pods.get(worker.id).env.FV_CLUSTER_DEADLINE;
  assert.equal((await call(`/api/clusters/${cid}/extend`, { method: "POST", body: { minutes: 30 }, headers: T() })).status, 202);
  const op = await waitOp(cid, "extend");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal((await call(`/api/clusters/${cid}`, { headers: T() })).j.cluster.deadline, before + 1800_000);
  assert.equal(mock.pods.get(worker.id).env.FV_CLUSTER_DEADLINE, wdl, "the worker keeps its launch deadline until restarted");
  assert.equal((await call(`/api/clusters/${cid}/gateway/stop`, { method: "POST", body: {}, headers: T() })).status, 404, "the gateway operations are gone");
  const tok = await call(`/api/clusters/${cid}/admin-token`, { method: "POST", body: {}, headers: T() });
  assert.equal(tok.status, 200, JSON.stringify(tok.j));
  assert.equal(tok.j.admin_token, mock.edgeAdmin);
  const k = await call(`/api/clusters/${cid}/mint-key`, { method: "POST", body: { name: "laptop" }, headers: T() });
  assert.equal(k.status, 201, JSON.stringify(k.j));
  // List and revoke at the edge (audited).
  const ks = await call(`/api/clusters/${cid}/keys`, { headers: T() });
  assert.equal(ks.status, 200, JSON.stringify(ks.j));
  assert.ok(ks.j.keys.some((x) => x.name === "laptop"));
  const kid = k.j.key.id;
  assert.equal((await call(`/api/clusters/${cid}/keys/not-a-key`, { method: "DELETE", headers: T() })).status, 400);
  assert.equal((await call(`/api/clusters/${cid}/keys/key_999999999999`, { method: "DELETE", headers: T() })).status, 404);
  const rv = await call(`/api/clusters/${cid}/keys/${kid}`, { method: "DELETE", headers: T() });
  assert.equal(rv.status, 200, JSON.stringify(rv.j));
  assert.deepEqual(rv.j.applied, ["edge"]);
  assert.equal((await call(`/api/clusters/${cid}/keys`, { headers: T() })).j.keys.find((x) => x.id === kid).revoked, true);
  assert.ok((await call("/api/audit", { headers: T() })).j.audit.some((a) => a.action === "cluster.revoke-key" && a.after.includes(kid)));
  const fv = (await call(`/api/clusters/${cid}/front`, { headers: T() })).j;
  assert.equal(fv.edge, true);
  assert.equal(fv.workers[0].front.ready, true, JSON.stringify(fv));
});

await step("GitHub: release dispatch, CI status; image tags", async () => {
  const r = await call("/api/github/release", { method: "POST", body: { action: "promote", target: "abcdef1", channel: "stable", dry_run: true }, headers: T() });
  assert.equal(r.status, 202, JSON.stringify(r.j));
  assert.equal(mock.dispatches[0].inputs.target, "abcdef1");
  assert.equal(mock.dispatches[0].inputs.dry_run, "true");
  assert.equal((await call("/api/github/release", { method: "POST", body: { action: "promote", target: "$(x)" }, headers: T() })).status, 400);
  const ci = (await call("/api/github/ci", { headers: T() })).j;
  assert.equal(ci.main[0].conclusion, "success");
  assert.ok((await call("/api/images/tags?filter=cpu", { headers: T() })).j.tags.includes("cpu-stable"));
});

await step("deadline backstop: the cron stops a cluster past its deadline", async () => {
  await d1Exec(w.dir, `UPDATE clusters SET deadline = ${Date.now() - 1000} WHERE id = '${cid}'`);
  const r = await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok(r.j.actions.some((a) => a.includes("deadline")), JSON.stringify(r.j));
  const op = await waitOp(cid, "down", 60000);
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.size, 0);
  const c = (await call(`/api/clusters/${cid}`, { headers: T() })).j.cluster;
  assert.equal(c.status, "stopped");
  assert.ok((await call("/api/alerts", { headers: T() })).j.alerts.some((a) => a.kind === "deadline"));
});

await step("the balance floor stops running clusters", async () => {
  await d1Exec(w.dir, "DELETE FROM operations");
  assert.equal((await call(`/api/clusters/${cid}/start`, { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(cid, "up")).status, "done");
  assert.equal(mock.pods.size, 1);
  await d1Exec(w.dir, "DELETE FROM operations");
  mock.balance = 7.5;
  const r = await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok(r.j.actions.some((a) => a.includes("balance floor")), JSON.stringify(r.j));
  assert.equal((await waitOp(cid, "down")).status, "done");
  assert.equal(mock.pods.size, 0);
  assert.ok(!mock.log.some((l) => l.method === "DELETE" && /ext/.test(l.path)), "external pods untouched");
  mock.balance = 50;
});

await step("direct cluster: the controller's admin token on every worker; keys minted, listed and revoked on the workers", async () => {
  await d1Exec(w.dir, "DELETE FROM operations");
  const r = await call("/api/clusters", { method: "POST", body: { spec: { name: "nogw", template: "tiny-cpu", image: { channel: "stable" }, cap_s: 3600, control_plane: "direct" } }, headers: T() });
  assert.equal(r.status, 201, JSON.stringify(r.j));
  const id = r.j.cluster.id;
  const before = new Set(mock.pods.keys());
  const mine = () => [...mock.pods.values()].filter((p) => !before.has(p.id));
  assert.equal((await call(`/api/clusters/${id}/start`, { method: "POST", body: {}, headers: T() })).status, 202);
  let op = await waitOp(id, "up");
  assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-5)));
  assert.equal(mine().length, 1);
  const w1 = mine()[0];
  assert.equal(w1.env.FV_SERVE_ROLE, "worker");
  assert.equal(w1.env.FV_DISPATCH_FRONT, undefined);
  assert.equal(w1.env.FV_WORKER_DIRECT, "1");
  assert.equal(w1.env.FV_KEY_STORE, "d1");
  assert.equal(w1.env.FV_AUTH_MODE, "keys");
  assert.match(w1.env.FV_ADMIN_TOKEN, /^fvadm_[0-9a-f]{48}$/);
  const tok = await call(`/api/clusters/${id}/admin-token`, { method: "POST", body: {}, headers: T() });
  assert.equal(tok.status, 200, JSON.stringify(tok.j));
  assert.equal(tok.j.admin_token, w1.env.FV_ADMIN_TOKEN);
  assert.equal(tok.j.direct, true);
  assert.deepEqual(tok.j.workers.map((x) => x.url), [`http://127.0.0.1:${mock.port}/pod/${w1.id}`]);
  assert.equal(tok.j.console, `http://127.0.0.1:${mock.port}/pod/${w1.id}/console/admin`);
  const e = (await call(`/api/clusters/${id}/env`, { headers: T() })).j;
  const wenv = Object.fromEntries(e.pods.find((p) => p.pod_id === w1.id).env.map((v) => [v.key, v]));
  assert.equal(wenv.FV_ADMIN_TOKEN.value, "••••••••", "masked in the env view");
  assert.deepEqual(e.needs_restart, []);
  // Scale-up: the new worker gets the same token (and the same D1 keys).
  assert.equal((await call(`/api/clusters/${id}/scale`, { method: "POST", body: { pool: "fake", count: 2 }, headers: T() })).status, 202);
  op = await waitOp(id, "scale");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  const w2 = mine().find((p) => p.id !== w1.id);
  assert.equal(w2.env.FV_ADMIN_TOKEN, w1.env.FV_ADMIN_TOKEN);
  const gv = (await call(`/api/clusters/${id}/front`, { headers: T() })).j;
  assert.equal(gv.direct, true);
  assert.equal(gv.workers.length, 2);
  assert.ok(gv.workers.every((x) => x.health.ok), JSON.stringify(gv));
  // Mint on one worker, list, revoke on every worker.
  const k = await call(`/api/clusters/${id}/mint-key`, { method: "POST", body: { name: "phone" }, headers: T() });
  assert.equal(k.status, 201, JSON.stringify(k.j));
  assert.match(k.j.api_key, /^fv_direct_key_/);
  assert.equal(k.j.propagation_s, 30);
  const kid = k.j.key.id;
  const l = await call(`/api/clusters/${id}/keys`, { headers: T() });
  assert.ok(l.j.keys.some((x) => x.id === kid && x.name === "phone"), JSON.stringify(l.j));
  assert.equal((await call(`/api/clusters/${id}/keys/nope`, { method: "DELETE", headers: T() })).status, 400);
  const rv = await call(`/api/clusters/${id}/keys/${kid}`, { method: "DELETE", headers: T() });
  assert.equal(rv.status, 200, JSON.stringify(rv.j));
  assert.deepEqual(rv.j.applied.sort(), [w1.id, w2.id].sort());
  assert.equal(mock.directKeys.find((x) => x.id === kid).revoked, true);
  assert.equal(mock.directCalls.filter((x) => x.method === "DELETE").length, 2, "the revocation went to every worker");
  // Stop: no pod left.
  assert.equal((await call(`/api/clusters/${id}/stop`, { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(id, "down")).status, "done");
  assert.equal(mine().length, 0);
});

await step("a second edge cluster: fronts behind the edge; register, ready from the families view, keys at the edge, scale, roll, one edge cluster at a time", async () => {
  await d1Exec(w.dir, "DELETE FROM operations");
  const edge = `http://127.0.0.1:${mock.port}/edge`;
  const r = await call("/api/clusters", { method: "POST", body: { spec: { name: "edgy", template: "tiny-cpu", image: { channel: "stable" }, cap_s: 3600, control_plane: "edge" } }, headers: T() });
  assert.equal(r.status, 201, JSON.stringify(r.j));
  assert.equal(r.j.cluster.spec.control_plane, "edge");
  const id = r.j.cluster.id;
  const before = new Set(mock.pods.keys());
  const mine = () => [...mock.pods.values()].filter((p) => !before.has(p.id));
  assert.equal((await call(`/api/clusters/${id}/start`, { method: "POST", body: {}, headers: T() })).status, 202);
  let op = await waitOp(id, "up");
  assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-6)));
  assert.ok(op.log.some((l) => /edge .*: up/.test(l.msg)), "the register phase checked the edge");
  assert.ok(op.log.some((l) => /ready = a ready front at the edge\) ready 1\/1: fake/.test(l.msg)), JSON.stringify(op.log.slice(-4)));
  assert.equal(mine().length, 1);
  const w1 = mine()[0];
  assert.equal(w1.env.FV_DISPATCH_FRONT, "1");
  assert.equal(w1.env.FV_DISPATCH_DO_URL, edge);
  assert.equal(w1.env.FV_PUBLIC_BASE_URL, edge);
  assert.equal(w1.env.FV_INTERNAL_TOKEN, mock.edgeInternal);
  assert.equal(w1.env.FV_DISPATCH_FAMILIES, "fake");
  assert.equal(w1.env.FV_D1_DATABASE_ID, "d1-edge-staging");
  assert.equal(w1.env.FV_WORKER_DIRECT, undefined);
  assert.equal(w1.env.FV_R2_BUCKET, undefined);
  assert.match(w1.payload.dockerStartCmd[0], /FV_DISPATCH_ENDPOINT=.*\[watchdog\]/s);
  // The cluster card and the families view.
  const detail = (await call(`/api/clusters/${id}`, { headers: T() })).j;
  assert.equal(detail.edge_url, edge);
  const gv = (await call(`/api/clusters/${id}/gateway`, { headers: T() })).j;
  assert.equal(gv.edge, true, "the old /gateway path still answers (an alias of /front)");
  assert.equal(gv.url, edge);
  assert.equal(gv.workers[0].front.ready, true, JSON.stringify(gv));
  // A second edge cluster is refused while this one runs.
  const r2 = await call("/api/clusters", { method: "POST", body: { spec: { name: "edgy2", template: "tiny-cpu", image: { channel: "stable" }, cap_s: 3600, control_plane: "edge" } }, headers: T() });
  assert.equal(r2.status, 201);
  assert.equal((await call(`/api/clusters/${r2.j.cluster.id}/start`, { method: "POST", body: {}, headers: T() })).status, 202);
  const op2 = await waitOp(r2.j.cluster.id, "up");
  assert.equal(op2.status, "failed");
  assert.match(op2.error, /one edge cluster at a time/);
  // Keys and the admin token are the edge's.
  const tok = await call(`/api/clusters/${id}/admin-token`, { method: "POST", body: {}, headers: T() });
  assert.equal(tok.j.admin_token, mock.edgeAdmin);
  assert.equal(tok.j.console, `${edge}/console/admin`);
  const k = await call(`/api/clusters/${id}/mint-key`, { method: "POST", body: { name: "edge-user" }, headers: T() });
  assert.equal(k.status, 201, JSON.stringify(k.j));
  assert.match(k.j.api_key, /^fv_edge_key_/);
  const kid = k.j.key.id;
  assert.ok((await call(`/api/clusters/${id}/keys`, { headers: T() })).j.keys.some((x) => x.id === kid));
  const epoch = mock.edgeEpoch;
  const rv = await call(`/api/clusters/${id}/keys/${kid}`, { method: "DELETE", headers: T() });
  assert.equal(rv.status, 200, JSON.stringify(rv.j));
  assert.deepEqual(rv.j.applied, ["edge"]);
  assert.equal(mock.edgeEpoch, epoch + 1, "one revoke at the edge");
  // Scale up and down (the drain goes to the front with the edge's internal token).
  assert.equal((await call(`/api/clusters/${id}/scale`, { method: "POST", body: { pool: "fake", count: 2 }, headers: T() })).status, 202);
  op = await waitOp(id, "scale");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mine().length, 2);
  const w2 = mine().find((p) => p.id !== w1.id);
  assert.equal(w2.env.FV_INTERNAL_TOKEN, mock.edgeInternal);
  assert.equal((await call(`/api/clusters/${id}/scale`, { method: "POST", body: { pool: "fake", count: 1 }, headers: T() })).status, 202);
  op = await waitOp(id, "scale");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mine().length, 1);
  assert.ok(mock.drained.length >= 1);
  // Roll to a commit: a new front, then the old one goes.
  const old = mine()[0].id;
  assert.equal((await call(`/api/clusters/${id}/roll`, { method: "POST", body: { target: "abcdef1" }, headers: T() })).status, 202);
  op = await waitOp(id, "roll");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mine().length, 1);
  assert.notEqual(mine()[0].id, old);
  assert.equal(mine()[0].env.FV_DISPATCH_FRONT, "1");
  // Stop: no pod left; the other edge cluster may start now.
  assert.equal((await call(`/api/clusters/${id}/stop`, { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(id, "down")).status, "done");
  assert.equal(mine().length, 0);
  assert.ok(!JSON.stringify(mock.edgeCalls).includes("gateway/pools"));
});

await step("build pods: policy, placement, up (create / reuse / start / replace), token, costs, runner, busy guard, CI wake, queued-job wake, backstop", async () => {
  mock.balance = 50;
  const A = { cookie, "x-csrf-token": csrf };
  const bpRows = () => [...mock.pods.values()].filter((p) => p.env?.FV_BUILD_TOKEN_SHA256);
  // Off by default: nothing is created.
  assert.equal((await call("/api/build-pods/up", { method: "POST", body: {}, headers: T() })).status, 403);
  const pol = await call("/api/build-pods/policy", { method: "PUT", body: { policy: { enabled: true, max_pods: 2 } }, headers: A });
  assert.equal(pol.j.policy.enabled, true);
  // Plan: cpu5c-32 has no stock; cpu3c-32 in EU-RO-1 (High) first.
  const plan = (await call("/api/build-pods/plan", { headers: T() })).j;
  assert.deepEqual(plan.candidates.slice(0, 3).map((c) => `${c.flavor}-${c.vcpu}@${c.dc}`), ["cpu3c-32@EU-RO-1", "cpu3c-32@EUR-IS-1", "cpu5c-16@EUR-IS-1"]);
  assert.equal(plan.server.image, "ghcr.io/zaitrarrio/fastvideo-rs-build-base:bb-0123456789abcdef");
  // up: EU-RO-1 has no instances after all -> the next candidate.
  // …and Runpod caps the container disk below the 200 GB asked: one retry with its cap (#41's fix, now here).
  mock.noStockDcs.add("EU-RO-1");
  mock.cpuDiskCap = 160;
  const up1 = await call("/api/build-pods/up", { method: "POST", body: {}, headers: T() });
  assert.equal(up1.status, 201, JSON.stringify(up1.j));
  mock.noStockDcs.clear();
  mock.cpuDiskCap = 0;
  assert.equal(up1.j.pod.disk_gb, 160);
  assert.equal(up1.j.action, "created");
  assert.match(up1.j.token, /^[0-9a-f]{64}$/);
  const bp1 = up1.j.pod;
  assert.equal(bp1.dc, "EUR-IS-1");
  assert.match(bp1.name, /^fv-build-eu-[0-9a-f]{6}$/);
  const rp = mock.pods.get(bp1.pod_id);
  assert.deepEqual([rp.payload.computeType, rp.payload.cpuFlavorIds[0], rp.payload.vcpuCount, rp.payload.dataCenterIds[0]], ["CPU", "cpu3c", 32, "EUR-IS-1"]);
  const { createHash } = await import("node:crypto");
  const { gunzipSync } = await import("node:zlib");
  assert.equal(rp.env.FV_BUILD_TOKEN_SHA256, createHash("sha256").update(up1.j.token).digest("hex"));
  assert.equal(gunzipSync(Buffer.from(rp.env.FV_BUILD_SERVER_B64, "base64")).toString(), mock.repoFiles["scripts/dev/build-pod-server.py"]);
  assert.equal(rp.env.FV_BUILD_ROOT, "/root/fvb-cache", "no volume: caches on the container disk");
  assert.ok(!JSON.stringify(up1.j).includes("FV_BUILD_SERVER_B64"), "the pod env is never returned");
  // A second up reuses it; the token needs an admin token.
  const up2 = await call("/api/build-pods/up", { method: "POST", body: {}, headers: T() });
  assert.equal(up2.j.action, "reused");
  assert.equal(up2.j.pod.id, bp1.id);
  const ro = (await call("/api/tokens", { method: "POST", body: { name: "bp-viewer", scope: "read" }, headers: A })).j.token;
  assert.equal((await call(`/api/build-pods/${bp1.id}/token`, { headers: { authorization: `Bearer ${ro}` } })).status, 403);
  assert.equal((await call(`/api/build-pods/${bp1.id}`, { headers: { authorization: `Bearer ${ro}` } })).j.pod.phase, "ready");
  // The cron: owner build-pod:<name> in the ledger; the runner registered with a 1-h token (never the PAT).
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  const pods = (await call("/api/pods", { headers: T() })).j.pods;
  assert.equal(pods.find((p) => p.pod_id === bp1.pod_id).owner, `build-pod:${bp1.name}`);
  assert.equal(mock.runnerRegs.length, 1);
  assert.deepEqual([mock.runnerRegs[0].token, mock.runnerRegs[0].labels, mock.runnerRegs[0].name, mock.runnerRegs[0].repo], ["<reg>", "fv-build,fv-build-eu", `fv-build-${bp1.pod_id}`, "zaitrarrio/fastvideo-rs"]);
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  let ov = (await call("/api/build-pods", { headers: T() })).j;
  assert.equal(ov.pods.find((p) => p.id === bp1.id).runner.state, "running");
  assert.equal(ov.secrets.runner_pat, true);
  // Busy: stop refuses with jobs active, force stops; the runner goes from GitHub.
  mock.bp[bp1.pod_id] = { ...mock.bp[bp1.pod_id], jobs_active: 2 };
  const busy = await call(`/api/build-pods/${bp1.id}/stop`, { method: "POST", body: {}, headers: T() });
  assert.equal(busy.status, 409);
  assert.match(busy.j.error, /2 job\(s\) active/);
  assert.equal((await call(`/api/build-pods/${bp1.id}/stop`, { method: "POST", body: { force: true }, headers: T() })).j.pod.state, "stopped");
  assert.equal(mock.pods.get(bp1.pod_id).desiredStatus, "EXITED");
  assert.ok(!mock.runners.some((r) => r.name === `fv-build-${bp1.pod_id}`), "runner deregistered");
  mock.bp[bp1.pod_id].jobs_active = 0;
  // up starts the stopped (current) pod; the cron registers its runner again.
  const up3 = await call("/api/build-pods/up", { method: "POST", body: {}, headers: T() });
  assert.equal(up3.j.action, "started");
  assert.equal(mock.pods.get(bp1.pod_id).desiredStatus, "RUNNING");
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.equal(mock.runnerRegs.length, 2);
  // main's server changes: the running pod is left alone (outdated), replaced only once stopped.
  mock.repoFiles["scripts/dev/build-pod-server.py"] += "# v2\n";
  assert.equal((await call("/api/build-pods/up", { method: "POST", body: {}, headers: T() })).j.action, "reused");
  ov = (await call("/api/build-pods", { headers: T() })).j;
  assert.equal(ov.pods.find((p) => p.id === bp1.id).outdated, true);
  await call(`/api/build-pods/${bp1.id}/stop`, { method: "POST", body: {}, headers: T() });
  const up4 = await call("/api/build-pods/up", { method: "POST", body: {}, headers: T() });
  assert.equal(up4.j.action, "created");
  assert.deepEqual(up4.j.replaced, [bp1.pod_id]);
  assert.ok(!mock.pods.has(bp1.pod_id), "the stopped outdated pod was deleted");
  const bp2 = up4.j.pod;
  // CI: a ci token reaches only /api/ci/*; an idle runner -> pod.
  const ci = (await call("/api/tokens", { method: "POST", body: { name: "gh-ci", scope: "ci" }, headers: A })).j.token;
  const CI = { authorization: `Bearer ${ci}` };
  assert.equal((await call("/api/overview", { headers: CI })).status, 403);
  assert.equal((await call(`/api/build-pods/${bp2.id}/token`, { headers: CI })).status, 403);
  await call("/api/collect", { method: "POST", body: {}, headers: T() }); // registers bp2's runner
  let ans = (await call("/api/ci/build-runner", { method: "POST", body: { workflow: "tools-release.yml", run_id: 7 }, headers: CI })).j;
  assert.equal(ans.builder, "pod", JSON.stringify(ans));
  // No idle runner: the pod's runner is busy -> a second pod is woken ("wait"); at max_pods -> github.
  mock.runners.forEach((r) => (r.busy = true));
  ans = (await call("/api/ci/build-runner", { method: "POST", body: {}, headers: CI })).j;
  assert.equal(ans.builder, "wait", JSON.stringify(ans));
  assert.equal(bpRows().filter((p) => p.desiredStatus === "RUNNING").length, 2);
  ans = (await call("/api/ci/build-runner", { method: "POST", body: {}, headers: CI })).j;
  assert.equal(ans.builder, "wait", "the woken pod is reused while its runner comes up");
  mock.runners.forEach((r) => (r.busy = false));
  // Stop everything; a queued fv-build job wakes a pod from the cron.
  ov = (await call("/api/build-pods", { headers: T() })).j;
  for (const p of ov.pods.filter((x) => x.state === "running")) assert.equal((await call(`/api/build-pods/${p.id}/stop`, { method: "POST", body: { force: true }, headers: T() })).status, 200);
  assert.equal(bpRows().filter((p) => p.desiredStatus === "RUNNING").length, 0);
  mock.ghQueued.push({ run_id: 4242, path: ".github/workflows/tools-release.yml", jobs: [{ id: 1, status: "completed", labels: ["ubuntu-latest"] }, { id: 2, status: "queued", labels: ["self-hosted", "fv-build"] }] });
  const col = (await call("/api/collect", { method: "POST", body: {}, headers: T() })).j;
  assert.ok(col.actions.some((a) => /started for 1 queued fv-build job/.test(a)), JSON.stringify(col.actions));
  mock.ghQueued.length = 0;
  // Backstop: idle far past its own idle stop -> the cron stops it (alert build_pod).
  const live = bpRows().find((p) => p.desiredStatus === "RUNNING");
  mock.bp[live.id] = { ...mock.bp[live.id], idle_s: 3 * 3600 };
  const col2 = (await call("/api/collect", { method: "POST", body: {}, headers: T() })).j;
  assert.ok(col2.actions.some((a) => /idle 180 min/.test(a)), JSON.stringify(col2.actions));
  assert.equal(live.desiredStatus, "EXITED");
  const al = (await call("/api/alerts", { headers: T() })).j.alerts;
  assert.ok(al.some((a) => a.kind === "build_pod" && a.target === live.id));
  // Costs land under build-pod:<name>; audit has the lifecycle.
  const costs = (await call("/api/costs?days=1", { headers: T() })).j;
  assert.ok(JSON.stringify(costs).includes("build-pod:fv-build-eu-"), "build pod spend by owner");
  const acts = new Set((await call("/api/audit?limit=500", { headers: T() })).j.audit.map((x) => x.action));
  for (const a of ["build_pod.create", "build_pod.stop", "build_pod.start", "build_pod.delete", "build_pod.token", "build_pod.wake", "build_pods.policy", "build_pod.ci_wake"]) assert.ok(acts.has(a), `audit has ${a}`);
  // Delete what is left (stopped pods), so later steps see no build pods.
  ov = (await call("/api/build-pods", { headers: T() })).j;
  for (const p of ov.pods.filter((x) => x.state !== "deleted")) assert.equal((await call(`/api/build-pods/${p.id}?force=1`, { method: "DELETE", headers: T() })).j.pod.state, "deleted");
  assert.equal(bpRows().length, 0);
});

await step("standalone pod: launch, the same up (direct, watchdog boot), costs under pod:<name>, status and logs as JSON, stop, start, delete", async () => {
  await d1Exec(w.dir, "DELETE FROM operations");
  const before = new Set(mock.pods.keys());
  const mine = () => [...mock.pods.values()].filter((p) => !before.has(p.id));
  const launch = { name: "solo1", variant: "cpu", compute: "CPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], channel: "stable", deadline_min: 30, env: { SOLO_FOO: "bar" } };
  const r = await call("/api/standalone", { method: "POST", body: launch, headers: T() });
  assert.equal(r.status, 201, JSON.stringify(r.j));
  const id = r.j.pod.id;
  let op = await waitOp(id, "up");
  assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-5)));
  assert.equal(mine().length, 1);
  const p1 = mine()[0];
  assert.match(p1.name, /^fv-pod-solo1-/);
  assert.equal(p1.env.FV_WORKER_DIRECT, "1");
  assert.equal(p1.env.SOLO_FOO, "bar");
  assert.ok(Number(p1.env.FV_CLUSTER_DEADLINE) > Date.now() / 1000, "its own backstop deadline");
  assert.match(p1.payload.dockerStartCmd[0], /\[watchdog\]/);
  assert.match(p1.payload.dockerStartCmd[0], /\[fv-boot\] start/);
  // Listed as a standalone pod with its pod, not among clusters.
  const list = (await call("/api/standalone", { headers: T() })).j.pods;
  assert.deepEqual(list.map((x) => [x.name, x.pod?.pod_id]), [["solo1", p1.id]]);
  assert.ok(!(await call("/api/clusters", { headers: T() })).j.clusters.some((c) => c.name === "solo1"));
  // Costs and owner: the same ledger as cluster pods.
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  const pods = (await call("/api/pods", { headers: T() })).j.pods;
  assert.equal(pods.find((x) => x.pod_id === p1.id)?.owner, "pod:solo1");
  const costs = (await call("/api/costs?days=1", { headers: T() })).j;
  assert.ok(costs.by_owner.some((x) => x.owner === "pod:solo1"), JSON.stringify(costs.by_owner));
  const st = (await call(`/api/pods/${p1.id}/status`, { headers: T() })).j;
  assert.equal(st.kind, "standalone");
  assert.ok(st.boot_timeline.some((x) => x.phase === "ready (at the edge / health)" && x.at), JSON.stringify(st.boot_timeline));
  const bt = (await call(`/api/pods/${p1.id}/boot`, { headers: T() })).j;
  assert.ok(bt.boot.t.create && bt.boot.t.ready);
  const lg = (await call(`/api/pods/${p1.id}/logs?source=control`, { headers: T() })).j;
  assert.ok(lg.lines.some((l) => /^boot: ready at \+/.test(l.msg)), JSON.stringify(lg.lines));
  // Stop deletes the pod and keeps the definition; start makes a new one; delete with a pod stops it first.
  assert.equal((await call("/api/standalone/solo1/stop", { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(id, "down")).status, "done");
  assert.equal(mine().filter((p) => mock.pods.has(p.id)).length, 0);
  assert.equal((await call("/api/standalone/solo1", { headers: T() })).j.pod.pod, null);
  assert.equal((await call("/api/standalone/solo1/start", { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(id, "up")).status, "done");
  assert.equal((await call("/api/standalone/solo1", { method: "DELETE", headers: T() })).status, 202);
  for (let i = 0; i < 120 && (await call("/api/standalone/solo1", { headers: T() })).status !== 404; i++) await sleep(500);
  assert.equal((await call("/api/standalone/solo1", { headers: T() })).status, 404, "the definition goes once the pod is gone");
  assert.equal([...mock.pods.values()].filter((p) => !before.has(p.id)).length, 0);
});

await step("GMI / Brev standalone pods: checks, create → running → tunnel report → health → delete, budget, failed boot, only ours", async () => {
  await d1Exec(w.dir, "DELETE FROM operations");
  const g = mock.gmi;
  const b = mock.brev;
  const fake = { variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], channel: "stable", deadline_min: 20 };
  const issues = async (body) => (await call("/api/standalone/validate", { method: "POST", body, headers: T() })).j;
  // The provider's fields, path-anchored: a product the owner did not allow, Runpod fields mixed in, a Hub download without approval.
  let v = await issues({ name: "gmi1", provider: "gmi", provider_gpu: "container.nope.x1", ...fake });
  assert.ok(v.issues.some((i) => i.path[0] === "provider_gpu" && /not allowed on GMI Cloud/.test(i.message)), JSON.stringify(v.issues));
  v = await issues({ name: "gmi1", provider: "gmi", provider_gpu: "container.h200.x1", gpu_types: ["NVIDIA H200"], region: "eu", ...fake });
  assert.ok(v.issues.some((i) => i.path[0] === "gpu_types") && v.issues.some((i) => i.path[0] === "region"), JSON.stringify(v.issues));
  v = await issues({ name: "gmi1", provider: "gmi", ...fake });
  assert.ok(v.issues.some((i) => i.path[0] === "provider_gpu" && /required/.test(i.message)));
  v = await issues({ name: "gmih3", provider: "gmi", provider_gpu: "container.h200.x1", preset: "h3-turbo", weights_source: "hub", deadline_min: 20 });
  assert.ok(v.issues.some((i) => i.path[0] === "weights_download_approved" && /owner's approval/.test(i.message)), JSON.stringify(v.issues));
  v = await issues({ name: "gmih3", provider: "gmi", provider_gpu: "container.h200.x1", preset: "h3-turbo", weights_source: "hub", weights_download_approved: true, deadline_min: 20 });
  assert.ok(v.issues.some((i) => i.path[0] === "weights_download_approved" && /FV_HUB_DOWNLOADS_APPROVED/.test(i.message)), "the Worker-level approval is needed too");
  v = await issues({ name: "gmih3", provider: "gmi", provider_gpu: "container.h200.x1", preset: "h3-turbo", deadline_min: 20 });
  assert.ok(v.issues.some((i) => i.path[0] === "weights_source" && /models need weights/.test(i.message)), "a real model with no weights");
  v = await issues({ name: "gmib", provider: "gmi", provider_gpu: "container.b200.x1", ...fake });
  assert.ok(v.issues.some((i) => i.path[0] === "max_gpu_dph" && /\$9\/hr/.test(i.message)), JSON.stringify(v.issues));
  v = await issues({ name: "gmi1", provider: "gmi", provider_gpu: "container.h200.x1", ...fake });
  assert.equal(v.ok, true, JSON.stringify(v.issues));
  assert.ok(!g.calls.some((c) => c.method !== "GET"), "validation creates nothing");

  // GMI: a fake-engine pod end to end.
  let r = await call("/api/standalone", { method: "POST", body: { name: "gmi1", provider: "gmi", provider_gpu: "container.h200.x1", ...fake }, headers: T() });
  assert.equal(r.status, 201, JSON.stringify(r.j));
  const gid = r.j.pod.id;
  let op = await waitOp(gid, "up", 90000);
  assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-6)));
  const gc = g.containers.find((c) => c.name.startsWith("fv-pod-gmi1-"));
  assert.ok(gc, "container created");
  assert.ok(g.calls.every((c) => c.auth), "every GMI call carries the key");
  assert.equal(g.templates.length, 1);
  assert.match(g.templates[0].path, /fastvideo-rs-serve.*@sha256:/, "the template is the resolved image digest");
  assert.equal(gc.command, "bash");
  assert.match(gc.args[1], /\[fv-boot\] start/);
  assert.match(gc.args[1], /cloudflared tunnel --no-autoupdate --url http:\/\/127\.0\.0\.1:8000/);
  assert.match(gc.args[1], /53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134/);
  assert.deepEqual(gc.ports, [{ containerPort: 8000, protocol: "TCP" }]);
  const genv = Object.fromEntries(gc.envs.map((e) => [e.name, e.value]));
  const gkey = `gmi:${gc.name}`;
  assert.equal(genv.FV_POD_ID, gkey);
  assert.equal(genv.FV_WORKER_ID, gkey);
  assert.equal(genv.FV_PROVIDER, "gmi");
  assert.equal(genv.FV_WORKER_DIRECT, "1");
  assert.equal(genv.FV_WEIGHTS_SOURCE, "none");
  assert.ok(!Object.values(genv).some((x) => /RUNPOD_SECRET/.test(x)), "no Runpod secret reference");
  assert.ok(!Object.values(genv).includes(SECRETS.RUNPOD_API_KEY), "the Runpod key never leaves Runpod");
  assert.equal(genv.FV_BACKSTOP_API_KEY, SECRETS.GMI_API_KEY, "its own self-delete key");
  // Named tunnels are not configured on this fv-control: the quick-tunnel fallback, no Cloudflare tunnel call.
  assert.equal(genv.FV_TUNNEL_TOKEN, undefined);
  assert.equal(genv.FV_TUNNEL_URL, undefined);
  assert.equal(mock.cf.calls.length, 0, "no tunnel / DNS call without the tunnel settings");
  assert.ok(Number(genv.FV_CLUSTER_DEADLINE) > Date.now() / 1000);
  assert.ok(mock.reports.some((x) => x.key === gkey && x.phase === "tunnel" && x.status === 200), JSON.stringify(mock.reports));
  let sp = (await call("/api/standalone/gmi1", { headers: T() })).j.pod;
  assert.equal(sp.pod.pod_id, gkey);
  assert.equal(sp.pod.url, `http://127.0.0.1:${mock.port}/pod/${gkey}`);
  assert.equal(sp.definition.provider, "gmi");
  assert.equal(sp.definition.provider_gpu, "container.h200.x1");
  assert.equal(sp.pod.cost_per_hr, 3.2, "the products price (320 / GMI_PRICE_DIVISOR 100)");
  // A report for a pod of another cluster, or with a non-tunnel URL, is refused.
  assert.equal((await call("/ingest/v1/endpoint", { method: "POST", body: { pod: gkey, phase: "tunnel", url: "https://evil.example.com" }, headers: { authorization: `Bearer ${genv.FV_ENDPOINT_REPORT_TOKEN}` } })).status, 400);
  assert.equal((await call("/ingest/v1/endpoint", { method: "POST", body: { pod: gkey, phase: "tunnel", url: sp.pod.url }, headers: { authorization: "Bearer nope" } })).status, 401);
  // The collector: our pod under pod:gmi1 at its $/hr; the stray fv- container is reported, the foreign one ignored; neither touched.
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  const pods = Object.fromEntries((await call("/api/pods", { headers: T() })).j.pods.map((p) => [p.pod_id, p]));
  assert.equal(pods[gkey]?.provider, "gmi");
  assert.equal(pods[gkey]?.owner, "pod:gmi1");
  assert.ok(!Object.keys(pods).some((k) => k.includes("someone-else")), "a foreign container is not listed");
  assert.equal(pods["gmi:fv-pod-stray-0101000000"]?.owner, "external:gmi");
  const al = (await call("/api/alerts", { headers: T() })).j.alerts;
  assert.ok(al.some((a) => a.kind === "gmi_orphan" && /not touched/.test(a.message)), JSON.stringify(al.map((a) => a.kind)));
  const costs = (await call("/api/costs?days=1", { headers: T() })).j;
  assert.ok(costs.by_owner.some((x) => x.owner === "pod:gmi1" && x.usd > 0), JSON.stringify(costs.by_owner));
  const lg = (await call(`/api/pods/${encodeURIComponent(gkey)}/provider-logs`, { headers: T() })).j;
  assert.ok(lg.lines.some((l) => /\[fv-boot\] start/.test(l)) && lg.lines.some((l) => /\[redacted\]/.test(l)), JSON.stringify(lg));
  const prov = (await call("/api/providers", { headers: T() })).j.providers.find((p) => p.id === "gmi");
  assert.equal(prov.budget_usd, 50);
  assert.deepEqual([prov.tunnel.mode, prov.tunnel.warning, prov.tunnel.error], ["quick", null, null], JSON.stringify(prov.tunnel));
  assert.ok(prov.month_usd > 0 && prov.running_dph === 3.2, JSON.stringify(prov));
  const offers = (await call("/api/providers/gmi/offers", { headers: T() })).j.offers;
  assert.deepEqual(offers.map((o) => [o.gpu, o.usd_per_hr, o.in_stock]), [["container.h200.x1", 3.2, true], ["container.b200.x1", 9, true]]);
  // Stop deletes our container only; delete removes the definition.
  assert.equal((await call("/api/standalone/gmi1/stop", { method: "POST", body: {}, headers: T() })).status, 202);
  op = await waitOp(gid, "down");
  assert.equal(op.status, "done", op.error);
  assert.ok(!g.containers.some((c) => c.name === gc.name), "our container is gone");
  assert.deepEqual(g.containers.map((c) => c.name).sort(), ["fv-pod-stray-0101000000", "someone-else"], "nothing else touched");
  assert.ok(g.calls.filter((c) => c.method === "DELETE").every((c) => c.path === `/containers/${gc.id}`));
  assert.equal((await call("/api/standalone/gmi1", { method: "DELETE", headers: T() })).status, 200, "no pod left: the definition goes at once");

  // A boot that reports a failure (a weights or tunnel error): the pod is failed and deleted at once.
  g.boot = "fail";
  r = await call("/api/standalone", { method: "POST", body: { name: "gmifail", provider: "gmi", provider_gpu: "container.h200.x1", ...fake }, headers: T() });
  assert.equal(r.status, 201);
  op = await waitOp(r.j.pod.id, "up", 90000);
  assert.equal(op.status, "failed");
  assert.ok(op.log.some((l) => /boot failed.*hub 404/.test(l.msg)), JSON.stringify({ log: op.log.map((l) => l.msg), error: op.error, reports: mock.reports }));
  assert.ok(!g.containers.some((c) => c.name.startsWith("fv-pod-gmifail-")), "deleted after the failure");
  g.boot = "ok";
  await call("/api/standalone/gmifail", { method: "DELETE", headers: T() });

  // Brev: the budget refuses a launch whose projection is over it ($1/hr × 10 h > $2), before any VM.
  r = await call("/api/standalone", { method: "POST", body: { name: "brevbig", provider: "brev", provider_gpu: "g5.xlarge-test", ...fake, deadline_min: 600 }, headers: T() });
  assert.equal(r.status, 201);
  op = await waitOp(r.j.pod.id, "up");
  assert.equal(op.status, "failed");
  assert.match(op.error, /NVIDIA Brev: .* over the budget \$2/);
  assert.equal(b.workspaces.length, 0, "no VM before the budget check");
  await call("/api/standalone/brevbig", { method: "DELETE", headers: T() });
  // Brev: a 20-minute fake-engine VM end to end, launched with the CLI.
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const dir = mkdtempSync(`${tmpdir()}/fvc-cli-`);
  writeFileSync(`${dir}/token`, token, { mode: 0o600 });
  const cli = async (...args) => (await promisify(execFile)("bash", [new URL("../../../scripts/serve/fv-control.sh", import.meta.url).pathname, ...args], { env: { ...process.env, FV_CONTROL_URL: B, FV_CONTROL_TOKEN_FILE: `${dir}/token`, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" } })).stdout;
  assert.match(await cli("providers"), /^gmi\ton\t\tbudget \$50, month \$[\d.]+\tcontainer\.h200\.x1,container\.b200\.x1$/m);
  await cli("pod", "launch", "brev1", "--provider", "brev", "--gpu", "g5.xlarge-test", "--fake", "--channel", "stable", "--deadline-min", "20");
  const bid = (await call("/api/standalone/brev1", { headers: T() })).j.pod.id;
  assert.equal((await call("/api/standalone/brev1", { headers: T() })).j.pod.definition.provider_gpu, "g5.xlarge-test", "--gpu is the instance type on brev");
  op = await waitOp(bid, "up", 90000);
  assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-6)));
  assert.equal(b.workspaces.length, 1);
  const ws = b.workspaces[0];
  assert.ok(b.calls.every((c) => c.auth) && b.calls.some((c) => c.path === `/organizations/${mock.brevOrg}/workspaces` && c.method === "POST"));
  // The startup script only installs the per-boot bootstrap; the launch is the run script it fetched from fv-control.
  assert.match(ws.startupScript, /systemctl enable fv-boot\.service/);
  assert.ok(!ws.startupScript.includes("FV_POD_ID") && !ws.startupScript.includes("docker run"), "no env or launch in the startup script");
  const boot1 = b.boots.find((x) => x.ws === ws.id);
  assert.equal(boot1?.status, 200, JSON.stringify(boot1));
  assert.match(boot1.run, /docker run -d --name fv-serve --restart no --gpus all --network host --env-file/);
  assert.match(boot1.run, /fastvideo-rs-serve.*@sha256:/);
  assert.match(boot1.run, /shutdown -h now/, "the host watchdog powers the VM off after the deadline");
  // brev-cli main's create body (the old one got 400 "Legacy workspace version unsupported" live).
  assert.equal(ws.body.workspaceVersion, "v1");
  assert.equal(ws.body.cloudCredId, "devplane-brev-1-credential", "the type's cloud_cred_id from the listing");
  assert.deepEqual([ws.body.workspaceTemplateId, ws.body.workspaceClassId, ws.body.diskStorage, ws.body.isStoppable], ["4nbb4lg2s", "2x8", "200Gi", true], "a stoppable type: isStoppable true");
  assert.ok(ws.body.startupScript === "" && ws.body.vmOnlyMode === false && ws.body.description === "" && Array.isArray(ws.body.applications), "Go zero values present; the script rides in vmBuild.lifeCycleScriptAttr");
  assert.ok(b.calls.some((c) => c.method === "GET" && c.path === `/instances/alltypesavailable/${mock.brevOrg}`));
  const bkey = `brev:${ws.name}`;
  const benv = mock.pods.get(bkey)?.env || {};
  assert.equal(benv.FV_POD_ID, bkey);
  assert.equal(benv.FV_PROVIDER, "brev");
  assert.ok(!("FV_BACKSTOP_API_KEY" in benv), "no account key in a Brev VM");
  sp = (await call("/api/standalone/brev1", { headers: T() })).j.pod;
  assert.equal(sp.pod.pod_id, bkey);
  assert.equal(sp.pod.cost_per_hr, 1, "BREV_PRICES");
  // The host script's reports: its facts and container_started in the provider logs and the boot timeline; heartbeats keep the phase.
  const plg = (await call(`/api/pods/${encodeURIComponent(bkey)}/provider-logs`, { headers: T() })).j;
  assert.ok(plg.lines.some((l) => /^\[report .*\] host: user=ubuntu uid=1000 sudo=yes docker=\/usr\/bin\/docker .*gpu=\[GPU 0: NVIDIA A10G\]/.test(l)), JSON.stringify(plg.lines));
  assert.ok(plg.lines.some((l) => /\] container_started: c0ffee000000/.test(l)) && plg.lines.some((l) => /\] heartbeat \(last\): container: no container/.test(l)), JSON.stringify(plg.lines));
  assert.deepEqual(plg.reports.map((x) => x.phase), ["run_fetched", "host", "container_started", "tunnel", "serve_ready"], "the host watcher saw /health 200");
  const btl = (await call(`/api/pods/${encodeURIComponent(bkey)}/boot`, { headers: T() })).j.timeline;
  const hostRow = btl.find((r) => r.phase === "host script (Brev VM)");
  assert.ok(hostRow?.at && /user=ubuntu/.test(hostRow.detail || ""), JSON.stringify(btl));
  assert.ok(btl.find((r) => r.phase === "tunnel reported")?.at, JSON.stringify(btl));
  assert.equal((await call("/ingest/v1/endpoint", { method: "POST", body: { pod: bkey, phase: "nonsense" }, headers: { authorization: `Bearer ${benv.FV_ENDPOINT_REPORT_TOKEN}` } })).status, 400);
  assert.equal((await call("/api/standalone/brev1", { method: "DELETE", headers: T() })).status, 202);
  for (let i = 0; i < 120 && (await call("/api/standalone/brev1", { headers: T() })).status !== 404; i++) await sleep(500);
  assert.equal(b.workspaces.length, 0, "the VM is deleted");
  assert.ok(b.calls.filter((c) => c.method === "DELETE").every((c) => c.path === `/workspaces/${ws.id}`));
  // A host script that fails (here docker run without a GPU driver): the boot fails at once with its reason, the VM goes.
  b.boot = "hostfail";
  await cli("pod", "launch", "brevhf", "--provider", "brev", "--gpu", "g5.xlarge-test", "--fake", "--channel", "stable", "--deadline-min", "20");
  const hid = (await call("/api/standalone/brevhf", { headers: T() })).j.pod.id;
  op = await waitOp(hid, "up", 90000);
  b.boot = "ok";
  assert.equal(op.status, "failed");
  assert.ok(op.log.some((l) => /FAILED \(host failed\): docker run \(exit 125\): .*could not select device driver/.test(l.msg)), JSON.stringify(op.log.map((l) => l.msg)));
  assert.equal(b.workspaces.length, 0, "deleted after the host failure");
  await call("/api/standalone/brevhf", { method: "DELETE", headers: T() });
  // The container exits after its start (the host watcher's container_exited, with its log): failed at once, deleted.
  b.boot = "exit";
  await cli("pod", "launch", "brevex", "--provider", "brev", "--gpu", "g5.xlarge-test", "--fake", "--channel", "stable", "--deadline-min", "20");
  const xid = (await call("/api/standalone/brevex", { headers: T() })).j.pod.id;
  op = await waitOp(xid, "up", 90000);
  b.boot = "ok";
  assert.equal(op.status, "failed");
  assert.ok(op.log.some((l) => /FAILED \(container exited\): container exited exit=2 .*fv-serve: config: auth\.key_store = d1/.test(l.msg)), JSON.stringify(op.log.map((l) => l.msg)));
  // Without FV_PROVIDER_SECRET_ENV's D1 settings the key store is memory (else fv-serve refuses to start), said in the log.
  assert.ok(op.log.some((l) => /NOTE: FV_KEY_STORE=memory: no D1 settings/.test(l.msg)), JSON.stringify(op.log.map((l) => l.msg)));
  assert.equal(b.workspaces.length, 0, "deleted after the exit");
  await call("/api/standalone/brevex", { method: "DELETE", headers: T() });
});

await step("Brev keep-on-stop: park on stop, warm restart (no download), restart failure / timeout, non-stoppable delete, limits, budget, only ours", async () => {
  // A second fv-control with Hub downloads approved (the simulated VMs never download anything), the live
  // instance-type list as prices (no BREV_PRICES), a short restart timeout.
  const w2 = await startWorker(mock, {
    ...SECRETS,
    BREV_ORG_ID: mock.brevOrg,
    BREV_INSTANCE_TYPES: "g5.xlarge-test,a100-test",
    BREV_BUDGET_USD: "100",
    BREV_RESTART_TIMEOUT_S: "30",
    FV_HUB_DOWNLOADS_APPROVED: "1",
    OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  });
  const b = mock.brev;
  b.workspaces = [];
  b.boots = [];
  // Not ours: a foreign workspace and an fv- named one fv-control never recorded. Never stopped, started or deleted.
  b.workspaces.push({ id: "wsforeign", name: "someone-else", status: "STOPPED", instanceType: "g5.xlarge-test", disk: {}, startupScript: "", body: {} });
  b.workspaces.push({ id: "wsstray", name: "fv-pod-stray-0101000000", status: "STOPPED", instanceType: "g5.xlarge-test", disk: {}, startupScript: "", body: {} });
  try {
    const call2 = async (path, { method = "GET", body, headers = {} } = {}) => {
      const r = await fetch(w2.url + path, { method, headers: { ...(body !== undefined ? { "content-type": "application/json" } : {}), ...headers }, body: body === undefined ? undefined : JSON.stringify(body) });
      const text = await r.text();
      bodies.push(text);
      let j = null;
      try { j = JSON.parse(text); } catch {}
      return { status: r.status, j, text };
    };
    const lr = await fetch(w2.url + "/api/auth/login", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ passphrase: PASSPHRASE }) });
    const ck = lr.headers.get("set-cookie").split(";")[0];
    const cs = (await lr.json()).csrf;
    const tok = (await call2("/api/tokens", { method: "POST", body: { name: "ci2", scope: "admin" }, headers: { cookie: ck, "x-csrf-token": cs } })).j.token;
    const H = { authorization: `Bearer ${tok}` };
    const wait2 = async (cluster, kind, timeoutMs = 120000) => {
      const t = Date.now();
      while (Date.now() - t < timeoutMs) {
        const op = (await call2(`/api/clusters/${cluster}/ops`, { headers: H })).j.operations.find((o) => o.kind === kind && !/is running/.test(o.error || ""));
        if (op && op.status !== "running") return op;
        await sleep(500);
      }
      const ops = (await call2(`/api/clusters/${cluster}/ops`, { headers: H })).j.operations;
      throw new Error(`operation ${kind} did not finish: ${JSON.stringify(ops.map((o) => [o.kind, o.status, (o.log || []).slice(-8).map((l) => l.msg)]))}; boots ${JSON.stringify(mock.brev.boots.slice(-2).map(({ run, env, ...x }) => x))}`);
    };
    const launch = async (name, gpu = "g5.xlarge-test") => {
      const r = await call2("/api/standalone", { method: "POST", body: { name, provider: "brev", provider_gpu: gpu, preset: "wan", weights_source: "hub", weights_download_approved: true, channel: "stable", deadline_min: 30, max_gpu_dph: 5 }, headers: H });
      assert.equal(r.status, 201, JSON.stringify(r.j));
      const op = await wait2(r.j.pod.id, "up", 150000);
      return { id: r.j.pod.id, op };
    };
    const stop = async (name, id) => {
      assert.equal((await call2(`/api/standalone/${name}/stop`, { method: "POST", body: {}, headers: H })).status, 202);
      return wait2(id, "down");
    };
    const parked = async () => (await call2("/api/providers/brev/parked", { headers: H })).j;
    const posts = () => b.calls.filter((c) => c.method === "POST" && c.path === `/organizations/${mock.brevOrg}/workspaces`).length;
    const posts0 = posts();
    const callsFrom = b.calls.length;

    // The launch form's outlook: cold on the stoppable type, the non-stoppable one flagged.
    const v = (await call2("/api/standalone/validate", { method: "POST", body: { name: "warm1", provider: "brev", provider_gpu: "g5.xlarge-test", preset: "wan", weights_source: "hub", weights_download_approved: true, deadline_min: 30, max_gpu_dph: 5 }, headers: H })).j;
    assert.equal(v.ok, true, JSON.stringify(v.issues));
    assert.ok(v.warnings.some((x) => x.path[0] === "provider_gpu" && /^cold: downloads fastwan22-ti2v-5b, wan22-ti2v-5b .*kept on stop/.test(x.message)), JSON.stringify(v.warnings));
    const v2 = (await call2("/api/standalone/validate", { method: "POST", body: { name: "warm1", provider: "brev", provider_gpu: "a100-test", preset: "wan", weights_source: "hub", weights_download_approved: true, deadline_min: 30, max_gpu_dph: 5 }, headers: H })).j;
    assert.ok(v2.warnings.some((x) => /a100-test is not stoppable: the weights download at every launch \(stoppable allowed types keep them: g5\.xlarge-test\)/.test(x.message)), JSON.stringify(v2.warnings));
    const dyn = (await call2("/api/schemas/dynamic", { headers: H })).j;
    const bg = dyn.provider_gpus.filter((g) => g.provider === "brev");
    assert.deepEqual(bg.map((g) => g.id), ["g5.xlarge-test", "a100-test"], "stoppable types first");
    assert.match(bg[0].detail, /\$1\.01\/hr · stoppable: weights kept on stop/, "the live list's price");

    // 1. Cold launch: both trees downloaded; the disk sized for them.
    const l1 = await launch("warm1");
    assert.equal(l1.op.status, "done", l1.op.error + JSON.stringify(l1.op.log.slice(-6)));
    const ws1 = b.workspaces.find((x) => x.name.startsWith("fv-pod-warm1-"));
    assert.equal(ws1.body.diskStorage, "200Gi");
    assert.equal(ws1.body.isStoppable, true, "a stoppable type is created with isStoppable true");
    let boot = b.boots.filter((x) => x.ws === ws1.id).at(-1);
    assert.deepEqual(boot.fetched.sort(), ["fastwan22-ti2v-5b", "wan22-ti2v-5b"]);
    const key = `brev:${ws1.name}`;
    let sp = (await call2("/api/standalone/warm1", { headers: H })).j.pod;
    assert.equal(sp.pod.cost_per_hr, 1.006, "the live list's base price");

    // 2. Stop: the workspace is stopped, not deleted, and parked with its trees.
    let op = await stop("warm1", l1.id);
    assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-6)));
    assert.ok(op.log.some((l) => /parked \(stopped; fastwan22-ti2v-5b, wan22-ti2v-5b kept on its 200 GB disk/.test(l.msg)), JSON.stringify(op.log.map((l) => l.msg)));
    assert.equal(ws1.status, "STOPPED");
    assert.ok(b.calls.some((c) => c.method === "PUT" && c.path === `/workspaces/${ws1.id}/stop`));
    assert.ok(!b.calls.some((c) => c.method === "DELETE" && c.path === `/workspaces/${ws1.id}`));
    let pk = await parked();
    assert.equal(pk.parked.length, 1);
    assert.equal(pk.parked[0].workspace_id, ws1.id);
    assert.deepEqual(Object.keys(pk.parked[0].trees).sort(), ["fastwan22-ti2v-5b", "wan22-ti2v-5b"]);
    assert.ok(Math.abs(pk.parked[0].storage_usd_per_day - 200 * 0.000132 * 24) < 1e-9);
    // The boot endpoint refuses a parked VM (its token is in its startup script).
    const tok1 = /printf 'Authorization: Bearer %s\\n' '([^']+)'/.exec(ws1.startupScript)[1];
    assert.equal((await fetch(w2.url + "/ingest/v1/brev-boot", { headers: { authorization: `Bearer ${tok1}` } })).status, 409);
    // The collector: parked storage in the ledger and the pods table (owner brev:parked); no orphan alert for it; the budget counts it.
    await call2("/api/collect", { method: "POST", body: {}, headers: H });
    await call2("/api/collect", { method: "POST", body: {}, headers: H });
    const pods = Object.fromEntries((await call2("/api/pods", { headers: H })).j.pods.map((p) => [p.pod_id, p]));
    assert.equal(pods[key]?.owner, "brev:parked");
    assert.equal(pods[key]?.desired_status, "PARKED");
    const costs = (await call2("/api/costs?days=1", { headers: H })).j;
    assert.ok(costs.by_owner.some((x) => x.owner === "brev:parked" && x.usd > 0), JSON.stringify(costs.by_owner));
    const al = (await call2("/api/alerts", { headers: H })).j.alerts;
    assert.ok(!al.some((a) => a.kind === "brev_orphan" && a.target === key), "a parked instance is ours");
    assert.ok(al.some((a) => a.kind === "brev_orphan" && /fv-pod-stray/.test(a.message)), "an unrecorded fv- one is reported");
    const prov = (await call2("/api/providers", { headers: H })).j.providers.find((p) => p.id === "brev");
    assert.ok(prov.parked_dph > 0, JSON.stringify(prov));

    // 3. The same preset and type again: the parked instance is restarted (no create), nothing downloaded, the new launch's env.
    const vw = (await call2("/api/standalone/validate", { method: "POST", body: { name: "warm2", provider: "brev", provider_gpu: "g5.xlarge-test", preset: "wan", weights_source: "hub", weights_download_approved: true, deadline_min: 30, max_gpu_dph: 5 }, headers: H })).j;
    assert.ok(vw.warnings.some((x) => new RegExp(`^warm: restarts parked ${ws1.name}`).test(x.message)), JSON.stringify(vw.warnings));
    const l2 = await launch("warm2");
    assert.equal(l2.op.status, "done", l2.op.error + JSON.stringify(l2.op.log.slice(-8)));
    assert.equal(posts(), posts0 + 1, "no second create");
    assert.ok(b.calls.some((c) => c.method === "PUT" && c.path === `/workspaces/${ws1.id}/start`));
    assert.ok(l2.op.log.some((l) => /warm restart of parked NVIDIA Brev/.test(l.msg)), JSON.stringify(l2.op.log.map((l) => l.msg)));
    boot = b.boots.filter((x) => x.ws === ws1.id).at(-1);
    assert.deepEqual([boot.kept.sort(), boot.fetched], [["fastwan22-ti2v-5b", "wan22-ti2v-5b"], []], "the trees are skipped");
    assert.equal(boot.env.FV_POD_ID, key, "it keeps its key");
    assert.ok(boot.env.FV_ENDPOINT_REPORT_TOKEN !== b.boots.filter((x) => x.ws === ws1.id)[0].env.FV_ENDPOINT_REPORT_TOKEN, "the new cluster's env (a fresh run script)");
    sp = (await call2("/api/standalone/warm2", { headers: H })).j.pod;
    assert.equal(sp.pod.pod_id, key);
    assert.equal((await parked()).parked.length, 0);
    op = await stop("warm2", l2.id);
    assert.equal(op.status, "done", op.error);
    assert.equal((await parked()).parked.length, 1, "parked again");

    // 4. Restart fails (no capacity): the parked one is held for the owner, a fresh instance is made (and downloads).
    b.startFail = true;
    const l3 = await launch("warm3");
    b.startFail = false;
    assert.equal(l3.op.status, "done", l3.op.error + JSON.stringify(l3.op.log.slice(-8)));
    assert.ok(l3.op.log.some((l) => /restart of parked .* failed .*no capacity.*held for the owner.*creating a fresh instance/.test(l.msg)), JSON.stringify(l3.op.log.map((l) => l.msg)));
    assert.equal(posts(), posts0 + 2);
    const ws3 = b.workspaces.find((x) => x.name.startsWith("fv-pod-warm3-"));
    assert.deepEqual(b.boots.filter((x) => x.ws === ws3.id).at(-1).fetched.sort(), ["fastwan22-ti2v-5b", "wan22-ti2v-5b"]);
    pk = await parked();
    assert.deepEqual(pk.parked.map((p) => [p.workspace_id, p.state]), [[ws1.id, "held"]]);
    assert.match(pk.parked[0].note, /restart failed: .*no capacity/);
    op = await stop("warm3", l3.id);
    assert.equal(op.status, "done", op.error);
    assert.deepEqual((await parked()).parked.map((p) => p.state).sort(), ["held", "parked"]);

    // 5. Restart that never runs (BREV_RESTART_TIMEOUT_S 30): held, a fresh one replaces it.
    b.startHang = true;
    const l4 = await launch("warm4");
    b.startHang = false;
    assert.equal(l4.op.status, "done", l4.op.error + JSON.stringify(l4.op.log.slice(-8)));
    assert.ok(l4.op.log.some((l) => /restart timed out/.test(l.msg)), JSON.stringify(l4.op.log.map((l) => l.msg)));
    assert.equal(posts(), posts0 + 3);
    assert.equal(ws3.status, "STOPPED", "stopped again after the timeout");
    assert.deepEqual((await parked()).parked.map((p) => [p.workspace_id, p.state]).sort(), [[ws1.id, "held"], [ws3.id, "held"]].sort());
    // Parking warm4's instance makes three: past brev_park_max 2, the oldest goes (only ours).
    op = await stop("warm4", l4.id);
    assert.equal(op.status, "done", op.error);
    pk = await parked();
    assert.equal(pk.parked.length, 2);
    assert.ok(op.log.some((l) => /parked instance deleted \(more than brev_park_max 2 parked\)/.test(l.msg)), JSON.stringify(op.log.map((l) => l.msg)));

    // 6. A non-stoppable type: deleted on stop as before.
    const l5 = await launch("cold5", "a100-test");
    assert.equal(l5.op.status, "done", l5.op.error + JSON.stringify(l5.op.log.slice(-8)));
    const ws5 = b.workspaces.find((x) => x.name.startsWith("fv-pod-cold5-"));
    assert.equal(ws5.body.isStoppable, false, "not stoppable: isStoppable false");
    op = await stop("cold5", l5.id);
    assert.equal(op.status, "done", op.error);
    assert.ok(op.log.some((l) => /not parked \(a100-test is not stoppable\): deleting/.test(l.msg)), JSON.stringify(op.log.map((l) => l.msg)));
    assert.ok(!b.workspaces.some((x) => x.id === ws5.id), "deleted");

    // 7. The CLI and the owner's actions: list, unpark a held one, delete one; the age limit.
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const { mkdtempSync, writeFileSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const dir = mkdtempSync(`${tmpdir()}/fvc-cli2-`);
    writeFileSync(`${dir}/token`, tok, { mode: 0o600 });
    const cli = async (...args) => (await promisify(execFile)("bash", [new URL("../../../scripts/serve/fv-control.sh", import.meta.url).pathname, ...args], { env: { ...process.env, FV_CONTROL_URL: w2.url, FV_CONTROL_TOKEN_FILE: `${dir}/token`, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" } })).stdout;
    const out = await cli("brev", "parked");
    assert.match(out, /^policy: at most 2 parked, 7 days each; storage \$[\d.]+\/day$/m);
    pk = await parked();
    const held = pk.parked.find((p) => p.state === "held");
    assert.ok(held, JSON.stringify(pk));
    assert.match(out, new RegExp(`^${held.workspace_id}\\t${held.name}\\theld\\tg5\\.xlarge-test\\t200 GB\\t`, "m"));
    assert.match(await cli("brev", "unpark", held.workspace_id), /: parked \(the next launch of its type may restart it\)/);
    const other = pk.parked.find((p) => p.workspace_id !== held.workspace_id);
    assert.match(await cli("brev", "delete-parked", other.workspace_id), /parked instance deleted \(deleted by /);
    assert.ok(!b.workspaces.some((x) => x.id === other.workspace_id));
    await assert.rejects(cli("brev", "delete-parked", "wsforeign"), /no parked Brev instance/);
    // brev_park_max_days 0.0001 (~9 s): the age limit deletes the last one at the next collect.
    const pol = (await call2("/api/policies", { headers: H })).j.policies;
    assert.equal((await call2("/api/policies", { method: "PUT", body: { ...pol, brev_park_max_days: 0.0001 }, headers: { cookie: ck, "x-csrf-token": cs } })).status, 200);
    await sleep(9000);
    await call2("/api/collect", { method: "POST", body: {}, headers: H });
    assert.equal((await parked()).parked.length, 0);
    assert.ok(!b.workspaces.some((x) => x.id === held.workspace_id), "deleted by age");

    // Only ours: the foreign and the unrecorded fv- workspaces were never stopped, started or deleted.
    for (const id of ["wsforeign", "wsstray"]) assert.ok(!b.calls.slice(callsFrom).some((c) => c.path.startsWith(`/workspaces/${id}`)), id);
    assert.deepEqual(b.workspaces.map((x) => x.id).sort(), ["wsforeign", "wsstray"]);
  } finally {
    w2.stop();
    b.workspaces = b.workspaces.filter((x) => !["wsforeign", "wsstray"].includes(x.id));
  }
});

await step("named pod tunnels: create → token in the pod's env, URL known up front → only its own URL accepted → delete removes DNS + tunnel; failed create; Brev park keeps them; leak cleanup; only ours; no token anywhere", async () => {
  // A third fv-control with named tunnels on the simulated Cloudflare account / zone (test/harness.mjs m.cf).
  const TUNNEL_API = "cftun_TEST_0123456789abcdef";
  mock.cf.token = TUNNEL_API;
  const c = mock.cf;
  const g = mock.gmi;
  const b = mock.brev;
  const w3 = await startWorker(mock, {
    ...SECRETS,
    CF_TUNNEL_API_TOKEN: TUNNEL_API,
    FV_TUNNEL_ZONE: "maximal.llc",
    FV_TUNNEL_HOST_TEMPLATE: "{pod}-fv-pods.maximal.llc",
    FV_TUNNEL_URL_TEMPLATE: `http://127.0.0.1:${mock.port}/tunnel/{host}`,
    GMI_PRODUCTS: "container.h200.x1",
    GMI_DEFAULT_IDC: "us-denver-1",
    GMI_BUDGET_USD: "50",
    BREV_ORG_ID: mock.brevOrg,
    BREV_INSTANCE_TYPES: "g5.xlarge-test",
    BREV_BUDGET_USD: "100",
    FV_HUB_DOWNLOADS_APPROVED: "1",
    OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  });
  // Not ours: a foreign tunnel, an fv- tunnel fv-control never recorded, a foreign record, an unrecorded fv-control record.
  const zone = c.zones[0].id;
  c.tunnels.push({ id: "00000000-0000-4000-8000-00000000f0e1", name: "someone-else", config_src: "cloudflare", created_at: "2026-10-01T00:00:00Z", deleted_at: null, secret: "x", token: "x", config: null, connections: [{ id: "c1", pod: "x" }] });
  c.tunnels.push({ id: "00000000-0000-4000-8000-0000000057a1", name: "fv-stray-0101-abcdef", config_src: "cloudflare", created_at: "2026-10-01T00:00:00Z", deleted_at: null, secret: "y", token: "y", config: null, connections: [] });
  c.dns.push({ id: "recforeign", zone, type: "A", name: "www.maximal.llc", content: "192.0.2.1", proxied: true, comment: null });
  c.dns.push({ id: "recghost", zone, type: "CNAME", name: "ghost-fv-pods.maximal.llc", content: "00000000-0000-4000-8000-0000000057a1.cfargotunnel.com", proxied: true, comment: "fv-control gmi:fv-pod-ghost" });
  const local = [];
  try {
    const call3 = async (path, { method = "GET", body, headers = {} } = {}) => {
      const r = await fetch(w3.url + path, { method, headers: { ...(body !== undefined ? { "content-type": "application/json" } : {}), ...headers }, body: body === undefined ? undefined : JSON.stringify(body) });
      const text = await r.text();
      bodies.push(text);
      local.push(text);
      let j = null;
      try { j = JSON.parse(text); } catch {}
      return { status: r.status, j, text };
    };
    const lr = await fetch(w3.url + "/api/auth/login", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ passphrase: PASSPHRASE }) });
    const ck = lr.headers.get("set-cookie").split(";")[0];
    const cs = (await lr.json()).csrf;
    const tok = (await call3("/api/tokens", { method: "POST", body: { name: "ci3", scope: "admin" }, headers: { cookie: ck, "x-csrf-token": cs } })).j.token;
    const H = { authorization: `Bearer ${tok}` };
    const wait3 = async (cluster, kind, timeoutMs = 120000) => {
      const t = Date.now();
      while (Date.now() - t < timeoutMs) {
        const op = (await call3(`/api/clusters/${cluster}/ops`, { headers: H })).j.operations.find((o) => o.kind === kind && !/is running/.test(o.error || ""));
        if (op && op.status !== "running") return op;
        await sleep(500);
      }
      throw new Error(`operation ${kind} did not finish`);
    };
    const msgs = (op) => op.log.map((l) => l.msg);
    const fake = { variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], channel: "stable", deadline_min: 20 };
    const cfCalls = (re, method) => c.calls.filter((x) => re.test(x.path) && (!method || x.method === method));

    // The settings: named, one level below the zone (Universal SSL covers it): no warning.
    const prov = (await call3("/api/providers", { headers: H })).j.providers.find((p) => p.id === "gmi");
    assert.deepEqual([prov.tunnel.mode, prov.tunnel.levels, prov.tunnel.warning, prov.tunnel.error], ["named", 1, null, null], JSON.stringify(prov.tunnel));
    assert.equal((await call3("/api/tunnels", { headers: H })).j.settings.host_template, "{pod}-fv-pods.maximal.llc");

    // 1. GMI: the tunnel, its ingress and its DNS record before the container; the token in the container's env only.
    let r = await call3("/api/standalone", { method: "POST", body: { name: "tun1", provider: "gmi", provider_gpu: "container.h200.x1", ...fake }, headers: H });
    assert.equal(r.status, 201, JSON.stringify(r.j));
    const gid = r.j.pod.id;
    let op = await wait3(gid, "up", 90000);
    assert.equal(op.status, "done", op.error + JSON.stringify(msgs(op).slice(-8)));
    const gc = g.containers.find((x) => x.name.startsWith("fv-pod-tun1-"));
    const gkey = `gmi:${gc.name}`;
    const t1 = c.tunnels.find((t) => t.name.startsWith(`${gc.name}-`));
    assert.ok(t1, JSON.stringify(c.tunnels.map((t) => t.name)));
    assert.match(t1.name, /^fv-pod-tun1-\d{10}-[0-9a-f]{6}$/);
    const host1 = `${t1.name}-fv-pods.maximal.llc`;
    const url1 = `http://127.0.0.1:${mock.port}/tunnel/${host1}`;
    assert.equal(t1.config_src, "cloudflare");
    assert.deepEqual(t1.config, { ingress: [{ hostname: host1, service: "http://127.0.0.1:8000" }, { service: "http_status:404" }] }, "no originRequest: websockets / SSE keep cloudflared's defaults");
    const rec1 = c.dns.find((x) => x.name === host1);
    assert.deepEqual([rec1.type, rec1.content, rec1.proxied, rec1.comment], ["CNAME", `${t1.id}.cfargotunnel.com`, true, `fv-control ${gkey}`]);
    assert.ok(c.calls.every((x) => x.auth), "every tunnel / DNS call carries CF_TUNNEL_API_TOKEN");
    assert.equal(cfCalls(/^\/zones\?name=maximal\.llc$/).length, 1, "the zone id is looked up once (cached)");
    // The create order: tunnel, ingress, DNS, token, then the provider's create.
    const order = mock.log.filter((x) => /^\/cf\/(accounts|zones\/)/.test(x.path) || (x.path === "/gmi/v1/containers" && x.method === "POST")).map((x) => `${x.method} ${x.path.replace(/[0-9a-f-]{32,36}/g, "<id>")}`);
    const i0 = order.indexOf("POST /cf/accounts/acct/cfd_tunnel");
    assert.deepEqual(order.slice(i0, i0 + 5), ["POST /cf/accounts/acct/cfd_tunnel", "PUT /cf/accounts/acct/cfd_tunnel/<id>/configurations", `POST /cf/zones/${zone}/dns_records`, "GET /cf/accounts/acct/cfd_tunnel/<id>/token", "POST /gmi/v1/containers"], JSON.stringify(order));
    const genv = Object.fromEntries(gc.envs.map((e) => [e.name, e.value]));
    assert.equal(genv.FV_TUNNEL_TOKEN, t1.token, "the pod got its tunnel's token");
    assert.equal(genv.FV_TUNNEL_URL, url1);
    assert.match(gc.args[1], /TUNNEL_TOKEN="\$FV_TUNNEL_TOKEN" \/usr\/local\/bin\/cloudflared tunnel --no-autoupdate run/);
    assert.match(gc.args[1], /53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134/, "the same pinned cloudflared");
    // The URL is known up front (logged at create, before the pod reports it) and served through the tunnel.
    const L = msgs(op);
    const at = L.findIndex((m) => m.includes(`named tunnel ${t1.name}: ${gkey} will answer at ${url1}`));
    assert.ok(at >= 0, JSON.stringify(L));
    assert.ok(L.findIndex((m) => /^pools /.test(m)) > at, JSON.stringify(L));
    assert.ok(mock.reports.some((x) => x.key === gkey && x.phase === "tunnel" && x.url === url1 && x.status === 200));
    assert.equal((await call3("/api/standalone/tun1", { headers: H })).j.pod.pod.url, url1);
    assert.ok(mock.tunnelHits.some((x) => x.host === host1 && x.pod === gkey && x.route === "/health"), "health went DNS -> tunnel -> ingress -> the pod");
    // Only its own host's URL is accepted: not a quick tunnel, not another host.
    const rep = (url) => call3("/ingest/v1/endpoint", { method: "POST", body: { pod: gkey, phase: "tunnel", url }, headers: { authorization: `Bearer ${genv.FV_ENDPOINT_REPORT_TOKEN}` } });
    assert.equal((await rep("https://abc-def.trycloudflare.com")).status, 400, "no quick-tunnel URL for a named-tunnel pod");
    assert.equal((await rep(`http://127.0.0.1:${mock.port}/tunnel/other-fv-pods.maximal.llc`)).status, 400);
    assert.equal((await rep(`https://${host1}`)).status, 400, "only the exact URL it was given");
    assert.equal((await rep(url1)).status, 200);
    // The owner's view; the sweep leaves a live pod's tunnel alone.
    let tv = (await call3("/api/tunnels", { headers: H })).j;
    assert.deepEqual(tv.tunnels.map((x) => [x.pod_id, x.host, x.state, x.tunnel_id, x.dns_record_id]), [[gkey, host1, "live", t1.id, rec1.id]]);
    // The CLI: the providers' tunnel line, the recorded tunnels.
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const { mkdtempSync, writeFileSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const dir = mkdtempSync(`${tmpdir()}/fvc-cli3-`);
    writeFileSync(`${dir}/token`, tok, { mode: 0o600 });
    const cli = async (...args) => (await promisify(execFile)("bash", [new URL("../../../scripts/serve/fv-control.sh", import.meta.url).pathname, ...args], { env: { ...process.env, FV_CONTROL_URL: w3.url, FV_CONTROL_TOKEN_FILE: `${dir}/token`, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" } })).stdout;
    assert.match(await cli("providers"), /^tunnels\tnamed\t\{pod\}-fv-pods\.maximal\.llc\tnamed pod tunnels: one Cloudflare Tunnel/m);
    assert.match(await cli("tunnels"), new RegExp(`^mode: named \\(\\{pod\\}-fv-pods\\.maximal\\.llc\\)\\n${gkey}\\t${host1.replace(/\./g, "\\.")}\\t${t1.name}\\tlive\\t$`, "m"));
    let sw = (await call3("/api/tunnels/sweep", { method: "POST", body: {}, headers: H })).j;
    assert.ok(!sw.actions.some((a) => a.includes(t1.name)), JSON.stringify(sw));
    assert.ok(!t1.deleted_at && c.dns.includes(rec1));
    // The provider's log is scrubbed of the tunnel token (and of the key).
    const lg = (await call3(`/api/pods/${encodeURIComponent(gkey)}/provider-logs`, { headers: H })).j;
    assert.ok(lg.lines.some((l) => /leak\? TUNNEL_TOKEN=\[redacted\]/.test(l)), JSON.stringify(lg.lines));

    // 2. Stop: DNS record, then the tunnel's (stale) connections, then the tunnel.
    assert.equal((await call3("/api/standalone/tun1/stop", { method: "POST", body: {}, headers: H })).status, 202);
    op = await wait3(gid, "down");
    assert.equal(op.status, "done", op.error);
    assert.ok(msgs(op).some((m) => m.includes(`tunnel ${t1.name} (${host1}) deleted: its pod was deleted`)), JSON.stringify(msgs(op)));
    assert.ok(!c.dns.includes(rec1), "DNS record deleted");
    assert.ok(t1.deleted_at, "tunnel deleted");
    const seq = mock.log.filter((x) => x.method === "DELETE" && x.path.startsWith("/cf/")).map((x) => x.path.replace(t1.id, "<t1>").replace(rec1.id, "<rec1>"));
    assert.deepEqual(seq.slice(-3), [`/cf/zones/${zone}/dns_records/<rec1>`, "/cf/accounts/acct/cfd_tunnel/<t1>/connections", "/cf/accounts/acct/cfd_tunnel/<t1>"], JSON.stringify(seq));
    tv = (await call3("/api/tunnels?all=1", { headers: H })).j;
    assert.equal(tv.tunnels.find((x) => x.name === t1.name).state, "deleted");
    assert.equal((await call3("/api/standalone/tun1", { method: "DELETE", headers: H })).status, 200);

    // 3. The provider's create fails: the tunnel and its record go at once.
    g.failCreate = 1;
    r = await call3("/api/standalone", { method: "POST", body: { name: "tun2", provider: "gmi", provider_gpu: "container.h200.x1", ...fake }, headers: H });
    op = await wait3(r.j.pod.id, "up", 90000);
    g.failCreate = 0;
    assert.equal(op.status, "failed");
    const t2 = c.tunnels.filter((t) => t.name.startsWith("fv-pod-tun2-"));
    assert.ok(t2.length >= 1 && t2.every((t) => t.deleted_at), JSON.stringify(t2.map((t) => [t.name, t.deleted_at])));
    assert.ok(!c.dns.some((x) => x.name.startsWith("fv-pod-tun2-")), "no DNS record left");
    assert.ok(msgs(op).some((m) => /tunnel fv-pod-tun2-.* deleted: the provider's create failed/.test(m)), JSON.stringify(msgs(op)));
    await call3("/api/standalone/tun2", { method: "DELETE", headers: H });

    // 4. Brev keep-on-stop: a parked instance keeps its tunnel and record; its warm restart runs the same tunnel;
    //    deleting the parked instance deletes them.
    b.boots = [];
    const launchB = async (name) => {
      const rr = await call3("/api/standalone", { method: "POST", body: { name, provider: "brev", provider_gpu: "g5.xlarge-test", preset: "wan", weights_source: "hub", weights_download_approved: true, channel: "stable", deadline_min: 30, max_gpu_dph: 5 }, headers: H });
      assert.equal(rr.status, 201, JSON.stringify(rr.j));
      return { id: rr.j.pod.id, op: await wait3(rr.j.pod.id, "up", 150000) };
    };
    const lb = await launchB("tunb");
    assert.equal(lb.op.status, "done", lb.op.error + JSON.stringify(msgs(lb.op).slice(-8)));
    const ws = b.workspaces.find((x) => x.name.startsWith("fv-pod-tunb-"));
    const bkey = `brev:${ws.name}`;
    const tb = c.tunnels.find((t) => t.name.startsWith(`${ws.name}-`));
    const hostB = `${tb.name}-fv-pods.maximal.llc`;
    const recB = c.dns.find((x) => x.name === hostB);
    assert.equal(recB.comment, `fv-control ${bkey}`);
    const boot1 = b.boots.filter((x) => x.ws === ws.id).at(-1);
    assert.equal(boot1.env.FV_TUNNEL_TOKEN, tb.token, "the run script (sealed in D1) carries the token");
    assert.ok(!(await call3(`/api/standalone/tunb`, { headers: H })).text.includes(tb.token));
    assert.equal((await call3("/api/standalone/tunb/stop", { method: "POST", body: {}, headers: H })).status, 202);
    op = await wait3(lb.id, "down");
    assert.ok(msgs(op).some((m) => /parked \(stopped;/.test(m)), JSON.stringify(msgs(op)));
    assert.ok(!tb.deleted_at && c.dns.includes(recB), "parked: tunnel and DNS kept");
    sw = (await call3("/api/tunnels/sweep", { method: "POST", body: {}, headers: H })).j;
    assert.ok(!tb.deleted_at, "the sweep keeps a parked instance's tunnel");
    const creates = cfCalls(/^\/accounts\/acct\/cfd_tunnel$/, "POST").length;
    const lb2 = await launchB("tunb2");
    assert.equal(lb2.op.status, "done", lb2.op.error + JSON.stringify(msgs(lb2.op).slice(-8)));
    assert.ok(msgs(lb2.op).some((m) => m.includes(`named tunnel ${tb.name}: ${bkey} will answer at http://127.0.0.1:${mock.port}/tunnel/${hostB} (kept with the parked instance)`)), JSON.stringify(msgs(lb2.op)));
    assert.equal(cfCalls(/^\/accounts\/acct\/cfd_tunnel$/, "POST").length, creates, "no new tunnel for a warm restart");
    assert.equal(b.boots.filter((x) => x.ws === ws.id).at(-1).env.FV_TUNNEL_TOKEN, tb.token);
    assert.equal((await call3("/api/standalone/tunb2", { headers: H })).j.pod.pod.url, `http://127.0.0.1:${mock.port}/tunnel/${hostB}`);
    assert.equal((await call3("/api/standalone/tunb2/stop", { method: "POST", body: {}, headers: H })).status, 202);
    op = await wait3(lb2.id, "down");
    assert.ok(!tb.deleted_at, "parked again: kept");
    assert.equal((await call3(`/api/providers/brev/parked/${ws.id}/delete`, { method: "POST", body: {}, headers: H })).status, 200);
    assert.ok(tb.deleted_at && !c.dns.includes(recB), "the parked instance deleted: its tunnel and record too");
    for (const n of ["tunb", "tunb2"]) await call3(`/api/standalone/${n}`, { method: "DELETE", headers: H });

    // 5. Leaks: a recorded tunnel whose pod is gone (fv-control crashed before its delete), and one made before a crash
    //    recorded its id. The sweep deletes both; strays get an alert and are never touched.
    const mkT = (name) => {
      const id = crypto.randomUUID();
      const t = { id, name, config_src: "cloudflare", created_at: new Date().toISOString(), deleted_at: null, secret: "s", token: Buffer.from(JSON.stringify({ a: "acct", t: id, s: "s" })).toString("base64"), config: null, connections: [{ id: "stale", pod: "gone" }] };
      c.tunnels.push(t);
      return t;
    };
    const tl = mkT("fv-pod-leak-0101000000-aaaaaa");
    const hl = `${tl.name}-fv-pods.maximal.llc`;
    c.dns.push({ id: "recleak", zone, type: "CNAME", name: hl, content: `${tl.id}.cfargotunnel.com`, proxied: true, comment: "fv-control gmi:fv-pod-leak-0101000000" });
    const tc = mkT("fv-pod-crash-0101000000-bbbbbb");
    const old = Date.now() - 3600_000;
    d1Exec(w3.dir, `INSERT INTO pod_tunnels (name, pod_id, host, url, zone_id, tunnel_id, dns_record_id, state, created_at, updated_at) VALUES ('${tl.name}', 'gmi:fv-pod-leak-0101000000', '${hl}', 'https://${hl}', '${zone}', '${tl.id}', 'recleak', 'live', ${old}, ${old}), ('${tc.name}', 'brev:fv-pod-crash-0101000000', '${tc.name}-fv-pods.maximal.llc', 'https://${tc.name}-fv-pods.maximal.llc', '${zone}', NULL, NULL, 'creating', ${old}, ${old})`);
    await settle3();
    sw = (await call3("/api/tunnels/sweep", { method: "POST", body: {}, headers: H })).j;
    assert.ok(sw.actions.some((a) => a.startsWith(`tunnel ${tl.name} (${hl}) deleted: its pod gmi:fv-pod-leak-0101000000 is gone (leak cleanup)`)), JSON.stringify(sw));
    assert.ok(tl.deleted_at && !c.dns.some((x) => x.id === "recleak"));
    assert.ok(sw.actions.some((a) => a.includes(`tunnel ${tc.name}: id ${tc.id} recorded`)), JSON.stringify(sw));
    sw = (await call3("/api/tunnels/sweep", { method: "POST", body: {}, headers: H })).j;
    assert.ok(tc.deleted_at, JSON.stringify(sw));
    // Only ours: the foreign and the unrecorded fv- tunnel and records are listed, alerted (fv- ones), never deleted.
    const al = (await call3("/api/alerts", { headers: H })).j.alerts.filter((a) => a.kind === "tunnel_orphan");
    assert.ok(al.some((a) => a.target === "fv-stray-0101-abcdef" && /not touched/.test(a.message)), JSON.stringify(al));
    assert.ok(al.some((a) => a.target === "ghost-fv-pods.maximal.llc"), JSON.stringify(al));
    assert.ok(!al.some((a) => /someone-else|www\.maximal/.test(a.message)), "a foreign tunnel / record is not ours to report");
    for (const id of ["00000000-0000-4000-8000-00000000f0e1", "00000000-0000-4000-8000-0000000057a1", "recforeign", "recghost"])
      assert.ok(!c.calls.some((x) => x.method === "DELETE" && x.path.includes(id)), `${id} untouched`);
    assert.ok(c.tunnels.filter((t) => ["someone-else", "fv-stray-0101-abcdef"].includes(t.name)).every((t) => !t.deleted_at));

    // 6. No tunnel token or API token in any response, the Worker's output or the op logs.
    const secrets = [TUNNEL_API, ...c.tunnels.filter((t) => t.token.length > 8).map((t) => t.token)];
    const out = w3.output();
    for (const s of secrets) {
      assert.ok(!local.some((x) => x.includes(s)), "a token in a response");
      assert.ok(!out.includes(s), "a token in the Worker's log");
    }
  } finally {
    w3.stop();
    mock.cf.token = null;
    b.workspaces = b.workspaces.filter((x) => !x.name.startsWith("fv-pod-tunb"));
    g.containers = g.containers.filter((x) => !x.name.startsWith("fv-pod-tun"));
  }
  async function settle3() {
    for (let i = 0, ok = 0; i < 40 && ok < 2; i++) {
      ok = (await fetch(`${w3.url}/healthz`).then((r) => r.ok).catch(() => false)) ? ok + 1 : 0;
      await sleep(250);
    }
  }
});

await step("audit log; no secret in any response", async () => {
  const a = (await call("/api/audit?limit=500", { headers: T() })).j.audit;
  const actions = new Set(a.map((x) => x.action));
  for (const want of ["auth.login", "token.mint", "cluster.define", "cluster.up", "env.set", "cluster.restart", "cluster.scale", "cluster.roll", "cluster.extend", "release.promote", "cluster.stop", "cluster.admin-token.reveal", "cluster.mint-key", "cluster.revoke-key", "doc.save", "doc.restore"]) assert.ok(actions.has(want), `audit has ${want}`);
  assert.ok(a.some((x) => x.action === "auth.login" && x.ok === 0), "failed logins are audited");
  const envSet = a.find((x) => x.action === "env.set" && x.target.includes("HF_TOKEN"));
  assert.ok(!envSet.after.includes("hf_supersecret"));
  const all = bodies.join("\n");
  for (const [k, v] of Object.entries(SECRETS)) assert.ok(!all.includes(v), `${k} leaked into a response`);
  assert.ok(!all.includes("hf_supersecret"), "a secret env value leaked");
  assert.ok(!all.includes(mock.edgeInternal), "the edge's internal token leaked");
  assert.ok(!all.includes(worker.env.FV_LOG_SHIP_TOKEN), "an ingest token leaked");
});

await step("login rate limit", async () => {
  let last = 0;
  for (let i = 0; i < 6; i++) last = (await call("/api/auth/login", { method: "POST", body: { passphrase: "nope" } })).status;
  assert.equal(last, 429);
});

console.log(`\n${passed} integration steps passed in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
if (process.env.FVC_UI) {
  console.log(`UI: ${B} (Ctrl-C to stop)`);
} else {
  w.stop();
  mock.close();
}
