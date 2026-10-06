// Integration test: the Worker under `wrangler dev` (workerd, local D1, R2,
// Durable Objects) against mocked Runpod / GHCR / GitHub / Cloudflare APIs
// and mocked gateway and worker pods (test/harness.mjs).
//   node test/integration/run.mjs            (npm run test:integration)
//   FVC_UI=1 …                               also keep it up for test/ui/smoke.mjs
import assert from "node:assert/strict";
import { d1Exec, hashPassphrase, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.cloudriftKey = SECRETS.CLOUDRIFT_API_KEY;
const w = await startWorker(mock, { ...SECRETS, OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET) });
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
async function step(name, fn) {
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
  assert.equal(p.j.pods.length, 2);
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

let gw, worker;
await step("start: gateway, worker, gateway gets the worker URLs; ready", async () => {
  await d1Exec(w.dir, "DELETE FROM operations");
  const s = await call(`/api/clusters/${cid}/start`, { method: "POST", body: {}, headers: T() });
  assert.equal(s.status, 202);
  assert.equal((await call(`/api/clusters/${cid}/start`, { method: "POST", body: {}, headers: T() })).status, 409, "one operation at a time");
  const op = await waitOp(cid, "up");
  assert.equal(op.status, "done", op.error + JSON.stringify(op.log.slice(-5)));
  const pods = [...mock.pods.values()];
  assert.equal(pods.length, 2);
  gw = pods.find((p) => p.env.FV_GATEWAY_TOML_B64);
  worker = pods.find((p) => p !== gw);
  // The gateway payload: CPU, the script's boot command and env.
  assert.equal(gw.payload.computeType, "CPU");
  assert.deepEqual(gw.payload.ports, ["8000/http"]);
  assert.match(gw.payload.dockerStartCmd[0], /\[watchdog\]/);
  assert.equal(gw.env.FV_BACKSTOP_API_KEY, SECRETS.RUNPOD_API_KEY);
  assert.ok(gw.env.FV_ADMIN_TOKEN_RECIPIENT);
  assert.equal(gw.env.FV_ADMIN_TOKEN, undefined);
  assert.match(gw.image, /@sha256:/);
  assert.match(gw.env.FV_LOG_SHIP_URL, /\/ingest\/v1\/logs$/);
  assert.ok(Number(gw.env.FV_CLUSTER_DEADLINE) > Date.now() / 1000);
  assert.equal(gw.patches, 1, "patched once with the worker URLs");
  assert.equal(gw.env.FV_POOL_FAKE_URLS, `http://127.0.0.1:${mock.port}/pod/${worker.id}`);
  assert.equal(gw.env.FV_CLUSTER_PODS, worker.id);
  assert.equal(gw.env.FV_GITHUB_TOKEN, undefined, "tiny-cpu does not pass the GitHub token");
  // The worker: CPU, fake engine config inline, internal token, public URL = the gateway.
  assert.equal(worker.payload.computeType, "CPU");
  assert.equal(worker.env.FV_SERVE_ROLE, "worker");
  assert.equal(worker.env.FV_INTERNAL_TOKEN, gw.env.FV_INTERNAL_TOKEN);
  assert.match(Buffer.from(worker.env.FV_WORKER_TOML_B64, "base64").toString(), /backend = "fake"/);
  assert.equal(worker.env.FV_PUBLIC_BASE_URL, `http://127.0.0.1:${mock.port}/pod/${gw.id}`);
  assert.match(worker.image, /gateway-stable/.test(worker.image) ? /./ : /@sha256:/);
  const c = await call(`/api/clusters/${cid}`, { headers: T() });
  assert.equal(c.j.cluster.status, "running");
  assert.equal(c.j.pods.filter((p) => p.status === "ready").length, 2);
});

await step("the collector: pods, owners, costs, samples, external attribution, alerts", async () => {
  const r = await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.equal(r.status, 200, JSON.stringify(r.j));
  assert.equal(r.j.balance, 50);
  const pods = (await call("/api/pods", { headers: T() })).j.pods;
  const by = Object.fromEntries(pods.map((p) => [p.pod_id, p]));
  assert.equal(by[gw.id].owner, "cluster:tiny");
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
  assert.ok(al.some((a) => a.kind === "pod_idle" && a.target === "extgpu00001"), JSON.stringify(al));
  assert.ok(mock.pods.has(worker.id) && !mock.log.some((l) => l.method === "DELETE" && l.path.includes("extgpu")), "external pods are never touched");
});

await step("CloudRift: rentals collected, deadline backstop, balance, terminate only ours", async () => {
  // The collector step above already ran the cron once.
  const cr = mock.cloudrift;
  assert.equal(cr.instances.find((i) => i.id === "cr-ours-1").status, "Inactive", "past its fv-deadline tag: terminated");
  assert.equal(cr.instances.find((i) => i.id === "cr-ours-2").status, "Active");
  assert.equal(cr.instances.find((i) => i.id === "cr-foreign").status, "Active", "a foreign rental is never touched");
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
  assert.equal(ov.cloudrift.balance, 30);
  assert.equal(ov.balance, 50, "the Runpod balance is separate");
  const prov = (await call("/api/providers", { headers: T() })).j.providers;
  assert.deepEqual(prov.map((p) => [p.id, p.enabled]), [["runpod", true], ["cloudrift", true]]);
  const price = (await call("/api/providers/cloudrift/price?gpu=RTX%20PRO%206000", { headers: T() })).j;
  assert.equal(price.offers[0].usd_per_hr, 1.3936);
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
  cr.balance = 5;
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok((await call("/api/alerts", { headers: T() })).j.alerts.some((a) => a.kind === "cloudrift_balance_floor"));
  assert.equal(cr.instances.find((i) => i.id === "cr-foreign").status, "Active");
  cr.balance = 30;
  await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok(!(await call("/api/alerts", { headers: T() })).j.alerts.some((a) => a.kind === "cloudrift_balance_floor"), "resolved");
});

await step("env at three levels: masked view, restart needed, rolling restart", async () => {
  assert.equal((await call("/api/env/account/RUST_LOG", { method: "PUT", body: { value: "warn" }, headers: T() })).status, 200);
  assert.equal((await call(`/api/env/cluster/${cid}/HF_TOKEN`, { method: "PUT", body: { value: "hf_supersecret", secret: true }, headers: T() })).status, 200);
  assert.equal((await call(`/api/env/pod/${worker.id}/RUST_LOG`, { method: "PUT", body: { value: "debug" }, headers: T() })).status, 200);
  assert.equal((await call(`/api/env/cluster/${cid}/FV_INTERNAL_TOKEN`, { method: "PUT", body: { value: "x" }, headers: T() })).status, 400);
  const e = (await call(`/api/clusters/${cid}/env`, { headers: T() })).j;
  assert.deepEqual(e.needs_restart.sort(), [gw.id, worker.id].sort());
  const wenv = Object.fromEntries(e.pods.find((p) => p.pod_id === worker.id).env.map((v) => [v.key, v]));
  assert.equal(wenv.RUST_LOG.value, "debug");
  assert.equal(wenv.RUST_LOG.source, "pod");
  assert.equal(wenv.HF_TOKEN.value, "••••••••");
  assert.equal(wenv.FV_INTERNAL_TOKEN.secret, true);
  const genv = Object.fromEntries(e.pods.find((p) => p.pod_id === gw.id).env.map((v) => [v.key, v]));
  assert.equal(genv.RUST_LOG.value, "warn");
  assert.equal(genv.FV_BACKSTOP_API_KEY.value, "••••••••");
  const s = await call(`/api/clusters/${cid}/restart`, { method: "POST", body: {}, headers: T() });
  assert.equal(s.status, 202);
  const op = await waitOp(cid, "restart");
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.get(worker.id).env.RUST_LOG, "debug");
  assert.equal(mock.pods.get(worker.id).env.HF_TOKEN, "hf_supersecret");
  assert.equal(mock.pods.get(gw.id).env.RUST_LOG, "warn");
  // Workers first, the gateway last.
  const patches = mock.log.filter((l) => l.method === "PATCH").slice(-2).map((l) => l.path.split("/").pop());
  assert.deepEqual(patches, [worker.id, gw.id]);
  assert.deepEqual((await call(`/api/clusters/${cid}/env`, { headers: T() })).j.needs_restart, []);
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
  const all = (await call(`/api/logs?pod=${worker.id}&level=trace`, { headers: T() })).j.lines;
  assert.equal(all.length, 3);
  assert.ok(!JSON.stringify(all).includes(SECRETS.RUNPOD_API_KEY), "secrets scrubbed from logs");
  assert.equal((await call(`/api/logs?pod=${worker.id}&level=warn`, { headers: T() })).j.lines.length, 1);
  assert.equal((await call(`/api/logs?pod=${worker.id}&level=trace&q=job_abc`, { headers: T() })).j.lines.length, 2);
  for (let i = 0; i < 20 && !got.length; i++) await sleep(100);
  assert.equal(got[0]?.lines?.length, 3, "live tail got the batch");
  ws.close();
  const d = await call(`/api/logs/download?pod=${worker.id}`, { headers: T(), raw: true });
  assert.equal(d.text.trim().split("\n").length, 3);
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
  assert.ok(dyn.env_keys.includes("HF_TOKEN"));
  assert.ok(dyn.regions.find((r) => r.id === "eu").volume === "jg48s6o1w0");
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
  assert.equal(mock.pods.size, 3);
  assert.equal(mock.pods.get(gw.id).env.FV_POOL_FAKE_URLS.split(",").length, 2);
  await d1Exec(w.dir, "DELETE FROM operations WHERE kind = 'scale'");
  assert.equal((await call(`/api/clusters/${cid}/scale`, { method: "POST", body: { pool: "fake", count: 1 }, headers: T() })).status, 202);
  op = await waitOp(cid, "scale", 90000);
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.equal(mock.pods.size, 2);
  assert.equal(mock.drained.length, 1);
  assert.equal(mock.pods.get(gw.id).env.FV_POOL_FAKE_URLS.split(",").length, 1);
  worker = [...mock.pods.values()].find((p) => p.id !== gw.id);
});

await step("rolling redeploy to a commit, gateway included", async () => {
  const old = worker.id;
  assert.equal((await call(`/api/clusters/${cid}/roll`, { method: "POST", body: { target: "abcdef1", gateway: true }, headers: T() })).status, 202);
  const op = await waitOp(cid, "roll", 120000);
  assert.equal(op.status, "done", JSON.stringify(op.log));
  assert.ok(!mock.pods.has(old), "old worker deleted");
  assert.ok(mock.drained.includes(old), "old worker drained first");
  const nw = [...mock.pods.values()].find((p) => p.id !== gw.id);
  assert.match(nw.image, /@sha256:/);
  assert.equal(nw.env.FV_IMAGE_DIGEST, "sha256:" + Buffer.from("gateway-sha-abcdef1").toString("hex").padEnd(64, "0").slice(0, 64));
  assert.equal(mock.pods.get(gw.id).image, nw.image, "the gateway moved to the same build (same pod id)");
  assert.equal(mock.pods.get(gw.id).env.FV_POOL_FAKE_URLS, `http://127.0.0.1:${mock.port}/pod/${nw.id}`);
  worker = nw;
});

await step("extend; stop and start the gateway; admin token and key minting through the gateway", async () => {
  const before = Number(mock.pods.get(gw.id).env.FV_CLUSTER_DEADLINE);
  assert.equal((await call(`/api/clusters/${cid}/extend`, { method: "POST", body: { minutes: 30 }, headers: T() })).status, 202);
  assert.equal((await waitOp(cid, "extend")).status, "done");
  assert.equal(Number(mock.pods.get(gw.id).env.FV_CLUSTER_DEADLINE), before + 1800);
  assert.equal((await call(`/api/clusters/${cid}/gateway/stop`, { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(cid, "gateway-stop")).status, "done");
  assert.equal(mock.pods.get(gw.id).desiredStatus, "EXITED");
  assert.equal((await call(`/api/clusters/${cid}/gateway/start`, { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(cid, "gateway-start")).status, "done");
  assert.equal(mock.pods.get(gw.id).desiredStatus, "RUNNING");
  const tok = await call(`/api/clusters/${cid}/admin-token`, { method: "POST", body: {}, headers: T() });
  assert.equal(tok.j.admin_token, mock.adminToken, "opened the sealed token");
  const k = await call(`/api/clusters/${cid}/mint-key`, { method: "POST", body: { name: "laptop" }, headers: T() });
  assert.equal(k.status, 201);
  assert.deepEqual(mock.minted, ["laptop"]);
  const gv = (await call(`/api/clusters/${cid}/gateway`, { headers: T() })).j;
  assert.equal(gv.pools.state[0].id, "fake");
});

await step("GitHub: release dispatch, CI status; image tags", async () => {
  const r = await call("/api/github/release", { method: "POST", body: { action: "promote", target: "abcdef1", channel: "stable", dry_run: true }, headers: T() });
  assert.equal(r.status, 202, JSON.stringify(r.j));
  assert.equal(mock.dispatches[0].inputs.target, "abcdef1");
  assert.equal(mock.dispatches[0].inputs.dry_run, "true");
  assert.equal((await call("/api/github/release", { method: "POST", body: { action: "promote", target: "$(x)" }, headers: T() })).status, 400);
  const ci = (await call("/api/github/ci", { headers: T() })).j;
  assert.equal(ci.main[0].conclusion, "success");
  assert.ok((await call("/api/images/tags?filter=gateway", { headers: T() })).j.tags.includes("gateway-stable"));
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
  assert.equal(mock.pods.size, 2);
  await d1Exec(w.dir, "DELETE FROM operations");
  mock.balance = 7.5;
  const r = await call("/api/collect", { method: "POST", body: {}, headers: T() });
  assert.ok(r.j.actions.some((a) => a.includes("balance floor")), JSON.stringify(r.j));
  assert.equal((await waitOp(cid, "down")).status, "done");
  assert.equal(mock.pods.size, 0);
  assert.ok(!mock.log.some((l) => l.method === "DELETE" && /ext/.test(l.path)), "external pods untouched");
  mock.balance = 50;
});

await step("import a runpod-cluster.sh state (and stop it through the controller)", async () => {
  const mk = async (body) => (await (await fetch(`http://127.0.0.1:${mock.port}/rp/rest/pods`, { method: "POST", headers: { authorization: `Bearer ${SECRETS.RUNPOD_API_KEY}`, "content-type": "application/json" }, body: JSON.stringify(body) })).json()).id;
  const g = await mk({ name: "fv-cluster-gw-1", computeType: "CPU", env: { FV_GATEWAY_TOML_B64: "eA==", FV_ADMIN_TOKEN: "fvadm_legacy" }, imageName: "ghcr.io/x@sha256:" + "a".repeat(64) });
  const wk = await mk({ name: "fv-cluster-wan-1", computeType: "GPU", env: { FV_INTERNAL_TOKEN: "legacytok" }, imageName: "ghcr.io/x@sha256:" + "b".repeat(64) });
  mock.adminToken = "fvadm_legacy";
  const state = {
    image: "ghcr.io/x@sha256:" + "a".repeat(64), images: {}, internal_token: "legacytok", url_signing_key: "sig", admin_token: "fvadm_legacy", auth: "keys", min_balance: "8.25",
    deadline: Math.floor(Date.now() / 1000) + 3600, gateway: { pod: g, cpu: "cpu3c", dph: 0.06, created: 1, dc: "EUR-IS-1", image: "ghcr.io/x@sha256:" + "a".repeat(64) },
    gateway_url: `http://127.0.0.1:${mock.port}/pod/${g}`, workers: { wan: { pod: wk, gpu: "RTX", dc: "EUR-IS-1", dph: 2.09, created: 1, image: "ghcr.io/x@sha256:" + "b".repeat(64), url: `http://127.0.0.1:${mock.port}/pod/${wk}` } },
  };
  const r = await call("/api/clusters/import", { method: "POST", body: { name: "legacy", state }, headers: T() });
  assert.equal(r.status, 201, JSON.stringify(r.j));
  const id = r.j.cluster.id;
  assert.equal(r.j.cluster.status, "running");
  assert.ok(!JSON.stringify(r.j).includes("legacytok"));
  const tok = await call(`/api/clusters/${id}/admin-token`, { method: "POST", body: {}, headers: T() });
  assert.equal(tok.j.admin_token, "fvadm_legacy");
  const e = (await call(`/api/clusters/${id}/env`, { headers: T() })).j;
  assert.equal(e.needs_restart.length, 2, "log shipping and the controller's env are new to the imported pods");
  assert.equal((await call(`/api/clusters/${id}/stop`, { method: "POST", body: {}, headers: T() })).status, 202);
  assert.equal((await waitOp(id, "down")).status, "done");
  assert.ok(!mock.pods.has(g) && !mock.pods.has(wk));
});

await step("audit log; no secret in any response", async () => {
  const a = (await call("/api/audit?limit=500", { headers: T() })).j.audit;
  const actions = new Set(a.map((x) => x.action));
  for (const want of ["auth.login", "token.mint", "cluster.define", "cluster.up", "env.set", "cluster.restart", "cluster.scale", "cluster.roll", "cluster.extend", "release.promote", "cluster.stop", "cluster.import", "cluster.admin-token.reveal", "doc.save", "doc.restore"]) assert.ok(actions.has(want), `audit has ${want}`);
  assert.ok(a.some((x) => x.action === "auth.login" && x.ok === 0), "failed logins are audited");
  const envSet = a.find((x) => x.action === "env.set" && x.target.includes("HF_TOKEN"));
  assert.ok(!envSet.after.includes("hf_supersecret"));
  const all = bodies.join("\n");
  for (const [k, v] of Object.entries(SECRETS)) assert.ok(!all.includes(v), `${k} leaked into a response`);
  assert.ok(!all.includes("hf_supersecret"), "a secret env value leaked");
  assert.ok(!all.includes(gw.env.FV_INTERNAL_TOKEN), "a cluster internal token leaked");
  assert.ok(!all.includes(gw.env.FV_LOG_SHIP_TOKEN), "an ingest token leaked");
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
