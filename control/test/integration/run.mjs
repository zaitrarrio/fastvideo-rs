// Integration test: the Worker under `wrangler dev` (workerd, local D1, R2,
// Durable Objects) against mocked Runpod / GHCR / GitHub / Cloudflare APIs
// and mocked worker pods and edge Worker (test/harness.mjs).
//   node test/integration/run.mjs            (npm run test:integration)
//   FVC_UI=1 …                               also keep it up for test/ui/smoke.mjs
import assert from "node:assert/strict";
import { d1Exec, hashPassphrase, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.cloudriftKey = SECRETS.CLOUDRIFT_API_KEY;
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  // The edge stand-in (test/harness.mjs `/edge/*`).
  EDGE_URL: `http://127.0.0.1:${mock.port}/edge`,
  EDGE_INTERNAL_TOKEN: mock.edgeInternal,
  EDGE_ADMIN_TOKEN: mock.edgeAdmin,
  EDGE_D1_DATABASE_ID: "d1-edge-staging",
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
  assert.ok(op.log.some((l) => /ready front at the edge: 1\/1/.test(l.msg)), JSON.stringify(op.log.slice(-4)));
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
  assert.deepEqual(prov.map((p) => [p.id, p.enabled]), [["runpod", true], ["cloudrift", true]]);
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
  assert.ok(op.log.some((l) => /ready front at the edge: 1\/1/.test(l.msg)), JSON.stringify(op.log.slice(-4)));
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
  mock.noStockDcs.add("EU-RO-1");
  const up1 = await call("/api/build-pods/up", { method: "POST", body: {}, headers: T() });
  assert.equal(up1.status, 201, JSON.stringify(up1.j));
  mock.noStockDcs.clear();
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
