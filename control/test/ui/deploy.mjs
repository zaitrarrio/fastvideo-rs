// UI test (headless Chromium, playwright-core), owner requests of 2026-10-18:
//  1. NVIDIA Brev disk size: the standalone launch form shows "Disk (Brev VM)" for provider brev with the picked
//     instance type's rule (elastic range / fixed size), refuses a size the type or the weights do not allow at the
//     field, and the chosen size reaches Brev's create request (diskStorage) and is shown back on the pod's page; a
//     cluster pool on Brev has the same field on the configuration page.
//  2. One-click deploy (#/deploy): several families, each with its pools (a template or a preset) and its instance
//     types from the launch form's table (Runpod: an ordered multi-pick with a fallback; Brev: one type with its
//     disk), the plan reviewed before anything is saved, a new cluster defined and started (the Brev pool's disk in
//     its create request), then pools added to that existing cluster.
// Screenshots go to test-results/; with FV_UI_SCREENSHOTS=1 also to docs/control/img/ (tracked files). Simulated
// upstreams only (test/harness.mjs).
//   CHROMIUM=/opt/pw-browsers/chromium node test/ui/deploy.mjs
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
const img = `${HERE}../docs/control/img`;
mkdirSync(out, { recursive: true });
const SHOTS = process.env.FV_UI_SCREENSHOTS === "1";
if (SHOTS) mkdirSync(img, { recursive: true });
const mock = await startMock();
// The wan preset's trees in the mock R2 bucket (as instance-tables.mjs): one-file manifests pinned by WEIGHTS_R2_EXTRA_TREES.
const WAN_TREES = ["fastwan22-ti2v-5b", "wan22-ti2v-5b"];
function mockManifest(dest) {
  const sha = (b) => createHash("sha256").update(b).digest("hex");
  const f = Buffer.from(`{"tree": "${dest}"}`);
  const m = Buffer.from(JSON.stringify({ format: "fv-weights-manifest/1", dest, rev: "rev1", part_size: 1 << 20, total_bytes: f.length, files: [{ path: "config.json", size: f.length, sha256: sha(f), parts: [sha(f)] }] }));
  mock.r2.objects.set(`manifests/${dest}/rev1.json`, m);
  mock.r2.objects.set(`weights/${dest}/rev1/config.json`, f);
  return { rev: "rev1", files: 1, bytes: f.length, manifest_sha256: sha(m), source: "test" };
}
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.brevToken = SECRETS.BREV_API_TOKEN;
mock.brevOrg = "org-ui-deploy";
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  BREV_ORG_ID: mock.brevOrg,
  BREV_INSTANCE_TYPES: "a100-test,g5.xlarge-test",
  BREV_BUDGET_USD: "60",
  WEIGHTS_R2_ENDPOINT: `http://127.0.0.1:${mock.port}`,
  WEIGHTS_R2_BUCKET: mock.r2.bucket,
  WEIGHTS_R2_ACCESS_KEY_ID: mock.r2.access,
  WEIGHTS_R2_SECRET_ACCESS_KEY: mock.r2.secret,
  WEIGHTS_R2_AUX: "0",
  WEIGHTS_R2_EXTRA_TREES: JSON.stringify(Object.fromEntries(WAN_TREES.map((d) => [d, mockManifest(d)]))),
});
const B = w.url;
let browser;
const fail = (e) => {
  console.log(`FAIL ${e.stack || e.message}`);
  console.log(w.output().slice(-2000));
  browser?.close();
  w.stop();
  mock.close();
  process.exit(1);
};
/** A screenshot in test-results/ and, with FV_UI_SCREENSHOTS=1, in docs/control/img/. */
async function shot(target, name) {
  await target.screenshot({ path: `${out}/${name}`, ...(target.url ? { fullPage: true } : {}) });
  if (SHOTS) copyFileSync(`${out}/${name}`, `${img}/${name}`);
}
try {
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  await page.goto(B + "/");
  await page.waitForSelector("#pass");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");
  const get = (p) => page.evaluate(async (u) => (await fetch(u, { credentials: "same-origin" })).json(), p);
  const waitOp = async (clusterId, kind) => {
    for (let i = 0; i < 180; i++) {
      const ops = (await get(`/api/v1/clusters/${clusterId}/ops`)).operations || [];
      const op = ops.find((o) => o.kind === kind);
      if (op && op.status !== "running") return op;
      await page.waitForTimeout(500);
    }
    throw new Error(`operation ${kind} of ${clusterId} did not finish`);
  };
  const creates = () => mock.brev.workspaces.map((x) => ({ name: x.name, type: x.instanceType, disk: x.body?.diskStorage }));

  // ================================================================ 1. Brev disk: the standalone launch form
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "brevdisk");
  await page.selectOption(`${SL} [data-ctl][data-path="preset"]`, JSON.stringify("wan"));
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, JSON.stringify("brev"));
  const BT = `${SL} [data-instance-table="brev"]`;
  await page.waitForSelector(`${BT} tr[data-row="g5.xlarge-test"]`, { timeout: 20000 });
  await page.waitForFunction((s) => !document.querySelector(`${s} [data-provider-status]`)?.hasAttribute("data-loading"), SL, { timeout: 15000 });
  const disk = `${SL} .ff-field[data-path="container_disk_gb"]`;
  await page.click(`${BT} tr[data-row="g5.xlarge-test"]`);
  await page.waitForSelector(`${disk}:has-text("Disk (Brev VM)")`);
  await page.waitForSelector(`${disk}:has-text("an elastic disk of 10–16384 GB")`);
  await page.click(`${SL} [data-ctl][data-path="weights_source"] button[role="radio"]:text-is("r2")`);
  // Too small for the VM and the wan trees: refused at the field.
  await page.fill(`${SL} [data-ctl][data-path="container_disk_gb"]`, "40");
  await page.waitForSelector(`${disk}.has-err .cf-err:not([hidden]):text-matches("too small for the VM", "i")`, { timeout: 15000 });
  // A fixed-disk type (a100-test: 850 GiB): its size only.
  await page.click(`${BT} tr[data-row="a100-test"]`);
  await page.waitForSelector(`${disk}:has-text("fixed 850 GB disk")`);
  await page.fill(`${SL} [data-ctl][data-path="container_disk_gb"]`, "500");
  await page.waitForSelector(`${disk}.has-err .cf-err:not([hidden]):text-matches("fixed 850 GB disk", "i")`, { timeout: 15000 });
  await shot(page.locator(`${SL} .ff-section:has(h3:text('Where'))`), "brev-disk-fixed-refused.png");
  // Back to the elastic type, 750 GB: valid; Launch.
  await page.click(`${BT} tr[data-row="g5.xlarge-test"]`);
  await page.fill(`${SL} [data-ctl][data-path="container_disk_gb"]`, "750");
  await page.waitForSelector(`${disk}:not(.has-err)`, { timeout: 15000 });
  await page.waitForFunction((s) => !document.querySelector(`${s} button[type=submit]`)?.disabled, SL, { timeout: 15000 });
  await shot(page.locator(`${SL} .ff-section:has(h3:text('Where'))`), "brev-disk-standalone-form.png");
  await page.click(`${SL} button[type=submit]`);
  await page.waitForFunction(() => location.hash.includes("standalone?id=brevdisk"), null, { timeout: 30000 });
  const sp = (await get("/api/v1/standalone/brevdisk")).pod;
  const op1 = await waitOp(sp.id, "up");
  assert.equal(op1.status, "done", `up: ${op1.error}`);
  assert.deepEqual(creates(), [{ name: creates()[0].name, type: "g5.xlarge-test", disk: "750Gi" }], "the chosen size is the create's diskStorage");
  // Shown back: the definition's disk and the VM's disk as created.
  await page.goto(`${B}/#/standalone?id=brevdisk`);
  await page.waitForSelector("[data-brev-disk]", { timeout: 20000 });
  assert.match(await page.textContent("[data-brev-disk]"), /^750 GB · \$[\d.]+\/day storage while stopped$/);
  assert.equal(await page.textContent("[data-def-disk]"), "750 GB (the Brev VM's disk)");
  await shot(page, "brev-disk-standalone-page.png");

  // ================================================================ 1b. Brev disk: a cluster pool (configuration page)
  const mk = await page.evaluate(async () => {
    const me = await (await fetch("/api/v1/auth/me", { credentials: "same-origin" })).json();
    const r = await fetch("/api/v1/clusters", {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json", "x-csrf-token": me.csrf },
      body: JSON.stringify({ spec: { name: "brevpool", control_plane: "direct", pools: [{ id: "wb", variant: "wan5b", count: 1, compute: "GPU", config: "/etc/fv/runpod-wan5b.toml", models: [{ id: "wan22-ti2v-5b", family: "wan", recipe: "wan-max" }], provider: "brev", provider_gpu: "g5.xlarge-test", weights_source: "r2", container_disk_gb: 900 }] } }),
    });
    return { status: r.status, j: await r.json() };
  });
  assert.equal(mk.status, 201, JSON.stringify(mk.j));
  await page.goto(`${B}/#/cluster/${mk.j.cluster.id}/config`);
  await page.waitForSelector(".cf-pool");
  if ((await page.getAttribute(".cf-pool .cf-caret", "aria-expanded")) !== "true") await page.click(".cf-pool .cf-caret");
  const pf = '.cf-field:has(input[aria-label="pools.0.container_disk_gb"])';
  await page.waitForSelector(`${pf}:has-text("Disk (GB, the Brev VM)")`, { timeout: 15000 });
  assert.equal(await page.inputValue('input[aria-label="pools.0.container_disk_gb"]'), "900");
  await page.waitForSelector(`${pf}:has-text("an elastic disk of 10-16384 GB")`, { timeout: 15000 });
  await shot(page.locator(".cf-pool").first(), "brev-disk-cluster-pool.png");

  // ================================================================ 2. One-click deploy
  await page.goto(`${B}/#/deploy`);
  await page.waitForSelector("[data-deploy-family=h3]", { timeout: 20000 });
  await page.fill("[data-deploy-name]", "oneclick");
  await page.selectOption('select[aria-label="Control plane"]', "direct");
  // All families: every family's card; the h3 card offers the templates (h3, fasth3) and the presets.
  await page.check("[data-deploy-all]");
  for (const f of ["h3", "ltx2", "wan"]) await page.waitForSelector(`[data-deploy-family-card=${f}]`);
  const h3opts = await page.$$eval("[data-deploy-pools=h3] option", (els) => els.map((e) => e.value));
  for (const v of ["h3-turbo", "vdn-h3", "template:h3", "template:fasth3"]) assert.ok(h3opts.includes(v), `h3 offers ${v}: ${h3opts}`);
  const wanOpts = await page.$$eval("[data-deploy-pools=wan] option", (els) => els.map((e) => e.value));
  assert.ok(wanOpts.includes("template:streaming"), `wan offers the streaming template: ${wanOpts}`);
  await page.selectOption("[data-deploy-pools=h3]", "template:fasth3");
  await page.waitForSelector("[data-deploy-family-card=h3]:has-text('pools: h3-turbo, fasth3-trim')");
  await shot(page, "deploy-all-families.png");
  // Two families only: ltx2 (the ltx template, Runpod: a primary GPU and a fallback) and wan (its pool on Brev).
  await page.uncheck("[data-deploy-all]");
  await page.uncheck("[data-deploy-family=h3]");
  await page.selectOption("[data-deploy-pools=ltx2]", "template:ltx");
  const RT = "[data-deploy-family-card=ltx2] [data-instance-table=runpod]";
  await page.waitForSelector(`${RT} table.it-table tr.it-can`, { timeout: 20000 });
  const rows = await page.$$eval(`${RT} tr.it-can`, (els) => els.map((e) => e.getAttribute("data-row")));
  assert.ok(rows.length >= 2, `two launchable Runpod rows: ${rows}`);
  await page.click(`${RT} tr[data-row="${rows[0]}"]`);
  await page.click(`${RT} tr[data-row="${rows[1]}"] [data-it-add]`);
  await page.waitForSelector("[data-deploy-picked=ltx2]:has-text('→')");
  const ltxPicked = await page.textContent("[data-deploy-picked=ltx2]");
  await page.selectOption("[data-deploy-provider=wan]", "brev");
  const WT = "[data-deploy-family-card=wan] [data-instance-table=brev]";
  await page.waitForSelector(`${WT} tr[data-row="g5.xlarge-test"]`, { timeout: 20000 });
  await page.click(`${WT} tr[data-row="g5.xlarge-test"]`);
  await page.selectOption('select[aria-label="wan weights"]', "r2");
  await page.fill("[data-deploy-disk=wan]", "600");
  await page.waitForSelector("[data-deploy-family-card=wan]:has-text('an elastic disk of 10–16384 GB')");
  await shot(page, "deploy-families-instance-types.png");
  // Review: what will be created, before anything is saved.
  await page.click("[data-deploy-review]");
  await page.waitForSelector("[data-deploy-review-card]", { timeout: 30000 });
  const rv = await page.textContent("[data-deploy-review-card]");
  assert.ok(rv.includes("valid"), `plan valid: ${rv}`);
  const poolRows = await page.$$eval("[data-deploy-pools-table] tbody tr", (els) => els.map((e) => [...e.children].map((c) => c.textContent)));
  assert.deepEqual(poolRows.map((r) => r[0]), ["ltx", "ltx-pro", "ltx-a2v", "ltx-ref2v", "wan"]);
  assert.match(poolRows[4][4], /^brev: g5\.xlarge-test · disk 600 GB/);
  assert.ok(poolRows[0][4].includes("→"), `the ltx pools' GPU types in order: ${poolRows[0][4]} (picked: ${ltxPicked})`);
  assert.equal((await get("/api/v1/clusters")).clusters.filter((c) => c.name === "oneclick").length, 0, "nothing saved by Review");
  await shot(page, "deploy-review.png");
  // Deploy: defined and started; the Brev pool's disk reaches its create request.
  await page.click("[data-deploy-apply]");
  await page.waitForSelector("[data-deploy-result]", { timeout: 30000 });
  const cl = (await get("/api/v1/clusters")).clusters.find((c) => c.name === "oneclick");
  assert.deepEqual(cl.spec.pools.map((p) => p.id), ["ltx", "ltx-pro", "ltx-a2v", "ltx-ref2v", "wan"]);
  assert.equal(cl.spec.pools[0].gpu_types.length, 2, "primary + fallback");
  assert.deepEqual([cl.spec.pools[4].provider, cl.spec.pools[4].provider_gpu, cl.spec.pools[4].container_disk_gb], ["brev", "g5.xlarge-test", 600]);
  const op2 = await waitOp(cl.id, "up");
  const wanCreate = creates().find((c) => c.name.includes("oneclick"));
  assert.ok(wanCreate, `a Brev create for the cluster's wan pool (up: ${op2.status} ${op2.error || ""}): ${JSON.stringify(creates())}`);
  assert.equal(wanCreate.disk, "600Gi", "the pool's disk is the create's diskStorage");
  await shot(page, "deploy-result.png");
  // Into the existing cluster: h3's default pool is added (the plan names the target).
  await page.click("[data-deploy-result] button:text('Deploy more')");
  await page.click("[data-deploy-target=existing]");
  await page.selectOption('select[aria-label="Existing cluster"]', cl.id);
  await page.uncheck("[data-deploy-family=ltx2]");
  await page.uncheck("[data-deploy-family=wan]");
  await page.check("[data-deploy-family=h3]");
  await page.selectOption("[data-deploy-pools=h3]", "h3-turbo");
  await page.click("[data-deploy-review]");
  await page.waitForSelector("[data-deploy-review-card]:has-text('Existing cluster oneclick')", { timeout: 30000 });
  await shot(page, "deploy-existing-review.png");
  await page.click("[data-deploy-apply]");
  await page.waitForSelector("[data-deploy-result]", { timeout: 30000 });
  const cl2 = (await get(`/api/v1/clusters/${cl.id}`)).cluster;
  assert.deepEqual(cl2.spec.pools.map((p) => p.id), ["ltx", "ltx-pro", "ltx-a2v", "ltx-ref2v", "wan", "h3-turbo"]);
  assert.deepEqual(errors.filter((e) => !/Failed to load resource/.test(e)), [], `page errors: ${errors}`);
  console.log(`ok   UI Brev disk (standalone form + page, cluster pool) and one-click deploy (families, templates, instance types, review, new + existing cluster) (screenshots in ${out}${SHOTS ? ` and ${img}` : ""})`);
  await browser.close();
  w.stop();
  mock.close();
} catch (e) {
  fail(e);
}
