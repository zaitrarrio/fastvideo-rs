// UI test (headless Chromium, playwright-core): the standalone launch screen loads no provider details on open (owner
// 2026-10-12: "we should not load provider details / prices until a provider is selected"). Opening it asks only the
// lists without live ones (?live=0) and the validator with ?live=0: no Runpod / GMI / Brev / GHCR call reaches the
// (simulated) upstreams. Picking a provider asks exactly that provider's details (?provider=<p>) once per page
// session, with a loading line; a pick changed while a load is in flight cancels it. Simulated upstreams only.
//   CHROMIUM=/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell node test/ui/lazy-providers.mjs
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.gmiKey = SECRETS.GMI_API_KEY;
mock.brevToken = SECRETS.BREV_API_TOKEN;
mock.brevOrg = "org-ui-lazy";
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  GMI_PRODUCTS: "container.h200.x1",
  GMI_DEFAULT_IDC: "us-denver-1",
  GMI_BUDGET_USD: "50",
  BREV_ORG_ID: mock.brevOrg,
  BREV_INSTANCE_TYPES: "a100-test,g5.xlarge-test",
  BREV_BUDGET_USD: "40",
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
/** The upstream a mock request belongs to (null: not a provider / registry API). */
const upstreamOf = (p) => (p.startsWith("/rp/") ? "runpod" : p.startsWith("/gmi/") ? "gmi" : p.startsWith("/brev/") ? "brev" : p.startsWith("/ghcr/") ? "ghcr" : null);
try {
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  /** The page's /api requests (method, path and query) and the failed (cancelled) ones. */
  const reqs = [];
  const failed = [];
  page.on("request", (r) => r.url().includes("/api/") && reqs.push(`${r.method()} ${r.url().slice(B.length)}`));
  page.on("requestfailed", (r) => r.url().includes("/api/") && failed.push(`${r.method()} ${r.url().slice(B.length)} ${r.failure()?.errorText}`));
  await page.goto(B + "/");
  await page.waitForSelector("#pass");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(500);

  // ---- open the launch screen: no provider detail anywhere
  reqs.length = 0;
  let mark = mock.log.length;
  const t0 = Date.now();
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  const openMs = Date.now() - t0;
  // The validator runs once (debounced) after the form draws: wait for it, then for anything else to settle.
  await page.waitForResponse((r) => r.url().includes("/api/standalone/validate"), { timeout: 10000 });
  await page.waitForTimeout(800);
  const opened = [...reqs];
  const upOpen = mock.log.slice(mark).map((x) => upstreamOf(x.path)).filter(Boolean);
  console.log(`  open: form in ${openMs} ms; requests: ${opened.join(", ")}`);
  assert.ok(!opened.some((r) => /[?&]provider=/.test(r)), `no provider details on open: ${opened}`);
  assert.ok(!opened.some((r) => /\/api\/providers/.test(r)), `no provider views on open: ${opened}`);
  assert.ok(opened.includes("GET /api/schemas/dynamic?live=0"), `the lists without live ones: ${opened}`);
  assert.ok(!opened.some((r) => r === "GET /api/schemas/dynamic" || r.startsWith("GET /api/schemas/dynamic?cluster")), `not the full (live) lists: ${opened}`);
  assert.ok(opened.filter((r) => r.startsWith("POST /api/standalone/validate")).every((r) => r === "POST /api/standalone/validate?live=0"), `the validator without live checks: ${opened}`);
  assert.deepEqual(upOpen, [], `no upstream API call on open: ${upOpen}`);
  // The form says to pick a provider; its fields (and Launch) wait for it.
  assert.equal(await page.$eval(`${SL} [data-ctl][data-path="provider"]`, (e) => e.value), "");
  assert.match(await page.textContent(`${SL} [data-provider-status]`), /Pick a provider/);
  assert.equal(await page.$(`${SL} [data-instance-table]`), null, "no GPU types (the instance table) before a pick");
  assert.equal(await page.$(`${SL} .ff-field[data-path="weights_source"]`), null, "no weights before a pick");
  assert.equal(await page.$(`${SL} .ff-field[data-path="provider_gpu"]`), null);
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "lazy-ui");
  await page.waitForSelector(`${SL} .ff-field[data-path="provider"].has-err .cf-err:not([hidden]):text-matches("choose a provider")`);
  assert.ok(await page.$eval(`${SL} button[type=submit]`, (b) => b.disabled), "Launch waits for a provider");
  await page.screenshot({ path: `${out}/40-launch-open-no-provider.png`, fullPage: true });

  // ---- pick Brev: exactly Brev's details (one request), a loading line, then its instance types with prices
  const pick = async (p) => {
    reqs.length = 0;
    mark = mock.log.length;
    await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, p ? JSON.stringify(p) : "");
  };
  const settle = async () => {
    await page.waitForFunction((s) => !document.querySelector(`${s} [data-provider-status]`)?.hasAttribute("data-loading"), SL, { timeout: 15000 });
    await page.waitForTimeout(800);
  };
  // Slow Brev's details down so the loading line is seen.
  await page.route("**/api/schemas/dynamic?provider=brev", async (r) => (await new Promise((res) => setTimeout(res, 600)), r.continue()));
  await pick("brev");
  await page.waitForSelector(`${SL} [data-provider-status][data-loading="brev"]:text-matches("loading brev")`);
  await settle();
  await page.unroute("**/api/schemas/dynamic?provider=brev");
  const brevReqs = reqs.filter((r) => r.startsWith("GET /api/schemas/dynamic") || r.startsWith("GET /api/providers"));
  // Brev's details and its instance-type table (owner 2026-10-13: the table loads when the provider is picked).
  assert.deepEqual(brevReqs.sort(), ["GET /api/providers/brev/instance-types", "GET /api/schemas/dynamic?provider=brev"], `only Brev's details and table: ${reqs}`);
  const upBrev = [...new Set(mock.log.slice(mark).map((x) => upstreamOf(x.path)).filter(Boolean))];
  // (Brev's type list may already be in the Worker's 1 h cache: no call at all is fine; any other provider is not.)
  assert.ok(!upBrev.includes("runpod") && !upBrev.includes("gmi") && !upBrev.includes("ghcr"), `no other provider is asked: ${upBrev}`);
  const opts = await page.$$eval(`${SL} .ff-field[data-path="provider_gpu"] option`, (els) => els.map((e) => `${e.value} ${e.textContent}`));
  assert.ok(opts.some((o) => /^g5\.xlarge-test .*\$1\.01\/hr/.test(o)), JSON.stringify(opts));
  // Owner decision 2026-10-13: a type Brev cannot stop is listed too, with a warning.
  assert.ok(opts.some((o) => /^a100-test .*\$1\.80\/hr · WARNING: can't be stopped: a stop deletes it \(its disk and weights go too\)/.test(o)), JSON.stringify(opts));
  assert.match(await page.textContent(`${SL} [data-provider-status]`), /2 instance types with live prices · this month \$0\.00 of the \$40 budget · no parked VMs/);
  await page.screenshot({ path: `${out}/41-launch-brev-picked.png`, fullPage: true });

  // ---- pick GMI: GMI's details only
  await pick("gmi");
  await settle();
  assert.deepEqual(reqs.filter((r) => r.startsWith("GET /api/schemas/dynamic")), ["GET /api/schemas/dynamic?provider=gmi"], `only GMI's details: ${reqs}`);
  assert.deepEqual(reqs.filter((r) => r.startsWith("GET /api/providers")), ["GET /api/providers/gmi/instance-types"], `GMI's table only: ${reqs}`);
  const upGmi = [...new Set(mock.log.slice(mark).map((x) => upstreamOf(x.path)).filter(Boolean))];
  assert.ok(upGmi.every((x) => x === "gmi"), `GMI only: ${upGmi}`);
  assert.ok((await page.$$eval(`${SL} .ff-field[data-path="provider_gpu"] option`, (els) => els.map((e) => e.textContent))).some((t) => /GMI Cloud · \$3\.20\/hr · available/.test(t)));

  // ---- back to Brev: kept for the session, no request
  await pick("brev");
  await settle();
  assert.deepEqual(reqs.filter((r) => r.startsWith("GET /api/schemas/dynamic")), [], `Brev's details are kept: ${reqs}`);
  // The table loads again on each switch (the Worker's cache answers it).
  assert.deepEqual(reqs.filter((r) => r.startsWith("GET /api/providers")), ["GET /api/providers/brev/instance-types"], `Brev's table again: ${reqs}`);

  // ---- Runpod picked while its load is in flight, then GMI (kept): the Runpod load is cancelled; a re-pick asks again
  await page.route("**/api/schemas/dynamic?provider=runpod", async (r) => (await new Promise((res) => setTimeout(res, 1500)), r.continue().catch(() => {})));
  await page.route("**/api/providers/runpod/instance-types", async (r) => (await new Promise((res) => setTimeout(res, 1500)), r.continue().catch(() => {})));
  failed.length = 0;
  await pick("runpod");
  await page.waitForSelector(`${SL} [data-provider-status][data-loading="runpod"]`);
  await pick("gmi");
  await settle();
  await page.waitForFunction(() => true);
  await page.waitForTimeout(1800);
  assert.ok(failed.some((f) => f.startsWith("GET /api/schemas/dynamic?provider=runpod")), `the stale Runpod load is cancelled: ${failed}`);
  assert.ok(failed.some((f) => f.startsWith("GET /api/providers/runpod/instance-types")), `the stale Runpod table load is cancelled: ${failed}`);
  assert.match(await page.textContent(`${SL} [data-provider-status]`), /products? with live prices/, "GMI's line, not Runpod's");
  await page.unroute("**/api/schemas/dynamic?provider=runpod");
  await page.unroute("**/api/providers/runpod/instance-types");
  await pick("runpod");
  await settle();
  assert.deepEqual(reqs.filter((r) => r.startsWith("GET /api/schemas/dynamic")), ["GET /api/schemas/dynamic?provider=runpod"], `Runpod asked again: ${reqs}`);
  const upRp = [...new Set(mock.log.slice(mark).map((x) => upstreamOf(x.path)).filter(Boolean))];
  assert.ok(upRp.every((x) => x === "runpod"), `Runpod only: ${upRp}`);
  // Runpod's fields: the instance table (the GPU pick: no separate GPU types list), the weights (the preset's
  // default: h3-turbo reads the volume) and the volume's region.
  await page.waitForSelector(`${SL} [data-instance-table="runpod"] table.it-table`);
  assert.equal(await page.$(`${SL} .ff-field[data-path="gpu_types"]`), null, "the table replaces the GPU types list");
  assert.equal(await page.textContent(`${SL} [data-ctl][data-path="weights_source"] [aria-checked="true"]`), "volume");
  await page.waitForSelector(`${SL} .ff-field[data-path="region"]`);
  assert.match(await page.textContent(`${SL} [data-provider-status]`), /GPU types with live \$\/hr and stock/);
  // The validator now runs live (a provider is picked), and Launch turns on.
  assert.ok(reqs.some((r) => r === "POST /api/standalone/validate"), `live validation once picked: ${reqs}`);
  await page.waitForFunction((s) => !document.querySelector(`${s} button[type=submit]`).disabled, SL, { timeout: 15000 });
  await page.screenshot({ path: `${out}/42-launch-runpod-picked.png`, fullPage: true });

  assert.deepEqual(errors.filter((e) => !/login required|401/.test(e)), [], "no console errors");
  console.log(`ok   UI lazy provider details: open asks no provider (form in ${openMs} ms), each pick asks only its provider, kept per session, stale loads cancelled (screenshots in ${out})`);
} catch (e) {
  fail(e);
}
await browser.close();
w.stop();
mock.close();
