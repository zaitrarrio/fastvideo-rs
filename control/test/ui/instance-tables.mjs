// UI test (headless Chromium, playwright-core): the launch screen's instance-type tables (owner 2026-10-13: "change
// the instance type selection into tables with all instance types supported by that provider … sortable, searchable,
// filterable, paginated. We load the table when switching providers or using a manual refresh button"). Checks: no
// table loads on open; picking a provider loads its table (one request, no refresh); sort (aria-sort), search,
// filters and pagination; a launchable row's click selects it, a non-launchable one does not; the selection persists
// across sort / filter / page and the launch submits the selected type; Refresh asks ?refresh=1 (and a second one
// within 30 s is served cached, saying so); switching provider swaps the tables and cancels a stale load.
// Runpod (owner 2026-10-13: "I should be able to launch Runpod in any region if I am using R2"): the Weights select
// recomputes the rows' launchable flags without a refetch (volume: EU-RO-1 only; r2: every NVIDIA Secure Cloud row);
// the table is an ordered multi-pick (click = primary, "+ fallback" / Shift+Enter appends, the Selected GPUs strip
// reorders and removes) and the launch posts gpu_types in that order with the data centres (r2 in US data centres).
// Every check waits for the state it asserts (the form's value as the strip / rows show it), not for time.
// Screenshots of each provider's table go to docs/control/img/ only with FV_UI_SCREENSHOTS=1 (they are tracked
// files: regenerate them deliberately when the table changes). Simulated upstreams only.
//   CHROMIUM=/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell node test/ui/instance-tables.mjs
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
const img = `${HERE}../docs/control/img`;
mkdirSync(out, { recursive: true });
const SHOTS = process.env.FV_UI_SCREENSHOTS === "1";
if (SHOTS) mkdirSync(img, { recursive: true });
/** A tracked screenshot (docs/control/img): only when asked (FV_UI_SCREENSHOTS=1). */
const docShot = async (el, name) => SHOTS && (await el.screenshot({ path: `${img}/${name}` }));
const mock = await startMock();
// The wan preset's trees in the mock R2 bucket (as explorer.mjs): one-file manifests pinned by WEIGHTS_R2_EXTRA_TREES.
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
mock.gmiKey = SECRETS.GMI_API_KEY;
mock.brevToken = SECRETS.BREV_API_TOKEN;
mock.brevOrg = "org-ui-tables";
// A third Runpod data centre (no weights volume there): H100 and H200 in stock, an AMD GPU (never launchable).
mock.runpodDcs.push({ id: "US-TX-3", name: "US-TX-3", location: "United States", storageSupport: true, listed: true, gpuAvailability: [
  { available: true, stockStatus: "Low", gpuTypeId: "NVIDIA H100 80GB HBM3" },
  { available: true, stockStatus: "High", gpuTypeId: "NVIDIA H200" },
] });
// Brev: 30 more types (pages of 10 / 25), some CPU-only, one AMD, two not stoppable; only two are allowed.
const gpus = [["H100", 80, 2.9], ["L40S", 48, 1.6], ["T4", 16, 0.5], ["A10G", 24, 1.1], ["RTX PRO Server 6000", 96, 3.4], ["V100", 16, 0.9]];
for (let i = 0; i < 30; i++) {
  const [g, mem, price] = gpus[i % gpus.length];
  const n = 1 + (i % 4);
  const cpu = i % 10 === 9;
  mock.brev.types.push({
    type: `test-${String(i).padStart(2, "0")}.${cpu ? "cpu" : g.toLowerCase().replace(/\s+/g, "")}x${n}`,
    supported_gpus: cpu ? undefined : [{ count: n, name: i === 7 ? "Radeon Pro V520" : g, manufacturer: i === 7 ? "AMD" : "NVIDIA", memory: `${mem}GiB` }],
    supported_storage: [{ size: "0B", type: "gp3", min_size: "10GiB", max_size: "16TiB", price_per_gb_hr: { currency: "USD", amount: "0.000132" } }],
    base_price: { currency: "USD", amount: (price * n + i / 100).toFixed(6) },
    memory: `${16 * n}GiB`,
    vcpu: 8 * n,
    location: ["us-east-1", "us-west-2", "eu-north1"][i % 3],
    provider: ["aws", "gcp", "nebius"][i % 3],
    stoppable: i % 11 !== 3,
    elastic_root_volume: true,
    estimated_deploy_time: "7m0s",
    is_available: i % 5 !== 0,
    supported_architectures: ["x86_64"],
    cloud_cred_id: "cred",
  });
}
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  GMI_PRODUCTS: "container.h200.x1",
  GMI_DEFAULT_IDC: "us-denver-1",
  GMI_BUDGET_USD: "50",
  BREV_ORG_ID: mock.brevOrg,
  BREV_INSTANCE_TYPES: "a100-test,g5.xlarge-test",
  BREV_BUDGET_USD: "40",
  // The R2 weights store (the mock's bucket), as in explorer.mjs: a Runpod pod with weights_source r2 validates.
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
try {
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  const reqs = [];
  const failed = [];
  page.on("request", (r) => r.url().includes("/api/") && reqs.push(`${r.method()} ${r.url().slice(B.length)}`));
  page.on("requestfailed", (r) => r.url().includes("/api/") && failed.push(`${r.method()} ${r.url().slice(B.length)} ${r.failure()?.errorText}`));
  const typeReqs = () => reqs.filter((r) => r.includes("/instance-types"));
  await page.goto(B + "/");
  await page.waitForSelector("#pass");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");
  await page.waitForLoadState("networkidle");

  // ---- open: no table, no instance-types request
  reqs.length = 0;
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  await page.waitForResponse((r) => r.url().includes("/api/standalone/validate"), { timeout: 10000 });
  await page.waitForTimeout(600);
  assert.deepEqual(typeReqs(), [], `no table on open: ${reqs}`);
  assert.equal(await page.$("[data-instance-table]"), null, "no table before a pick");
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "tables-ui");

  const pick = async (p) => {
    reqs.length = 0;
    await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, JSON.stringify(p));
  };
  const T = (p) => `${SL} [data-instance-table="${p}"]`;
  // Loaded: the table drawn, its load done, and the provider's details (whose arrival redraws the form) in too, so
  // no redraw lands between a check's action and its assertion.
  const loaded = async (p) => {
    await page.waitForSelector(`${T(p)} table.it-table`, { timeout: 15000 });
    await page.waitForFunction((s) => !document.querySelector(`${s} .it-status`)?.hasAttribute("data-loading"), T(p), { timeout: 15000 });
    await page.waitForFunction((s) => !document.querySelector(`${s} [data-provider-status]`)?.hasAttribute("data-loading"), SL, { timeout: 15000 });
  };
  const names = (p) => page.$$eval(`${T(p)} tbody tr`, (rs) => rs.map((r) => r.getAttribute("data-row")));
  const count = (p) => page.textContent(`${T(p)} [data-it-count]`);

  // ---- Brev: the table loads on the pick (one request, no refresh)
  await pick("brev");
  await loaded("brev");
  assert.deepEqual(typeReqs(), ["GET /api/providers/brev/instance-types"], `Brev's table, once: ${reqs}`);
  assert.match(await count("brev"), /^1–25 of 32 · 2 launchable · /, await count("brev"));
  const headers = await page.$$eval(`${T("brev")} thead th`, (ths) => ths.map((t) => t.textContent.trim()));
  for (const h of ["Name", "Spec", "Available", "Cost $/hr", "$/GPU-hr", "Model", "Model family", "RAM", "vCPU", "GPU", "GPU family", "VRAM", "Storage type", "Storage size", "Region", "Stoppable", "Deploy", "Launchable"])
    assert.ok(headers.includes(h), `column ${h}: ${headers}`);
  // The launchable rows come first (the server's order).
  assert.deepEqual((await names("brev")).slice(0, 2).sort(), ["a100-test", "g5.xlarge-test"]);

  // ---- sort: Cost ascending, then descending (aria-sort), unknown last
  const costTh = `${T("brev")} th[data-col="usd_per_hr"]`;
  await page.click(`${costTh} button`);
  assert.equal(await page.getAttribute(costTh, "aria-sort"), "ascending");
  const costs = async () => (await page.$$eval(`${T("brev")} tbody tr td:nth-child(4)`, (tds) => tds.map((t) => Number(t.textContent.replace(/[$,]/g, ""))))).filter(Number.isFinite);
  let c = await costs();
  assert.deepEqual(c, [...c].sort((a, b) => a - b), `ascending: ${c}`);
  await page.click(`${costTh} button`);
  assert.equal(await page.getAttribute(costTh, "aria-sort"), "descending");
  c = await costs();
  assert.deepEqual(c, [...c].sort((a, b) => b - a), `descending: ${c}`);
  assert.equal(await page.getAttribute(`${T("brev")} th[data-col="name"]`, "aria-sort"), "none");

  // ---- select a launchable row (selection persists across sort / filter / page); a non-launchable one is refused
  await page.fill(`${T("brev")} [data-it-search]`, "g5.xlarge");
  assert.deepEqual(await names("brev"), ["g5.xlarge-test"]);
  await page.click(`${T("brev")} tr[data-row="g5.xlarge-test"]`);
  await page.waitForFunction((s) => document.querySelector(`${s} [data-ctl][data-path="provider_gpu"]`)?.value?.includes("g5.xlarge-test"), SL);
  assert.equal(await page.getAttribute(`${T("brev")} tr[data-row="g5.xlarge-test"]`, "aria-selected"), "true");
  assert.match(await page.textContent(`${T("brev")} [data-it-selected]`), /Selected: g5\.xlarge-test · 1× A10G 80 GB/);
  await page.fill(`${T("brev")} [data-it-search]`, "");
  await page.click(`${T("brev")} tr[data-row="test-01.l40sx2"]`, { force: true }); // not in BREV_INSTANCE_TYPES (aria-disabled)
  assert.match(await page.getAttribute(`${T("brev")} tr[data-row="test-01.l40sx2"]`, "title"), /not launchable: not in BREV_INSTANCE_TYPES/);
  assert.match(await page.textContent(`${T("brev")} [data-it-selected]`), /Selected: g5\.xlarge-test/, "a non-launchable row is not selected");
  await page.click(`${T("brev")} th[data-col="name"] button`);
  assert.match(await page.textContent(`${T("brev")} [data-it-selected]`), /Selected: g5\.xlarge-test/, "kept across sort");

  // ---- filters: launchable, stoppable, min VRAM, max cost, model family, GPU family, region, available
  const filt = async (k, v) => {
    const sel = `${T("brev")} [data-it-filter="${k}"]`;
    if ((await page.$eval(sel, (e) => e.tagName)) === "SELECT") await page.selectOption(sel, v);
    else await page.fill(sel, v);
  };
  await filt("launchable", "yes");
  assert.deepEqual((await names("brev")).sort(), ["a100-test", "g5.xlarge-test"]);
  await filt("launchable", "no");
  assert.match(await count("brev"), /of 30 \(filtered from 32\)/);
  await filt("launchable", "");
  await filt("stoppable", "no");
  const nonStop = await names("brev");
  assert.ok(nonStop.includes("a100-test") && nonStop.includes("test-03.a10gx4") && !nonStop.includes("g5.xlarge-test"), `not stoppable: ${nonStop}`);
  await filt("stoppable", "");
  await filt("minVram", "200");
  const big = await page.$$eval(`${T("brev")} tbody tr td:nth-child(12)`, (tds) => tds.map((t) => Number(/(\d+) GB$/.exec(t.textContent)?.[1])));
  assert.ok(big.length > 0 && big.every((v) => v >= 200), `≥ 200 GB VRAM in all: ${big}`);
  await filt("minVram", "");
  await filt("maxCost", "1");
  const cheap = await page.$$eval(`${T("brev")} tbody tr td:nth-child(4)`, (tds) => tds.map((t) => Number(t.textContent.replace("$", ""))));
  assert.ok(cheap.length && cheap.every((x) => x <= 1), `≤ $1/hr: ${cheap}`);
  await filt("maxCost", "");
  await filt("family", "Hopper");
  assert.ok((await names("brev")).every((n) => /h100/.test(n)), await names("brev"));
  await filt("family", "");
  await filt("gpuFamily", "datacenter");
  assert.ok(!(await names("brev")).some((n) => /cpu/.test(n)), "CPU-only rows have no GPU family");
  await filt("gpuFamily", "");
  await filt("region", "eu-north1");
  assert.ok((await page.$$eval(`${T("brev")} tbody tr td:nth-child(15)`, (tds) => tds.map((t) => t.textContent))).every((t) => t.startsWith("eu-north1")));
  await filt("region", "");
  await filt("available", "no");
  assert.match(await count("brev"), /of 6 \(filtered from 32\)/);
  await filt("available", "");
  assert.match(await page.textContent(`${T("brev")} [data-it-selected]`), /Selected: g5\.xlarge-test/, "kept across filters");

  // ---- pagination: 10 per page, next / prev
  await page.selectOption(`${T("brev")} [data-it-pagesize]`, "10");
  assert.equal((await names("brev")).length, 10);
  assert.match(await page.textContent(`${T("brev")} [data-it-page]`), /page 1 of 4/);
  const first = await names("brev");
  await page.click(`${T("brev")} [data-it-next]`);
  assert.match(await page.textContent(`${T("brev")} [data-it-page]`), /page 2 of 4/);
  assert.notDeepEqual(await names("brev"), first);
  assert.match(await page.textContent(`${T("brev")} [data-it-selected]`), /Selected: g5\.xlarge-test/, "kept across pages");
  await page.click(`${T("brev")} [data-it-prev]`);
  assert.deepEqual(await names("brev"), first);

  // ---- keyboard: a focused launchable row selects with Enter
  await page.focus(`${T("brev")} tr[data-row="a100-test"]`);
  await page.keyboard.press("Enter");
  await page.waitForFunction((s) => document.querySelector(`${s} [data-ctl][data-path="provider_gpu"]`)?.value?.includes("a100-test"), SL);
  assert.equal(await page.getAttribute(`${T("brev")} tr[data-row="a100-test"]`, "aria-selected"), "true");
  assert.equal(await page.getAttribute(`${T("brev")} tr[data-row="g5.xlarge-test"]`, "aria-selected"), "false");
  await docShot(await page.$(T("brev")), "instance-table-brev.png");

  // ---- Refresh: ?refresh=1, a new "updated" time; a second one within 30 s is served cached and says so
  const at0 = Number(await page.getAttribute(`${T("brev")} .it-status`, "data-at"));
  const brevCalls0 = mock.brev.calls.filter((x) => x.path.startsWith("/instances/alltypesavailable")).length;
  reqs.length = 0;
  await page.waitForTimeout(20);
  await page.click(`${T("brev")} [data-it-refresh]`);
  await loaded("brev");
  assert.deepEqual(typeReqs(), ["GET /api/providers/brev/instance-types?refresh=1"], `refresh: ${reqs}`);
  assert.equal(mock.brev.calls.filter((x) => x.path.startsWith("/instances/alltypesavailable")).length, brevCalls0 + 1, "the refresh reads Brev again");
  assert.ok(Number(await page.getAttribute(`${T("brev")} .it-status`, "data-at")) > at0, "a newer list");
  assert.doesNotMatch(await page.textContent(`${T("brev")} .it-status`), /cached/);
  await page.click(`${T("brev")} [data-it-refresh]`);
  await loaded("brev");
  assert.match(await page.textContent(`${T("brev")} .it-status`), /\(cached\) · refreshed recently: the cached list \(again in \d+ s\)/);
  assert.equal(mock.brev.calls.filter((x) => x.path.startsWith("/instances/alltypesavailable")).length, brevCalls0 + 1, "rate-limited: no second read");
  assert.match(await page.textContent(`${T("brev")} [data-it-selected]`), /Selected: a100-test/, "kept across refresh");

  // ---- switch to GMI: its table replaces Brev's
  await pick("gmi");
  await loaded("gmi");
  assert.equal(await page.$(T("brev")), null, "Brev's table is gone");
  assert.deepEqual(typeReqs(), ["GET /api/providers/gmi/instance-types"]);
  assert.deepEqual((await names("gmi")).sort(), ["container.b200.x1@us-denver-1", "container.h200.x1@us-denver-1"]);
  assert.match(await page.getAttribute(`${T("gmi")} tr[data-row="container.b200.x1@us-denver-1"]`, "title"), /not in GMI_PRODUCTS/);
  await page.click(`${T("gmi")} tr[data-row="container.h200.x1@us-denver-1"]`);
  await page.waitForFunction((s) => document.querySelector(`${s} [data-ctl][data-path="provider_gpu"]`)?.value?.includes("container.h200.x1"), SL);
  assert.match(await page.textContent(`${T("gmi")} [data-it-selected]`), /Selected: container\.h200\.x1 · 1× H200 141 GB · \$3\.20\/hr · us-denver-1/);
  await docShot(await page.$(T("gmi")), "instance-table-gmi.png");

  // ---- Runpod picked while its table loads, then Brev: the stale load is cancelled and Brev's table shows
  await page.route("**/api/providers/runpod/instance-types", async (r) => (await new Promise((res) => setTimeout(res, 1500)), r.continue().catch(() => {})));
  failed.length = 0;
  await pick("runpod");
  await page.waitForSelector(`${T("runpod")} .it-status[data-loading]`);
  await pick("brev");
  await loaded("brev");
  await page.waitForTimeout(1800);
  assert.ok(failed.some((f) => f.startsWith("GET /api/providers/runpod/instance-types")), `the stale Runpod table load is cancelled: ${failed}`);
  assert.equal(await page.$(T("runpod")), null, "no Runpod table");
  await page.unroute("**/api/providers/runpod/instance-types");

  // ---- Runpod: GPU types × data centres. Weights volume (h3-turbo's default): only EU-RO-1 (the volume) is launchable
  await pick("runpod");
  await loaded("runpod");
  const rp = await names("runpod");
  assert.ok(rp.includes("NVIDIA RTX PRO 6000 Blackwell Server Edition@EU-RO-1") && rp.includes("NVIDIA H100 80GB HBM3@US-CA-2") && rp.includes("NVIDIA H200@US-CA-2") && rp.includes("NVIDIA H200@US-TX-3"), rp.join(", "));
  const WS = `${SL} [data-ctl][data-path="weights_source"]`;
  const weights = (w) => page.click(`${WS} button[role="radio"]:text-is("${w}")`);
  assert.equal(await page.textContent(`${WS} [aria-checked="true"]`), "volume", "h3-turbo: the volume by default");
  assert.equal(await page.$(`${SL} .ff-field[data-path="gpu_types"]`), null, "no separate GPU types list: the table is the pick");
  assert.ok(await page.$(`${SL} .ff-field[data-path="region"]`), "the volume's region");
  assert.equal(await page.$(`${SL} .ff-field[data-path="data_centers"]`), null);
  assert.match(await page.getAttribute(`${T("runpod")} tr[data-row="NVIDIA H200@US-CA-2"]`, "title"), /not launchable: no fv weights volume in US-CA-2/);
  assert.match(await count("runpod"), / · 2 launchable · /, await count("runpod"));
  const rtx = await page.$$eval(`${T("runpod")} tr[data-row="NVIDIA RTX PRO 6000 Blackwell Server Edition@EU-RO-1"] td`, (tds) => tds.map((t) => t.textContent));
  assert.equal(rtx[2], "yes (Low)");
  assert.equal(rtx[6], "Blackwell");
  assert.equal(rtx[10], "datacenter");
  await page.click(`${T("runpod")} th[data-col="usd_per_hr"] button`);
  await page.click(`${T("runpod")} tr[data-row="NVIDIA H100 80GB HBM3@EU-RO-1"]`);
  await page.waitForSelector(`${T("runpod")} tr[data-row="NVIDIA H100 80GB HBM3@EU-RO-1"][aria-selected="true"]`);
  assert.match(await page.textContent(`${T("runpod")} [data-it-selected]`), /Selected: NVIDIA H100 80GB HBM3 · 1× H100 SXM 80 GB · \$3\.29\/hr · EU-RO-1/);

  // ---- the wan preset: its default is r2 (the 500 GB EU volume holds H3 + LTX only). Weights r2: every NVIDIA Secure
  // Cloud row is launchable (recomputed here: no request); the location is any data centre (no region, no volume)
  reqs.length = 0;
  await page.selectOption(`${SL} [data-ctl][data-path="preset"]`, JSON.stringify("wan"));
  await page.waitForSelector(`${WS} [aria-checked="true"]:text-is("r2")`);
  await page.waitForSelector(`${SL} .ff-field[data-path="data_centers"]`);
  await page.waitForSelector(`${T("runpod")} tr[data-row="NVIDIA H200@US-CA-2"].it-can`);
  assert.deepEqual(typeReqs(), [], `the weights switch refetches nothing: ${reqs}`);
  assert.match(await page.getAttribute(`${T("runpod")} tr[data-row="NVIDIA H200@US-CA-2"]`, "title"), /^select NVIDIA H200/);
  assert.match(await count("runpod"), / · 6 launchable · /, await count("runpod"));
  assert.equal(await page.$(`${SL} .ff-field[data-path="region"]`), null, "no region with r2");
  assert.equal(await page.$(`${SL} .ff-field[data-path="volume"]`), null, "no volume with r2");
  // The data centres offered: every one Runpod lists (location, the picked type's stock), fv-control's others, any.
  const dcChips = await page.$$eval(`${SL} [data-ctl][data-path="data_centers"] button`, (bs) => bs.map((b) => b.textContent));
  assert.ok(dcChips.some((t) => /^US-CA-2 United States · stock H100 SXM: High/.test(t)), dcChips.join(" | "));
  assert.ok(dcChips.some((t) => /^US-TX-3 United States · stock H100 SXM: Low/.test(t)), dcChips.join(" | "));
  assert.ok(dcChips.some((t) => /^EU-NL-1 .*not listed now/.test(t)), "fv-control's other data centres");
  assert.match(dcChips[dcChips.length - 1], /^any Runpod places it/);
  // Back to the volume: the rule returns (no refetch); then r2 again.
  await weights("volume");
  await page.waitForSelector(`${T("runpod")} tr[data-row="NVIDIA H200@US-CA-2"].it-cannot`);
  await page.waitForSelector(`${SL} .ff-field[data-path="region"]`);
  await weights("r2");
  await page.waitForSelector(`${T("runpod")} tr[data-row="NVIDIA H200@US-CA-2"].it-can`);
  assert.deepEqual(typeReqs(), [], `no refetch: ${reqs}`);

  // ---- the ordered multi-pick: a click = the primary; "+ fallback" appends (the same type in another data centre
  // adds the data centre, not the type); Shift+Enter appends from the keyboard; the strip reorders and removes
  const strip = () => page.$$eval(`${T("runpod")} [data-it-strip] li`, (ls) => ls.map((l) => l.getAttribute("data-it-pick")));
  // The picked data centres in their order (a chip shows "<n>. " once two or more are picked).
  const dcs = () =>
    page.$$eval(`${SL} [data-ctl][data-path="data_centers"] button[aria-pressed="true"]`, (bs) =>
      bs
        .map((b) => /^(?:(\d+)\. )?(\S+)/.exec(b.textContent))
        .sort((a, b) => Number(a[1] || 0) - Number(b[1] || 0))
        .map((m) => m[2]),
    );
  const until = async (fn, want, what) => {
    for (let i = 0; i < 100; i++) {
      if (JSON.stringify(await fn()) === JSON.stringify(want)) return;
      await page.waitForTimeout(50);
    }
    assert.deepEqual(await fn(), want, what);
  };
  const H100 = "NVIDIA H100 80GB HBM3";
  const H200 = "NVIDIA H200";
  const RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition";
  await page.click(`${T("runpod")} tr[data-row="${H200}@US-CA-2"]`);
  await until(strip, [H200], "the click picks the primary");
  await until(dcs, ["US-CA-2"], "its data centre");
  await page.click(`${T("runpod")} [data-it-add="${H200}@US-TX-3"]`);
  await until(dcs, ["US-CA-2", "US-TX-3"], "the same type elsewhere adds the data centre");
  assert.deepEqual(await strip(), [H200], "not the type twice");
  await page.click(`${T("runpod")} [data-it-add="${H100}@US-CA-2"]`);
  await until(strip, [H200, H100], "a fallback type");
  assert.deepEqual(await dcs(), ["US-CA-2", "US-TX-3"]);
  // Each picked row shows its type's order; H100 in US-TX-3 is tried too (every data centre × every type).
  assert.equal(await page.getAttribute(`${T("runpod")} tr[data-row="${H200}@US-TX-3"]`, "data-it-order"), "1");
  assert.equal(await page.getAttribute(`${T("runpod")} tr[data-row="${H100}@US-TX-3"]`, "data-it-order"), "2");
  assert.equal(await page.getAttribute(`${T("runpod")} tr[data-row="${H100}@EU-RO-1"]`, "data-it-order"), null);
  // Keyboard: Shift+Enter on a focused row appends it.
  await page.focus(`${T("runpod")} tr[data-row="${RTX}@EU-RO-1"]`);
  await page.keyboard.press("Shift+Enter");
  await until(strip, [H200, H100, RTX], "Shift+Enter appends");
  await until(dcs, ["US-CA-2", "US-TX-3", "EU-RO-1"], "with its data centre");
  // The strip: move H200 down (H100 first), remove the RTX PRO 6000; the data centre chip of EU-RO-1 off; "any" last.
  await page.click(`${T("runpod")} [data-it-pick="${H200}"] [data-it-down]`);
  await until(strip, [H100, H200, RTX], "moved down");
  await page.click(`${T("runpod")} [data-it-pick="${RTX}"] [data-it-remove]`);
  await until(strip, [H100, H200], "removed");
  await page.click(`${SL} [data-ctl][data-path="data_centers"] button[aria-pressed="true"]:has-text("EU-RO-1")`);
  await until(dcs, ["US-CA-2", "US-TX-3"], "EU-RO-1 off");
  await page.click(`${SL} [data-ctl][data-path="data_centers"] button:text-matches("^any")`);
  await until(dcs, ["US-CA-2", "US-TX-3", "any"], "any last");
  assert.match(await page.textContent(`${T("runpod")} [data-it-selected]`), /Selected: NVIDIA H100 80GB HBM3 · 1× H100 SXM 80 GB · \$3\.29\/hr · US-CA-2/, "the primary: the first type in the first data centre");
  await docShot(await page.$(T("runpod")), "instance-table-runpod.png");

  // ---- launch: an r2 pod in US data centres, the fallbacks in order (the preflight and the create answered here)
  let posted = null;
  await page.route("**/api/preflight", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true, errors: [], warnings: [] }) }));
  await page.route("**/api/standalone", (r) => (r.request().method() === "POST" ? ((posted = r.request().postDataJSON()), r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ pod: { name: "tables-ui" } }) })) : r.continue()));
  await page.waitForFunction((s) => !document.querySelector(`${s} button[type=submit]`).disabled, SL, { timeout: 15000 }).catch(async (e) => {
    throw new Error(`${e.message}: ${await page.$$eval(`${SL} .cf-err:not([hidden])`, (es) => es.map((x) => `${x.closest("[data-path]")?.getAttribute("data-path")}: ${x.textContent}`).join(" | "))} ${await page.textContent(`${SL} .ff-summary, ${SL} [class*=summary]`).catch(() => "")}`);
  });
  // (After the launch the page opens the simulated pod, which the Worker does not know: its 404 is expected.)
  assert.deepEqual(errors.filter((e) => !/login required|401/.test(e)), [], "no console errors");
  await page.click(`${SL} button[type=submit]`);
  for (let i = 0; i < 50 && !posted; i++) await page.waitForTimeout(100);
  assert.ok(posted, "the launch was posted");
  assert.equal(posted.provider, "runpod");
  assert.equal(posted.weights_source, "r2");
  assert.deepEqual(posted.gpu_types, [H100, H200]);
  assert.deepEqual(posted.data_centers, ["US-CA-2", "US-TX-3", "any"]);
  assert.equal(posted.region, undefined, "no region (the volume's) with r2");
  assert.equal(posted.volume, undefined, "no volume with r2");

  console.log(`ok   UI instance-type tables: load on pick only, sort / search / filters / pages, selection kept and launched, refresh=1 (rate-limited), provider switch cancels stale loads, Runpod weights recompute launchability, ordered multi-pick launched as gpu_types + data_centers${SHOTS ? ` (screenshots in ${img})` : ""}`);
} catch (e) {
  fail(e);
}
await browser.close();
w.stop();
mock.close();
