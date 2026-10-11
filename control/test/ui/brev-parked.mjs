// UI test (headless Chromium, playwright-core): NVIDIA Brev keep-on-stop on the Pods page (the parked instances'
// card: weights, disk, $/day, state, Unpark, Delete) and the launch form's Brev instance types (stoppable first,
// live price) with the warm / cold outlook at the instance-type field. Simulated Brev only (test/harness.mjs).
//   CHROMIUM=/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell node test/ui/brev-parked.mjs
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { d1Exec, hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.brevToken = SECRETS.BREV_API_TOKEN;
mock.brevOrg = "org-ui-1";
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  BREV_ORG_ID: mock.brevOrg,
  BREV_INSTANCE_TYPES: "a100-test,g5.xlarge-test",
  BREV_BUDGET_USD: "50",
  FV_HUB_DOWNLOADS_APPROVED: "1",
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
  // Seed: two parked instances fv-control made (one held after a failed restart) and their simulated workspaces.
  const t = Date.now();
  const rev = "3e187042a324f6f5fb68fd22110a78725253de8f";
  for (const [id, state, days, note] of [["wsui01", "parked", 1, null], ["wsui02", "held", 2, "restart failed: no capacity in the same provider/region"]]) {
    const name = `fv-pod-ui${id.slice(-2)}-1010000000`;
    d1Exec(
      w.dir,
      `INSERT INTO brev_instances (workspace_id, pod_id, name, instance_type, location, stoppable, disk_gb, storage_usd_per_gb_hr, trees, launch_trees, state, created_at, parked_at, updated_at, note)
       VALUES ('${id}', 'brev:${name}', '${name}', 'g5.xlarge-test', 'us-east-1', 1, 250, 0.000132, '{"fastwan22-ti2v-5b":"${rev}"}', '{}', '${state}', ${t - days * 86400_000}, ${t - days * 86400_000}, ${t}, ${note ? `'${note}'` : "NULL"})`,
    );
    mock.brev.workspaces.push({ id, name, status: "STOPPED", healthStatus: "", instanceType: "g5.xlarge-test", dns: "", createdAt: new Date(t).toISOString(), startupScript: "", body: {}, disk: {} });
  }

  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  await page.goto(B + "/");
  await page.waitForSelector("#pass");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");

  // ---- the Pods page: the parked card
  await page.goto(`${B}/#/pods`);
  const card = page.locator(".card", { hasText: "Parked Brev instances" });
  await card.waitFor();
  const text = await card.textContent();
  for (const s of ["Parked Brev instances (2): $1.58/day storage", "fv-pod-ui01-1010000000", "fv-pod-ui02-1010000000", "fastwan22-ti2v-5b@3e18704", "250 GB", "$0.79", "1.0 d", "2.0 d", "held", "restart failed: no capacity", "At most 2 parked, 7 days each"])
    assert.ok(text.includes(s), `parked card: ${s} in ${text}`);
  assert.equal(await card.locator("button:text('Unpark')").count(), 1, "Unpark only on the held one");
  assert.equal(await card.locator("button:text('Delete')").count(), 2);
  await page.screenshot({ path: `${out}/20-brev-parked.png`, fullPage: true });
  // Unpark the held one: it is parked again.
  await card.locator("tr", { hasText: "fv-pod-ui02" }).locator("button:text('Unpark')").click();
  await page.waitForFunction(() => !document.body.textContent.includes("restart failed: no capacity"));
  assert.equal(await page.locator(".card", { hasText: "Parked Brev instances" }).locator("button:text('Unpark')").count(), 0);
  // Delete one (confirmed): the workspace goes; nothing else is touched.
  await page.locator(".card", { hasText: "Parked Brev instances" }).locator("tr", { hasText: "fv-pod-ui01" }).locator("button:text('Delete')").click();
  await page.waitForSelector("text=Parked Brev instances (1)");
  assert.deepEqual(mock.brev.workspaces.map((x) => x.id), ["wsui02"]);
  assert.ok(mock.brev.calls.some((c) => c.method === "DELETE" && c.path === "/workspaces/wsui01"));
  await page.screenshot({ path: `${out}/21-brev-parked-after.png`, fullPage: true });

  // ---- the launch form: Brev instance types, stoppable first with the live price; the outlook at the field.
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, JSON.stringify("brev"));
  await page.waitForSelector(`${SL} .ff-field[data-path="provider_gpu"]`);
  // The instance types (a combo: the options' values and details), stoppable first with the live list's price.
  const opts = await page.$$eval(`${SL} .ff-field[data-path="provider_gpu"] option`, (els) => els.map((e) => `${e.value} ${e.textContent}`));
  assert.match(opts[0] || "", /^g5\.xlarge-test .*\$1\.01\/hr · stoppable: weights kept on stop \(warm restarts\)/, JSON.stringify(opts));
  assert.match(opts[1] || "", /^a100-test .*not stoppable: weights download at every launch/, JSON.stringify(opts));
  // The warm / cold outlook shows at the instance-type field (the validator's warning) once Hub weights are picked.
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "uiwarm");
  await page.fill(`${SL} [data-ctl][data-path="provider_gpu"]`, "g5.xlarge-test");
  const ws = `${SL} [data-ctl][data-path="weights_source"]`;
  if ((await page.$eval(ws, (e) => e.tagName)) === "SELECT") await page.selectOption(ws, JSON.stringify("hub"));
  else await page.click(`${ws} [role=radio]:text-is("hub")`);
  await page.waitForSelector(`${SL} .ff-field[data-path="weights_download_approved"]`);
  await page.check(`${SL} [data-ctl][data-path="weights_download_approved"]`);
  await page.waitForSelector(`${SL} .ff-field[data-path="provider_gpu"] .ff-warn:not([hidden]):text-matches("^(cold|warm): ")`, { timeout: 20000 });
  const outlook = await page.textContent(`${SL} .ff-field[data-path="provider_gpu"] .ff-warn`);
  assert.match(outlook, /^cold: downloads h3-base \(~144 GB from the Hub\): boot ~\d+ min; kept on stop \(the next launch is warm\)$/, outlook);
  await page.screenshot({ path: `${out}/22-brev-launch-types.png`, fullPage: true });

  assert.deepEqual(errors.filter((e) => !/login required|401/.test(e)), [], "no console errors");
  console.log(`ok   UI Brev keep-on-stop: parked card (weights, disk, $/day, held, Unpark, Delete), launch form instance types (screenshots in ${out})`);
} catch (e) {
  fail(e);
}
await browser.close();
w.stop();
mock.close();
