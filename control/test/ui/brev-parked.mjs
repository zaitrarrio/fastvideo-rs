// UI test (headless Chromium, playwright-core): NVIDIA Brev keep-on-stop on the Pods page (the parked instances'
// card: weights, disk, $/day, state, reuse status, the policy line (owner rule 2026-10-11: never auto-deleted), Unpark,
// the owner's Delete; a VM whose stop came while it was DEPLOYING (stop_pending, owner rule 2026-10-13), with its GPU $/day) and the launch
// form's Brev instance types (every allowed one, stoppable first, live price; owner decision 2026-10-13: the
// non-stoppable ones with a warning: a stop deletes them) with the warm / cold outlook at the instance-type field. Simulated Brev only (test/harness.mjs).
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
  // A VM whose stop came while it was still DEPLOYING (stop_pending, owner rule 2026-10-13): billed at its GPU $/hr
  // until the cron stops it (once it leaves DEPLOYING) or deletes it.
  d1Exec(
    w.dir,
    `INSERT INTO brev_instances (workspace_id, pod_id, name, instance_type, location, stoppable, disk_gb, storage_usd_per_gb_hr, trees, launch_trees, state, created_at, parked_at, updated_at, note, usd_per_hr, stop_tried_at)
     VALUES ('wsui03', 'brev:fv-pod-ui03-1010000000', 'fv-pod-ui03-1010000000', 'a100-test', 'houston-usa-1', 0, 850, NULL, '{}', '{}', 'stop_pending', ${t - 3600_000}, ${t - 3600_000}, ${t}, 'stop_pending: still DEPLOYING when its stop came', 1.8, ${t})`,
  );
  mock.brev.workspaces.push({ id: "wsui03", name: "fv-pod-ui03-1010000000", status: "DEPLOYING", healthStatus: "", instanceType: "a100-test", dns: "", createdAt: new Date(t).toISOString(), startupScript: "", body: {}, disk: {} });

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
  for (const s of ["Parked Brev instances (3): $1.58/day storage, $43.20/day still running", "running (stop pending)", "$43.20 GPU", "still DEPLOYING when its stop came", "stop pending: stopped once it leaves DEPLOYING", "a VM whose stop does not succeed is deleted instead (owner rule 2026-10-13)", "fv-pod-ui01-1010000000", "fv-pod-ui02-1010000000", "fastwan22-ti2v-5b@3e18704", "250 GB", "$0.79", "1.0 d", "2.0 d", "held", "restart failed: no capacity", "never deleted by fv-control (owner rule 2026-10-11)", "alert past 2 parked or 7 days each", "Deleting a stopped VM is yours only", "reusable: the next launch of g5.xlarge-test", "(Unpark: now)"])
    assert.ok(text.includes(s), `parked card: ${s} in ${text}`);
  assert.ok(!/then the oldest is deleted/.test(text), "no auto-delete wording");
  assert.equal(await card.locator("button:text('Unpark')").count(), 1, "Unpark only on the held one");
  assert.equal(await card.locator("button:text('Delete')").count(), 3, "the running one too (the owner's delete)");
  await page.screenshot({ path: `${out}/20-brev-parked.png`, fullPage: true });
  // Unpark the held one: it is parked again.
  await card.locator("tr", { hasText: "fv-pod-ui02" }).locator("button:text('Unpark')").click();
  await page.waitForFunction(() => !document.body.textContent.includes("restart failed: no capacity"));
  assert.equal(await page.locator(".card", { hasText: "Parked Brev instances" }).locator("button:text('Unpark')").count(), 0);
  // Delete one (confirmed): the workspace goes; nothing else is touched.
  await page.locator(".card", { hasText: "Parked Brev instances" }).locator("tr", { hasText: "fv-pod-ui01" }).locator("button:text('Delete')").click();
  await page.waitForSelector("text=Parked Brev instances (2)");
  assert.deepEqual(mock.brev.workspaces.map((x) => x.id), ["wsui02", "wsui03"]);
  assert.ok(mock.brev.calls.some((c) => c.method === "DELETE" && c.path === "/workspaces/wsui01"));
  await page.screenshot({ path: `${out}/21-brev-parked-after.png`, fullPage: true });

  // ---- the launch form: Brev instance types, stoppable first with the live price; the outlook at the field.
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, JSON.stringify("brev"));
  await page.waitForSelector(`${SL} .ff-field[data-path="provider_gpu"]`);
  // The instance types (a combo: the options' values and details): every allowed one, stoppable first, with the live
  // list's price; a type Brev cannot stop says what it keeps billing after a stop (owner decision 2026-10-13).
  await page.waitForFunction((s) => [...document.querySelectorAll(`${s} .ff-field[data-path="provider_gpu"] option`)].some((e) => /\/hr/.test(e.textContent || "")), SL, { timeout: 20000 });
  const opts = await page.$$eval(`${SL} .ff-field[data-path="provider_gpu"] option`, (els) => els.map((e) => `${e.value} ${e.textContent}`));
  assert.match(opts[0] || "", /^g5\.xlarge-test .*\$1\.01\/hr · stoppable: stopped and kept on stop \(reused by the next launch\)/, JSON.stringify(opts));
  assert.match(opts[1] || "", /^a100-test .*\$1\.80\/hr · WARNING: can't be stopped: a stop deletes it \(its disk and weights go too\)/, JSON.stringify(opts));
  // The warm / cold outlook shows at the instance-type field (the validator's warning) once Hub weights are picked.
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "uiwarm");
  await page.fill(`${SL} [data-ctl][data-path="provider_gpu"]`, "g5.xlarge-test");
  const ws = `${SL} [data-ctl][data-path="weights_source"]`;
  if ((await page.$eval(ws, (e) => e.tagName)) === "SELECT") await page.selectOption(ws, JSON.stringify("hub"));
  else await page.click(`${ws} [role=radio]:text-is("hub")`);
  await page.waitForSelector(`${SL} .ff-field[data-path="weights_download_approved"]`);
  await page.check(`${SL} [data-ctl][data-path="weights_download_approved"]`);
  await page.waitForSelector(`${SL} .ff-field[data-path="provider_gpu"] .ff-warn:not([hidden]):text-matches("^(cold|warm|reuse): ")`, { timeout: 20000 });
  const outlook = await page.textContent(`${SL} .ff-field[data-path="provider_gpu"] .ff-warn`);
  // The parked fv-pod-ui02 (the same type; 25 GB on its 250 GB disk) is reused before a new VM: h3-base and the FastH3 LoRA download onto it.
  assert.match(outlook, /^reuse: restarts parked fv-pod-ui02-1010000000 \(none of the trees at the pinned revisions on it; downloads FastH3-4-step-Preview-v1-LoRA, h3-base ~151 GB onto it\): boot ~\d+ min$/, outlook);
  await page.screenshot({ path: `${out}/22-brev-launch-types.png`, fullPage: true });

  assert.deepEqual(errors.filter((e) => !/login required|401/.test(e)), [], "no console errors");
  console.log(`ok   UI Brev keep-on-stop: parked card (weights, disk, $/day, held, Unpark, Delete, a stop-pending VM), launch form instance types (non-stoppable: a stop deletes it) (screenshots in ${out})`);
} catch (e) {
  fail(e);
}
await browser.close();
w.stop();
mock.close();
