// UI test (headless Chromium, playwright-core): the NVIDIA Brev monthly budget as the owner enters it (owner report
// 2026-10-18: "when I try to launch a Brev instance it is not recognizing the budget I enter it is always sets the
// budget to $10"). The Worker var BREV_BUDGET_USD is $10 here, as on staging. The launch form shows the budget field
// under Provider with $10 from the Worker var; a 12-hour launch is refused at the provider before Launch ("over the
// monthly budget $10"); the owner enters $75 and saves: the request carries 75, the field, the provider line and the
// check use 75, Launch goes through (the Brev VM is created), the value is stored (policies) and is still 75 after a
// reload. The one-click deploy page shows the same budget. Simulated Brev only (test/harness.mjs).
//   CHROMIUM=/opt/pw-browsers/chromium node test/ui/brev-budget.mjs
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.brevToken = SECRETS.BREV_API_TOKEN;
mock.brevOrg = "org-ui-budget";
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  BREV_ORG_ID: mock.brevOrg,
  BREV_INSTANCE_TYPES: "g5.xlarge-test",
  // The staging value the owner kept getting.
  BREV_BUDGET_USD: "10",
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
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 1000 } })).newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  /** The budget saves the page sent (their bodies). */
  const saves = [];
  page.on("request", (r) => r.method() === "PUT" && r.url().includes("/api/v1/providers/") && r.url().endsWith("/budget") && saves.push({ url: r.url().slice(B.length), body: r.postDataJSON() }));
  await page.goto(B + "/");
  await page.waitForSelector("#pass");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");
  /** The API as the page's session (CSRF token from /auth/me on writes). */
  const api = (path, init = {}) =>
    page.evaluate(async ([p, i]) => {
      const csrf = i.method && i.method !== "GET" ? (await (await fetch("/api/v1/auth/me", { credentials: "same-origin" })).json()).csrf : undefined;
      const r = await fetch(p, { ...i, credentials: "same-origin", headers: { "content-type": "application/json", ...(csrf ? { "x-csrf-token": csrf } : {}) } });
      return r.json();
    }, [path, init]);

  // ---- the launch form: Brev picked, the budget field shows the Worker var's $10
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} [data-ctl][data-path="name"]`);
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "budget-ui");
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, JSON.stringify("brev"));
  const BF = `${SL} [data-budget-field="brev"]`;
  await page.waitForSelector(BF, { timeout: 20000 });
  assert.equal(await page.inputValue(`${BF} [data-budget-input]`), "10");
  assert.match(await page.textContent(BF), /Monthly budget \(NVIDIA Brev, \$\)/);
  assert.match(await page.textContent(BF), /not per pod: the per-pod cap is Max GPU \$\/hr/);
  assert.match(await page.textContent(BF), /Now \$10, from the Worker var BREV_BUDGET_USD until you save one here/);
  assert.ok(await page.$eval(`${BF} [data-budget-save]`, (b) => b.disabled), "Save waits for a change");
  await page.waitForFunction((s) => /of the \$10 monthly budget \(the Worker var BREV_BUDGET_USD\)/.test(document.querySelector(`${s} [data-provider-status]`)?.textContent || ""), SL, { timeout: 20000 });

  // A fake-engine VM on g5.xlarge-test ($1.006/hr) for 12 h: $12.07, over $10: refused at the provider, Launch off.
  // Custom (the fake engine) after Brev is picked: no Runpod-only compute / volume is set (it was: "compute GPU" refused).
  await page.selectOption(`${SL} [data-ctl][data-path="preset"]`, "");
  await page.waitForSelector(`${SL} select[data-ctl][data-path="variant"]`);
  await page.fill(`${SL} [data-ctl][data-path="provider_gpu"]`, "g5.xlarge-test");
  await page.fill(`${SL} [data-ctl][data-path="deadline_min"]`, "720");
  const provErr = `${SL} .ff-field[data-path="provider"] .cf-err:not([hidden])`;
  await page.waitForSelector(`${provErr}:text-matches("over the monthly budget \\\\$10")`, { timeout: 20000 });
  assert.match(await page.textContent(provErr), /NVIDIA Brev: \$0\.00 spent this month \+ \$1\.01\/hr for 12\.00 h = \$12\.07, over the monthly budget \$10 \(raise it: the launch form's "Monthly budget" field/);
  assert.ok(await page.$eval(`${SL} button[type=submit]`, (b) => b.disabled), "Launch is off while over the budget");
  await page.screenshot({ path: `${out}/50-brev-budget-over.png`, fullPage: true });

  // ---- the owner enters $75 and saves: the request carries 75; the field, the line and the check use it
  await page.fill(`${BF} [data-budget-input]`, "75");
  await page.click(`${BF} [data-budget-save]`);
  await page.waitForSelector(`${BF} [data-budget-msg]:text("saved: $75 per month")`);
  assert.deepEqual(saves, [{ url: "/api/v1/providers/brev/budget", body: { budget_usd: 75 } }]);
  assert.equal(await page.inputValue(`${BF} [data-budget-input]`), "75");
  assert.match(await page.textContent(BF), /Now \$75, set here \(Settings → Policies brev_budget_usd\)/);
  assert.match(await page.textContent(`${SL} [data-provider-status]`), /this month \$0\.00 of the \$75 monthly budget/);
  assert.doesNotMatch(await page.textContent(`${SL} [data-provider-status]`), /\$10/);
  await page.waitForFunction((s) => !document.querySelector(`${s} .ff-field[data-path="provider"] .cf-err:not([hidden])`), SL, { timeout: 20000 });
  await page.waitForFunction((s) => !document.querySelector(`${s} button[type=submit]`).disabled, SL, { timeout: 20000 });
  // A redraw (another field changed) keeps the saved value in the field.
  await page.fill(`${SL} [data-ctl][data-path="deadline_min"]`, "700");
  await page.fill(`${SL} [data-ctl][data-path="deadline_min"]`, "720");
  assert.equal(await page.inputValue(`${BF} [data-budget-input]`), "75");
  // Stored where every check reads it.
  const pol = await api("/api/v1/policies");
  assert.equal(pol.policies.brev_budget_usd, 75);
  const pv = await api("/api/v1/providers?provider=brev");
  assert.equal(pv.providers[0].budget_usd, 75);
  assert.equal(pv.providers[0].budget_source, "policy");
  await page.screenshot({ path: `${out}/51-brev-budget-saved.png`, fullPage: true });

  // ---- Launch: the start's budget check passes with $75 (it failed with "over the budget $10" before the fix)
  await page.waitForFunction((s) => !document.querySelector(`${s} button[type=submit]`).disabled, SL, { timeout: 20000 });
  await page.click(`${SL} button[type=submit]:text('Launch')`);
  await page.waitForSelector("h1:text('budget-ui')", { timeout: 60000 });
  let sp;
  for (let i = 0; i < 120; i++) {
    sp = await api("/api/v1/standalone/budget-ui");
    if (mock.brev.workspaces.some((x) => x.name.startsWith("fv-pod-budget-ui-")) || sp.pod.last_op?.status === "failed") break;
    await page.waitForTimeout(500);
  }
  assert.ok(!/budget/.test(sp.pod.last_op?.error || ""), `no budget refusal: ${sp.pod.last_op?.error}`);
  assert.ok(mock.brev.workspaces.some((x) => x.name.startsWith("fv-pod-budget-ui-")), `the Brev VM was created: ${JSON.stringify(sp.pod.last_op || sp.pod.op)}`);
  assert.equal(sp.pod.definition.provider, "brev");
  // The launch's log on the pod's page: the Brev budget line, no Runpod balance / floor line (a Brev-only launch
  // printed "balance $0.00, account $0.00/hr, …; projected $0.00 at the deadline (floor $8)" before).
  await page.waitForFunction(() => /NVIDIA Brev: \$[\d.]+ spent this month \+ ~\$[\d.]+\/hr \(this launch \$1\.01\/hr\) for 12\.00 h → projected \$[\d.]+ at the deadline; budget \$75 \(within\)/.test(document.body.innerText), null, { timeout: 30000 });
  assert.doesNotMatch(await page.textContent("body"), /\(floor \$\d|balance \$0\.00, account \$0\.00\/hr/);
  await page.screenshot({ path: `${out}/52-brev-budget-launched.png`, fullPage: true });
  assert.ok(!(await api("/api/v1/standalone/budget-ui", { method: "DELETE" })).error, "deleted");

  // ---- a reload: the stored $75 is what the form shows (not the Worker var's $10)
  await page.goto(`${B}/`);
  await page.reload();
  await page.goto(`${B}/#/standalone`);
  await page.waitForSelector(`${SL} [data-ctl][data-path="name"]`);
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, JSON.stringify("brev"));
  await page.waitForSelector(BF, { timeout: 20000 });
  assert.equal(await page.inputValue(`${BF} [data-budget-input]`), "75");
  await page.waitForFunction((s) => /of the \$75 monthly budget/.test(document.querySelector(`${s} [data-provider-status]`)?.textContent || ""), SL, { timeout: 20000 });

  // ---- one-click deploy: a Brev family shows the same budget; a save there is read back too
  await page.goto(`${B}/#/deploy?family=wan`);
  await page.waitForSelector("[data-deploy-family-card]");
  const fam = await page.$eval("[data-deploy-family-card]", (e) => e.getAttribute("data-deploy-family-card"));
  await page.selectOption(`[data-deploy-provider="${fam}"]`, "brev");
  const DF = `[data-deploy-family-card="${fam}"] [data-budget-field="brev"]`;
  await page.waitForSelector(DF, { timeout: 20000 });
  assert.equal(await page.inputValue(`${DF} [data-budget-input]`), "75");
  await page.fill(`${DF} [data-budget-input]`, "60");
  await page.click(`${DF} [data-budget-save]`);
  for (let i = 0; i < 40 && (await api("/api/v1/policies")).policies.brev_budget_usd !== 60; i++) await page.waitForTimeout(250);
  assert.equal((await api("/api/v1/policies")).policies.brev_budget_usd, 60);
  assert.deepEqual(saves.at(-1), { url: "/api/v1/providers/brev/budget", body: { budget_usd: 60 } });
  await page.waitForSelector(`[data-deploy-family-card="${fam}"] [data-budget-field="brev"] [data-budget-msg]:text("saved: $60 per month")`);
  assert.equal(await page.inputValue(`[data-deploy-family-card="${fam}"] [data-budget-field="brev"] [data-budget-input]`), "60");
  await page.screenshot({ path: `${out}/53-deploy-brev-budget.png`, fullPage: true });

  assert.deepEqual(errors.filter((e) => !/login required|401/.test(e)), [], "no console errors");
  console.log(`ok   UI Brev monthly budget: the entered value is saved, read back, shown and used by the launch (Worker var $10 → $75), kept after a reload, editable from one-click deploy (screenshots in ${out})`);
} catch (e) {
  fail(e);
}
await browser.close();
w.stop();
mock.close();
