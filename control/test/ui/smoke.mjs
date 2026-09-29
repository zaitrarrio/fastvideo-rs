// UI smoke test in headless Chromium (playwright-core): the dashboard over
// the Worker under `wrangler dev` and the mocked APIs (test/harness.mjs).
//   npx playwright-core install chromium-headless-shell   (once)
//   node test/ui/smoke.mjs                                  (npm run test:ui)
// Screenshots land in test-results/.
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
const w = await startWorker(mock, { ...SECRETS, OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET) });
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
  // Seed: an API token, a running tiny cluster, two collector passes.
  const login = await fetch(`${B}/api/auth/login`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ passphrase: PASSPHRASE }) });
  const cookie = login.headers.get("set-cookie").split(";")[0];
  const { csrf } = await login.json();
  const tok = (await (await fetch(`${B}/api/tokens`, { method: "POST", headers: { cookie, "x-csrf-token": csrf, "content-type": "application/json" }, body: JSON.stringify({ name: "seed" }) })).json()).token;
  const api = (p, body) => fetch(B + p, { method: body ? "POST" : "GET", headers: { authorization: `Bearer ${tok}`, "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined }).then((r) => r.json());
  const c = await api("/api/clusters", { spec: { name: "tiny", template: "tiny-cpu", cap_s: 3600 } });
  await api(`/api/clusters/${c.cluster.id}/start`, {});
  for (let i = 0; i < 60; i++) {
    const ops = await api(`/api/clusters/${c.cluster.id}/ops`);
    if (ops.operations[0]?.status !== "running") break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  await api("/api/collect", {});
  await api("/api/collect", {});

  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto(B + "/");
  await page.waitForSelector("#pass");
  await page.screenshot({ path: `${out}/01-login.png` });
  await page.fill("#pass", "wrong");
  await page.click("button[type=submit]");
  await page.waitForSelector("text=wrong passphrase");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");
  await page.waitForSelector(".tile");
  const tiles = await page.$$eval(".tile .k", (els) => els.map((e) => e.textContent));
  assert.deepEqual(tiles.slice(0, 3), ["Balance", "Burn", "Time to floor"]);
  assert.equal(await page.textContent(".tile .v"), "$50.00");
  assert.ok((await page.$$(".chart svg path")).length >= 3, "charts drew lines");
  await page.screenshot({ path: `${out}/02-dashboard.png`, fullPage: true });
  // Hover a chart: the tooltip appears.
  const svg = await page.$(".chart svg");
  const box = await svg.boundingBox();
  await page.mouse.move(box.x + box.width * 0.9, box.y + box.height / 2);
  assert.equal(await page.isVisible("#tip"), true, "tooltip on hover");

  for (const [hash, heading] of [["#/clusters", "Clusters"], ["#/pods", "Pods"], ["#/env", "Environment"], ["#/logs", "Logs"], ["#/costs", "Costs"], ["#/releases", "Releases"], ["#/settings", "Settings"]]) {
    await page.goto(`${B}/${hash}`);
    await page.waitForSelector(`h1:text('${heading}')`);
    await page.screenshot({ path: `${out}/03-${heading.toLowerCase()}.png`, fullPage: true });
  }
  await page.goto(`${B}/#/cluster/${c.cluster.id}`);
  await page.waitForSelector("h1:has-text('tiny')");
  assert.ok(await page.isVisible("text=Stop (delete pods)"));
  await page.screenshot({ path: `${out}/04-cluster.png`, fullPage: true });
  // ---- the smart editor: edit → invalid → error shown → fix → diff → plan → save → history → restore.
  await page.goto(`${B}/#/cluster/${c.cluster.id}`);
  await page.waitForSelector(".fv-panel[data-kind=cluster-spec] .fv-form");
  const panel = ".fv-panel[data-kind=cluster-spec]";
  assert.equal(await page.textContent(`${panel} .badge`), "v0");
  // Form: the pools table and a toggle are there.
  assert.ok(await page.isVisible(`${panel} .fv-table table`), "pools table");
  await page.click(`${panel} .tabs button:text('JSON')`);
  await page.waitForSelector(`${panel} .cm-content`);
  const setDoc = (fn) => page.evaluate(([sel, src]) => {
    const view = window.FVEditor.viewOf(document.querySelector(sel + " .cm-editor"));
    const f = new Function("t", src);
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: f(view.state.doc.toString()) } });
  }, [panel, fn]);
  await setDoc(`return t.replace(/"cap_s": \\d+/, '"cap_s": 1')`);
  await page.waitForSelector(`${panel} .cm-lintRange-error`);
  await page.waitForSelector(`${panel} .fv-issues li:has-text("cap_s")`);
  await page.screenshot({ path: `${out}/08-editor-invalid.png`, fullPage: false });
  // Hover docs from the schema.
  const capKey = await page.$(`${panel} .cm-content >> text=cap_s`);
  await capKey.hover();
  await page.waitForSelector(".cm-tooltip-hover:has-text('Backstop')");
  // Completion of keys: an optional key that is not in the document.
  await page.evaluate((sel) => { const v = window.FVEditor.viewOf(document.querySelector(sel + " .cm-editor")); const p = v.state.doc.toString().indexOf("{") + 1; v.dispatch({ selection: { anchor: p }, changes: { from: p, insert: "\n  \"log_l" } }); v.focus(); }, panel);
  await page.keyboard.press("Control+Space");
  await page.waitForSelector(".cm-tooltip-autocomplete li:has-text('log_level')");
  await page.keyboard.press("Escape");
  // Fix: a valid cap_s, and a pool count change in the form.
  await setDoc(`return t.replace(/\\n  "log_l/, "").replace(/"cap_s": 1\\b/, '"cap_s": 3000')`);
  await page.waitForSelector(`${panel} .fv-status.ok`);
  await page.click(`${panel} .tabs button:text('Form')`);
  const count = await page.$(`${panel} .fv-table tbody tr:first-child input[type=number]`);
  await count.fill("2");
  await page.click(`${panel} button:text('Review, plan & save')`);
  await page.waitForSelector(`${panel} .fv-diff .add`);
  await page.waitForSelector(`${panel} .fv-plan li`);
  const planText = await page.textContent(`${panel} .fv-plan`);
  assert.match(planText, /more worker|Scale fake to 2/, planText);
  assert.match(planText, /within the floor|over the floor/);
  await page.screenshot({ path: `${out}/09-editor-review.png`, fullPage: false });
  await page.click(`${panel} button:has-text('Save (v0')`);
  await page.waitForFunction((sel) => document.querySelector(sel + " .badge")?.textContent === "v1", panel);
  // A stale edit elsewhere is refused.
  const stale = await fetch(`${B}/api/docs/cluster-spec/${c.cluster.id}`, { method: "PUT", headers: { authorization: `Bearer ${tok}`, "content-type": "application/json" }, body: JSON.stringify({ doc: {}, version: 0 }) });
  assert.ok(stale.status === 409 || stale.status === 400);
  // History and restore.
  await page.click(`${panel} button:text('History')`);
  await page.waitForSelector(`${panel} .fv-history details summary`);
  await page.click(`${panel} .fv-history details summary`);
  await page.click(`${panel} button:text('Restore the version before it')`);
  await page.waitForFunction((sel) => document.querySelector(sel + " .badge")?.textContent === "v2", panel);
  const spec = await (await fetch(`${B}/api/docs/cluster-spec/${c.cluster.id}`, { headers: { authorization: `Bearer ${tok}` } })).json();
  assert.equal(spec.doc.cap_s, 3600, "restored");
  assert.equal(spec.doc.pools[0].count, 1);

  // ---- env editor: a secret through the write-only field never reaches the page.
  await page.goto(`${B}/#/env?cluster=${c.cluster.id}`);
  const envp = ".fv-panel[data-kind=env]";
  await page.waitForSelector(`${envp} .fv-table`);
  await page.fill(`${envp} input[aria-label="new key"]`, "HF_TOKEN");
  await page.click(`${envp} button:text('+ add')`);
  await page.check(`${envp} tr[data-key=HF_TOKEN] input[type=checkbox]`);
  await page.fill(`${envp} tr[data-key=HF_TOKEN] input[type=password]`, "hf_editor_secret");
  await page.click(`${envp} .tabs button:text('JSON')`);
  assert.ok(!(await page.textContent(`${envp} .cm-content`)).includes("hf_editor_secret"), "JSON tab masks the pending secret");
  await page.click(`${envp} button:text('Review & save')`);
  await page.waitForSelector(`${envp} .fv-diff .add`);
  await page.click(`${envp} button:has-text('Save (v')`);
  await page.waitForFunction((sel) => document.querySelector(sel + " .badge")?.textContent === "v1", envp);
  assert.ok(!(await page.content()).includes("hf_editor_secret"), "the secret is never rendered");
  await page.waitForSelector("details summary:has-text('needs restart'), p:has-text('No pods running')");
  await page.screenshot({ path: `${out}/10-env-editor.png`, fullPage: true });

  // ---- read-only JSON tree: search and copy.
  const pods = await (await fetch(`${B}/api/pods`, { headers: { authorization: `Bearer ${tok}` } })).json();
  await page.goto(`${B}/#/pod/${pods.pods[0].pod_id}`);
  await page.waitForSelector(".fv-tree input[type=search]");
  await page.fill(".fv-tree input[type=search]", "cost_per_hr");
  await page.waitForSelector(".fv-thead.hit");
  await page.screenshot({ path: `${out}/11-tree.png`, fullPage: false });

  // Dark mode.
  await page.goto(`${B}/#/`);
  await page.waitForSelector(".tile");
  await page.click("#themeBtn");
  await page.waitForSelector(".tile");
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  await page.screenshot({ path: `${out}/06-dark.png`, fullPage: true });
  await page.click("#themeBtn");
  const bg2 = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  assert.notEqual(bg, bg2, "theme toggles");

  // Phone width: no horizontal page scroll.
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, colorScheme: "dark" });
  const pp = await phone.newPage();
  pp.on("pageerror", (e) => errors.push(e.message));
  await pp.goto(B + "/");
  await pp.fill("#pass", PASSPHRASE);
  await pp.click("button[type=submit]");
  await pp.waitForSelector(".tile");
  for (const hash of ["#/", "#/clusters", "#/pods", "#/costs", "#/settings"]) {
    await pp.goto(`${B}/${hash}`);
    await pp.waitForSelector("h1");
    await pp.waitForTimeout(300);
    const over = await pp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    assert.ok(over <= 1, `no horizontal scroll on ${hash} (${over}px)`);
  }
  await pp.goto(`${B}/#/`);
  await pp.waitForSelector(".tile");
  await pp.screenshot({ path: `${out}/07-phone-dark.png`, fullPage: true });

  // Logout.
  await page.click("#logoutBtn");
  await page.waitForSelector("#pass");
  assert.deepEqual(errors.filter((e) => !/login required|401/.test(e)), [], "no console errors");
  console.log(`ok   UI smoke: login, dashboard, charts + tooltip, every page, env secret masked, dark mode, phone width (screenshots in ${out})`);
} catch (e) {
  fail(e);
}
await browser.close();
w.stop();
mock.close();
