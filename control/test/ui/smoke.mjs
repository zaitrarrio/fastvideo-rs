// UI smoke test in headless Chromium (playwright-core): the dashboard over
// the Worker under `wrangler dev` and the mocked APIs (test/harness.mjs).
//   npx playwright-core install chromium-headless-shell   (once)
//   node test/ui/smoke.mjs                                  (npm run test:ui)
// Screenshots land in test-results/.
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { d1Exec, hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  // The edge stand-in (test/harness.mjs `/edge/*`): clusters run behind the edge.
  EDGE_URL: `http://127.0.0.1:${mock.port}/edge`,
  EDGE_INTERNAL_TOKEN: mock.edgeInternal,
  EDGE_ADMIN_TOKEN: mock.edgeAdmin,
  EDGE_D1_DATABASE_ID: "d1-edge-staging",
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

  for (const [hash, heading] of [["#/clusters", "Clusters"], ["#/pods", "Pods"], ["#/env", "Environment"], ["#/logs", "Logs"], ["#/costs", "Costs"], ["#/releases", "Releases"], ["#/settings", "Settings"], ["#/standalone", "Standalone pods"]]) {
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
  await page.click("details.rawspec summary");
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
  await page.click(`${envp} button:text('+ variable')`);
  await page.fill(`${envp} input[aria-label="variable name"]`, "HF_TOKEN");
  await page.press(`${envp} input[aria-label="variable name"]`, "Tab");
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

  // ---- the operations controls (wip/ui-dashboard): alert resolve, build pod card, templates and presets,
  // roll / restart pickers, add pool, key list and revoke, pool env, policy text.
  const accept = (d) => d.accept();
  page.on("dialog", accept);
  mock.buildHealth = { extbuild0001: { ok: true, ready: true, phase: "ready", boot: 1, uptime_s: 3600, idle_s: 0, idle_stop_in_s: null, max_stop_in_s: 25200, idle_stop_s: 1200, max_s: 28800, max_grace_s: 1800, jobs_active: 1, self_stop: { attempts: 2, next_at: null, reason: "idle 20 min", at: Math.floor(Date.now() / 1000) - 600, ok: null, error: "REST stop: HTTP 403 Forbidden" }, jobs: [{ id: "1002-abc", agent: "wt-ui-dash", state: "running", seconds: 42 }] } };
  d1Exec(w.dir, `INSERT INTO alerts (key, kind, severity, target, message, opened_at, last_seen_at) VALUES ('ui:test', 'pod_down', 'warn', 'x', 'ui test alert', ${Date.now()}, ${Date.now()})`);
  await page.goto(`${B}/#/`);
  await page.waitForSelector(".buildpod[data-pod=extbuild0001]");
  const bpText = await page.textContent(".buildpod[data-pod=extbuild0001]");
  for (const s of ["cap stop in", "7.0 h", "FAILED", "REST stop: HTTP 403", "wt-ui-dash", "paused: jobs running", "controller backstop", "cap in 8.0 h"]) assert.ok(bpText.includes(s), `build pod card: ${s} in ${bpText}`);
  await page.click(".alert:has-text('ui test alert') button:text('Resolve')");
  await page.waitForSelector(".alert:has-text('ui test alert')", { state: "detached" });
  assert.ok((await (await fetch(`${B}/api/alerts`, { headers: { authorization: `Bearer ${tok}` } })).json()).alerts.every((a) => a.key !== "ui:test"), "resolved");
  await page.screenshot({ path: `${out}/12-dashboard-buildpod.png`, fullPage: true });
  // Managed build pods: Up creates one (policy enabled by API), the card shows it with Stop / Delete.
  await fetch(`${B}/api/build-pods/policy`, { method: "PUT", headers: { authorization: `Bearer ${tok}`, "content-type": "application/json" }, body: JSON.stringify({ policy: { enabled: true } }) });
  await page.reload();
  await page.click("button:text('Up (reuse / start / create)')");
  await page.waitForSelector(".buildpod[data-bp]", { timeout: 20000 });
  const mText = await page.textContent(".buildpod[data-bp]");
  for (const s of ["EU-RO-1", "cpu3c 32 vCPU", "main@", "idle stop 20 min", "Stop", "Delete"]) assert.ok(mText.includes(s), `managed build pod card: ${s} in ${mText}`);
  await page.screenshot({ path: `${out}/12b-dashboard-build-pods.png`, fullPage: true });
  // Clusters: every template, and a pool preset added to the spec being defined.
  await page.goto(`${B}/#/clusters`);
  await page.waitForSelector("select[aria-label=Template] option[value=ltx]", { state: "attached" });
  const tpls = await page.$$eval("select[aria-label=Template] option", (els) => els.map((e) => e.value));
  assert.deepEqual(tpls.sort(), ["h3", "longlive", "ltx", "standard", "tiny-cpu", "wan"]);
  // New cluster from a template on the configuration page; a pool preset added (its licence confirmed).
  await page.selectOption("select[aria-label=Template]", "ltx");
  await page.click("button:text('New cluster')");
  await page.waitForSelector(".cf-pool");
  assert.deepEqual(await page.$$eval(".cf-pool-head b", (els) => els.map((e) => e.textContent)), ["ltx", "ltx-pro", "ltx-a2v", "ltx-ref2v"]);
  // (the page's dialog handler accepts the licence confirm)
  await page.selectOption("select[aria-label='Pool preset']", "longlive");
  await page.waitForSelector(".cf-pool-head b:text('longlive')");
  await page.click(".cf-main .tabs button:text('JSON')");
  await page.waitForSelector(".cf-json .cm-editor");
  const edDoc = () => page.evaluate(() => window.FVEditor.viewOf(document.querySelector(".cf-json .cm-editor")).state.doc.toString());
  assert.equal(JSON.parse(await edDoc()).pools.map((p) => p.id).join(","), "ltx,ltx-pro,ltx-a2v,ltx-ref2v,longlive");
  await page.screenshot({ path: `${out}/13-clusters-presets.png`, fullPage: true });
  // Cluster: roll picker (cancelled), add a pool, keys, restart picker.
  await fetch(`${B}/api/clusters/${c.cluster.id}/mint-key`, { method: "POST", headers: { authorization: `Bearer ${tok}`, "content-type": "application/json" }, body: JSON.stringify({ name: "ui-key" }) });
  await page.goto(`${B}/#/cluster/${c.cluster.id}`);
  await page.waitForSelector("button:text('Roll to…'):not([disabled])");
  await page.click("button:text('Roll to…')");
  await page.waitForSelector("dialog.ff-dialog .fv-chip:has-text('fake')");
  assert.equal(await page.getAttribute("dialog.ff-dialog .fv-chip:has-text('fake')", "aria-pressed"), "true");
  assert.ok(await page.isVisible("dialog.ff-dialog [data-ctl][data-path=target]"));
  await page.screenshot({ path: `${out}/14-roll-picker.png` });
  await page.click("dialog.ff-dialog button:text('Cancel')");
  await page.waitForSelector("dialog.ff-dialog", { state: "detached" });
  await page.click("button:text('Add pool…')");
  await page.waitForSelector("dialog select[aria-label='Pool preset'] option[value=fastwan21]", { state: "attached" });
  await page.selectOption("dialog select[aria-label='Pool preset']", "fastwan21");
  await page.click("dialog .presets button:text('Add pool')");
  for (let i = 0; i < 40; i++) {
    const d = await (await fetch(`${B}/api/docs/cluster-spec/${c.cluster.id}`, { headers: { authorization: `Bearer ${tok}` } })).json();
    if (d.doc.pools.some((p) => p.id === "fastwan21")) break;
    assert.ok(i < 39, "the pool was added");
    await page.waitForTimeout(250);
  }
  await page.waitForSelector("td:text-is('fastwan21')");
  await page.click("button:text('Keys')");
  await page.waitForSelector("#gwKeys td:text('ui-key')");
  await page.click("#gwKeys button:text('Revoke')");
  await page.waitForSelector("#gwKeys .badge:text('revoked')");
  assert.match(await page.textContent("#gwOut"), /revoked key_/);
  await page.screenshot({ path: `${out}/15-cluster-keys.png`, fullPage: true });
  await page.click("button:text('Restart…')");
  await page.waitForSelector("dialog.pick input[value='pool:fake']");
  assert.ok(!(await page.$("dialog.pick input[value='pool:gateway']")), "no gateway pool to restart");
  await page.check("dialog.pick input[value='pool:fake']");
  await page.screenshot({ path: `${out}/16-restart-picker.png` });
  await page.click("dialog.pick button:text('Restart')");
  await page.waitForSelector("text=running: restart");
  const ops = await (await fetch(`${B}/api/clusters/${c.cluster.id}/ops`, { headers: { authorization: `Bearer ${tok}` } })).json();
  assert.deepEqual(JSON.parse(ops.operations.find((o) => o.kind === "restart").params), { pools: ["fake"] });
  // Env: the pool level, with the engine's keys suggested.
  await page.goto(`${B}/#/env?cluster=${c.cluster.id}&pool=fake`);
  await page.waitForSelector("select[aria-label=Pool]");
  const poolp = ".fv-panel[data-kind=env] >> nth=2";
  await page.waitForSelector(`${poolp} >> button:text('+ variable')`);
  assert.ok(await page.$(`datalist option[value=FASTVIDEO_ATTN_SAGE]`), "engine env keys suggested");
  await page.click(`${poolp} >> button:text('+ variable')`);
  await page.fill(`${poolp} >> input[aria-label="variable name"]`, "FASTVIDEO_ATTN_SAGE");
  await page.press(`${poolp} >> input[aria-label="variable name"]`, "Tab");
  // A known engine key: a select of the values the engine accepts.
  await page.selectOption(`${poolp} >> tr[data-key=FASTVIDEO_ATTN_SAGE] select`, "0");
  await page.click(`${poolp} >> button:text('Review & save')`);
  await page.click(`${poolp} >> button:has-text('Save (v')`);
  for (let i = 0; i < 40; i++) {
    const v = await (await fetch(`${B}/api/env/pool/${c.cluster.id}:fake`, { headers: { authorization: `Bearer ${tok}` } })).json();
    if (v.vars?.some((x) => x.key === "FASTVIDEO_ATTN_SAGE" && x.value === "0")) break;
    assert.ok(i < 39, "the pool variable was saved");
    await page.waitForTimeout(250);
  }
  await page.screenshot({ path: `${out}/17-env-pool.png`, fullPage: true });
  await page.goto(`${B}/#/settings`);
  await page.waitForSelector("text=build pod backstop");
  assert.ok(!(await page.content()).includes("Auto-actions touch controller clusters only, never external pods"), "the stale policy text is gone");
  page.off("dialog", accept);

  // ---- standalone pods: launch from the form (session + CSRF), its page with status, cost and the boot timeline.
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} [data-ctl][data-path=name]`);
  await page.fill(`${SL} [data-ctl][data-path=name]`, "ui-solo");
  // Custom: the cpu variant with the fake config and fake-wan (what "custom" starts with), CPU, no volume.
  await page.selectOption(`${SL} [data-ctl][data-path=preset]`, "");
  await page.waitForSelector(`${SL} select[data-ctl][data-path=variant]`);
  assert.equal(await page.inputValue(`${SL} [data-ctl][data-path=config]`), "/etc/fv/runpod-fake.toml");
  await page.waitForFunction((s) => !document.querySelector(`${s} button[type=submit]`).disabled, SL, { timeout: 15000 });
  await page.click(`${SL} button[type=submit]:text('Launch')`);
  await page.waitForSelector("h1:text('ui-solo')");
  await page.waitForSelector("h2:text('Boot timeline')", { timeout: 60000 });
  await page.waitForSelector("td:text('Runpod create accepted')", { timeout: 60000 });
  await page.screenshot({ path: `${out}/12-standalone.png`, fullPage: true });
  const sp = await (await fetch(`${B}/api/standalone/ui-solo`, { headers: { authorization: `Bearer ${tok}` } })).json();
  assert.equal(sp.pod.definition.variant, "cpu");
  assert.equal(sp.pod.definition.compute, "CPU");
  await fetch(`${B}/api/standalone/ui-solo`, { method: "DELETE", headers: { authorization: `Bearer ${tok}` } });

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
  for (const hash of ["#/", "#/clusters", "#/pods", "#/standalone", "#/env", "#/costs", "#/settings"]) {
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
