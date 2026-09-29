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
  // Set a secret env var through the UI; it shows masked.
  await page.goto(`${B}/#/env?cluster=${c.cluster.id}`);
  await page.waitForSelector("h1:text('Environment')");
  const inputs = await page.$$("section.card input[placeholder=NAME]");
  await inputs[1].fill("HF_TOKEN");
  const vals = await page.$$("section.card input[placeholder=value]");
  await vals[1].fill("hf_ui_secret");
  const secs = await page.$$("section.card input[type=checkbox]");
  await secs[1].check();
  const sets = await page.$$("section.card button.primary:text('Set')");
  await sets[1].click();
  await page.waitForSelector("text=pod(s) need a restart");
  assert.ok(!(await page.content()).includes("hf_ui_secret"), "the secret is never rendered");
  await page.screenshot({ path: `${out}/05-env.png`, fullPage: true });

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
