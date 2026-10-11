// UI test (headless Chromium, playwright-core): named pod tunnels (src/tunnels.ts) with a host template two levels
// below the zone: the Pods page's tunnels card and the launch form say loudly that Universal SSL does not cover it
// (an Advanced Certificate Manager certificate is needed); a recorded tunnel is listed, its token nowhere on the page.
// Simulated Cloudflare / GMI only (test/harness.mjs).
//   CHROMIUM=/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell node test/ui/tunnels.mjs
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { d1Exec, hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
mock.gmiKey = SECRETS.GMI_API_KEY;
mock.cf.token = "cftun_UI_0123456789abcdef";
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  GMI_PRODUCTS: "container.h200.x1",
  GMI_DEFAULT_IDC: "us-denver-1",
  GMI_BUDGET_USD: "50",
  CF_TUNNEL_API_TOKEN: mock.cf.token,
  FV_TUNNEL_ZONE: "maximal.llc",
  FV_TUNNEL_HOST_TEMPLATE: "{pod}.fv-pods.maximal.llc",
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
  // One recorded live tunnel (as fv-control records it) of a GMI pod.
  const t = Date.now();
  d1Exec(
    w.dir,
    `INSERT INTO pod_tunnels (name, pod_id, host, url, zone_id, tunnel_id, dns_record_id, state, created_at, updated_at)
     VALUES ('fv-pod-ui1-1011000000-abc123', 'gmi:fv-pod-ui1-1011000000', 'fv-pod-ui1-1011000000-abc123.fv-pods.maximal.llc', 'https://fv-pod-ui1-1011000000-abc123.fv-pods.maximal.llc', 'z', 't-ui-1', 'r-ui-1', 'live', ${t}, ${t})`,
  );

  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  const validations = [];
  page.on("response", async (r) => r.url().includes("/api/standalone/validate") && validations.push(await r.text().catch(() => "")));
  await page.goto(B + "/");
  await page.waitForSelector("#pass");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");

  // ---- the Pods page: the tunnels card, its warning and the recorded tunnel
  await page.goto(`${B}/#/pods`);
  const card = page.locator(".card", { hasText: "Pod tunnels (GMI / Brev)" });
  await card.waitFor();
  const text = await card.textContent();
  for (const s of ["Pod tunnels (GMI / Brev): named · 1 recorded", "{pod}.fv-pods.maximal.llc", "2 levels below maximal.llc", "Universal SSL covers only *.maximal.llc", "Advanced Certificate Manager certificate for *.fv-pods.maximal.llc", "{pod}-fv-pods.maximal.llc", "gmi:fv-pod-ui1-1011000000", "fv-pod-ui1-1011000000-abc123.fv-pods.maximal.llc", "live"])
    assert.ok(text.includes(s), `tunnels card: ${s} in ${text}`);
  assert.equal(await card.locator("[data-tunnel-warning]").count(), 1);
  await page.screenshot({ path: `${out}/30-pod-tunnels.png`, fullPage: true });

  // ---- the launch form: a GMI pod shows the same warning at its provider field
  await page.goto(`${B}/#/standalone`);
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, JSON.stringify("gmi"));
  await page.waitForSelector(`${SL} .ff-field[data-path="provider_gpu"]`);
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "uitun");
  await page.fill(`${SL} [data-ctl][data-path="provider_gpu"]`, "container.h200.x1");
  await page.waitForSelector(`${SL} .ff-field[data-path="provider"] .ff-warn:not([hidden]):text-matches("Advanced Certificate Manager")`, { timeout: 20000 }).catch((e) => {
    throw new Error(`${e.message}\nlast validations: ${validations.slice(-2).join("\n")}`);
  });
  const warn = await page.textContent(`${SL} .ff-field[data-path="provider"] .ff-warn`);
  assert.match(warn, /WARNING: FV_TUNNEL_HOST_TEMPLATE \{pod\}\.fv-pods\.maximal\.llc puts pod hosts 2 levels below maximal\.llc/, warn);
  await page.screenshot({ path: `${out}/31-launch-tunnel-warning.png`, fullPage: true });

  assert.ok(!(await page.content()).includes(mock.cf.token), "the API token is never on a page");
  assert.deepEqual(errors.filter((e) => !/login required|401/.test(e)), [], "no console errors");
  console.log(`ok   UI named pod tunnels: the tunnels card (mode, two-level host warning, recorded tunnel) and the launch form's warning (screenshots in ${out})`);
} catch (e) {
  fail(e);
}
await browser.close();
w.stop();
mock.close();
