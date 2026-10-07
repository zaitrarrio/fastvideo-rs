// UI test of the serverless console (docs/control/serverless.md "Console",
// src/serverless/console.ts) in headless Chromium (playwright-core): the
// endpoint page's "Open console", fv-serve's console served by fv-control
// under /serverless/<endpoint>, a fal model page (text-to-video, and
// image-to-video with an uploaded image) and the Native API page, each run
// as Runpod queue jobs on a simulated worker (test/harness.mjs fakeServe).
//   node test/ui/console.mjs
// CHROMIUM=<path> picks the browser; SHOTS=<dir> where screenshots go (default test-results/).
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright-core";
import { d1Exec, fakeServe, hashPassphrase, HERE, JOBS_TABLE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = process.env.SHOTS || `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  CONSOLE_SUBMIT_WAIT_MS: "0",
  CONSOLE_POLL_MS: "200",
  CONSOLE_CACHE_WAIT_MS: "8000",
});
const B = w.url;
let browser;
const fail = (e) => {
  console.log(`FAIL ${e.stack || e.message}`);
  console.log(w.output().slice(-3000));
  browser?.close();
  w.stop();
  mock.close();
  process.exit(1);
};

try {
  const EID = "rpuicons00001";
  const now = Date.now();
  const spec = JSON.stringify({ name: "ui-cons", mode: "queue", variant: "cpu", compute: "CPU", workers_min: 0, workers_max: 1, idle_timeout_s: 5, flashboot: false, execution_timeout_s: 600, deadline_min: 120, deadline_action: "delete", image: { channel: "stable" }, cpu_flavors: ["cpu3c"], vcpu: 2, network_volume: null, scaler_type: "QUEUE_DELAY", scaler_value: 4, container_disk_gb: 20 });
  d1Exec(w.dir, `INSERT INTO serverless_endpoints (id, name, endpoint_id, template_id, mode, spec, image, status, created_at, created_by, updated_at) VALUES ('se_uicons', 'ui-cons', '${EID}', 'tpl', 'queue', '${spec}', 'ghcr.io/x/serve@sha256:0123456789abcdef', 'active', ${now}, 'owner', ${now})`);
  d1Exec(w.dir, JOBS_TABLE, "fv-jobs");
  for (let ok = 0, i = 0; i < 40 && ok < 2; i++) {
    ok = (await fetch(`${B}/healthz`).then((r) => r.ok).catch(() => false)) ? ok + 1 : 0;
    await new Promise((r) => setTimeout(r, 250));
  }
  mock.queue.onRun = fakeServe(mock, { steps: 2 });
  mock.queue.workers = 1;

  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 920 } });
  const page = await ctx.newPage();
  const errors = [];
  // The model page asks for the Reactor runtime's /schema (404 here: no Reactor through the proxy) and logs the 404.
  page.on("console", (m) => m.type() === "error" && !/status of 404/.test(m.text()) && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  const shot = (n) => page.screenshot({ path: `${out}/console-${n}.png`, fullPage: true });

  // ---- the dashboard: the endpoint page's "Open console"
  await page.goto(B + "/");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");
  await page.goto(`${B}/#/serverless?ep=se_uicons`);
  const open = page.locator("#open-console");
  await open.waitFor();
  const href = await open.getAttribute("href");
  assert.equal(href, `/serverless/${EID}/console`);
  assert.equal(await open.getAttribute("target"), "_blank");

  // ---- the console's home: served models from the cached capabilities, the fal app, no key field
  await page.goto(B + href);
  await page.waitForSelector('#served tr[data-model="fake-wan"]', { timeout: 20000 });
  await page.waitForSelector('[data-endpoint="fastvideo/fake-wan/text-to-video"]');
  assert.equal(await page.locator("#key-fields").isHidden(), true, "no API key field (the proxy is authenticated)");
  assert.equal(await page.inputValue("#base"), `${B}/serverless/${EID}`);
  assert.equal(await page.locator("#base").getAttribute("readonly"), "");
  assert.match(await page.textContent("#embed-note"), /Serverless endpoint fvc-ui-cons .* Live pages/);
  const nav = await page.locator(".topnav a").allTextContents();
  assert.deepEqual(nav, ["Models", "Native API"], "live pages and API keys are off");
  await page.waitForSelector('#status-strip [data-state="ready"]');
  await shot("home");

  // ---- a fal model page: Run → queue job → the video
  await page.click('[data-endpoint="fastvideo/fake-wan/text-to-video"]');
  await page.waitForURL(`**/serverless/${EID}/console/models/fastvideo/fake-wan/text-to-video`);
  await page.waitForSelector('[data-input="prompt"]');
  assert.deepEqual(await page.locator("#tasks a").allTextContents(), ["Text to video", "Image to video"]);
  await page.fill('[data-input="prompt"]', "a red fox in the snow");
  await page.click("#run");
  await page.waitForSelector("#video:not([hidden])", { timeout: 30000 });
  const src = await page.getAttribute("#video", "src");
  assert.match(src, /\/media\/f-sim-\d+\.mp4$/);
  assert.equal(await page.textContent("#result-status"), "completed");
  assert.match(await page.textContent("#output-json"), /"seed": 42/);
  assert.equal(await page.locator('#history tbody tr').count(), 1);
  await shot("model");
  const sent = mock.queue.ran.filter((r) => r.input.method === "POST" && r.input.path === "/fastvideo/fake-wan/text-to-video");
  assert.equal(sent.length, 1);
  assert.equal(sent[0].input.body.prompt, "a red fox in the snow");
  assert.equal(sent[0].input.wait, true);

  // ---- image-to-video: the image uploads through fv-control (R2), the job gets its signed URL
  await page.click('#tasks a[data-task="image-to-video"]');
  await page.waitForSelector('[data-field-file="image_url"]', { state: "attached" });
  await page.setInputFiles('[data-field-file="image_url"]', { name: "fox.png", mimeType: "image/png", buffer: Buffer.from("89504e470d0a1a0a", "hex") });
  await page.fill('[data-input="prompt"]', "the fox turns its head");
  await page.waitForFunction(() => !document.querySelector('[data-field="image_url"]')?.textContent?.includes("uploading"));
  await page.click("#run");
  await page.waitForSelector("#video:not([hidden])", { timeout: 30000 });
  const i2v = mock.queue.ran.filter((r) => r.input.path === "/fastvideo/fake-wan/image-to-video").at(-1);
  assert.ok(i2v, "the image-to-video job went out");
  assert.match(i2v.input.body.image_url, new RegExp(`/serverless-uploads/[^/]+/fox\\.png$`));
  const img = await fetch(i2v.input.body.image_url);
  assert.equal(img.status, 200, "the worker can read the upload back");
  assert.equal(Buffer.from(await img.arrayBuffer()).toString("hex"), "89504e470d0a1a0a");

  // ---- the Native API page
  await page.click('.topnav a:text("Native API")');
  await page.waitForURL(`**/serverless/${EID}/console/native`);
  await page.waitForSelector('#model option[value="fake-wan"]', { state: "attached" });
  await page.fill('#n-prompt', "a lighthouse at dusk");
  await page.click("#run");
  await page.waitForSelector('body[data-job="succeeded"]', { timeout: 30000 });
  assert.match(await page.getAttribute("#video", "src"), /\/media\/fvjob_sim\d+\.mp4$/);
  await shot("native");

  // ---- a page the proxy does not serve
  const off = await page.goto(`${B}/serverless/${EID}/console/stream`);
  assert.equal(off.status(), 404);
  assert.match(await page.textContent("main"), /Not available on a serverless endpoint/);

  assert.deepEqual(errors, [], `browser errors: ${errors.join("\n")}`);
  writeFileSync(`${out}/console-served.txt`, (mock.served || []).join("\n"));
  console.log(`ok   serverless console UI (${mock.queue.ran.length} queue jobs; screenshots in ${out})`);
  await browser.close();
  w.stop();
  mock.close();
} catch (e) {
  fail(e);
}
