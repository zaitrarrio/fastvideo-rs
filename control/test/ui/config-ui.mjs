// The typed-configuration UI checks on their own (test/ui/config-validation.mjs; also run by explorer.mjs):
//   node test/ui/config-ui.mjs        CHROMIUM=<path>, SHOTS=<dir>
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";
import { hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";
import { configValidationChecks } from "./config-validation.mjs";

const out = process.env.SHOTS || `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
const w = await startWorker(mock, { ...SECRETS, OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET), EDGE_URL: `http://127.0.0.1:${mock.port}/edge`, EDGE_INTERNAL_TOKEN: mock.edgeInternal, EDGE_ADMIN_TOKEN: mock.edgeAdmin, EDGE_D1_DATABASE_ID: "d1-edge-staging" });
const B = w.url;
let browser;
let page;
try {
  const login = await fetch(`${B}/api/auth/login`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ passphrase: PASSPHRASE }) });
  const cookie = login.headers.get("set-cookie").split(";")[0];
  const { csrf } = await login.json();
  const tok = (await (await fetch(`${B}/api/tokens`, { method: "POST", headers: { cookie, "x-csrf-token": csrf, "content-type": "application/json" }, body: JSON.stringify({ name: "seed" }) })).json()).token;
  const raw = (p, body, method) => fetch(B + p, { method: method || (body ? "POST" : "GET"), headers: { authorization: `Bearer ${tok}`, "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const api = (p, body, method) => raw(p, body, method).then((r) => r.json());
  const tiny = await api("/api/clusters", { spec: { name: "tiny", template: "tiny-cpu", cap_s: 3600 } });
  await api(`/api/clusters/${tiny.cluster.id}/start`, {});
  for (let i = 0; i < 60; i++) {
    const ops = await api(`/api/clusters/${tiny.cluster.id}/ops`);
    if (ops.operations[0]?.status !== "running") break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  page = await (await browser.newContext({ viewport: { width: 1440, height: 920 } })).newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(B + "/");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");
  await configValidationChecks({ page, B, api, raw, out, tiny });
  if (errors.length) throw new Error(`page errors: ${errors.join("; ")}`);
  console.log(`ok   UI typed configuration: bound controls, inline errors, submit gated, coverage audit, server refusals (screenshots in ${out})`);
} catch (e) {
  console.log(`FAIL ${e.stack || e.message}`);
  if (page) await page.screenshot({ path: `${out}/FAIL.png`, fullPage: true }).catch(() => {});
  console.log(w.output().slice(-1500));
  process.exitCode = 1;
}
await browser?.close();
w.stop();
mock.close();
