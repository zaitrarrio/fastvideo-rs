#!/usr/bin/env node
// The console's own click, traced (docs/serve/tracing.md): headless
// Chromium opens a model page, ticks Trace, sets the prompt (and the given
// select fields), clicks Run and waits for the Trace tab, i.e. for the video
// to be playable and the beacon to be sent. Then it reads every event of
// the trace from the server and prints the waterfall
// (crates/fastvideo-serve/console/trace.js).
//
//   node scripts/serve/trace-browser.mjs --base URL --page minimax/h3-turbo/text-to-video \
//        [--key-file F] [--set resolution=480P] [--set duration=5] [--n 3] [--out DIR]
//
// Needs Playwright with Chromium (NODE_PATH with `playwright`; the build
// pod image and the dev containers have it; never `playwright install`).

import { writeFileSync, mkdirSync, readFileSync, appendFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const HERE = dirname(fileURLToPath(import.meta.url));
const { analyze, textReport } = await import(join(HERE, '../../crates/fastvideo-serve/console/trace.js'));
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.FV_PLAYWRIGHT || 'playwright');

const o = { n: 1, sets: [], out: 'artifacts/trace-browser', prompt: 'A red fox runs through fresh snow in a pine forest at dawn, cinematic.' };
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i]; const v = () => process.argv[++i];
  if (a === '--base') o.base = v().replace(/\/+$/, '');
  else if (a === '--page') o.page = v();
  else if (a === '--key-file') o.key = readFileSync(v(), 'utf8').trim();
  else if (a === '--set') o.sets.push(v().split('='));
  else if (a === '--prompt') o.prompt = v();
  else if (a === '--n') o.n = Number(v());
  else if (a === '--out') o.out = v();
  else if (a === '--label') o.label = v();
  else throw new Error('unknown argument ' + a);
}
mkdirSync(o.out, { recursive: true });
// Behind an egress proxy (HTTPS_PROXY), Chromium goes through it too.
const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
const browser = await chromium.launch(proxy ? { proxy: { server: proxy } } : {});
// FV_BROWSER_IGNORE_TLS=1: a sandbox whose egress proxy re-signs TLS with its own CA.
const ctx = await browser.newContext({ ignoreHTTPSErrors: process.env.FV_BROWSER_IGNORE_TLS === '1' });
await ctx.addInitScript(([key]) => {
  localStorage.setItem('fv.trace', '1');
  if (key) localStorage.setItem('fv.key', key);
}, [o.key || '']);
const page = await ctx.newPage();
page.on('requestfailed', (r) => console.error('request failed: ' + r.url().slice(0, 120) + ' ' + (r.failure() || {}).errorText));
// FV_BROWSER_RELAY=1: requests to --base go through Node's fetch (a sandbox
// whose egress refuses some of the browser's requests). Browser-side times
// then include the relay; the bench client's times are the reference.
if (process.env.FV_BROWSER_RELAY === '1') {
  await page.route(o.base + '/**', async (route) => {
    const q = route.request();
    // FV_BROWSER_LOCAL_CONSOLE=1: this checkout's console scripts instead of the pod's.
    const asset = process.env.FV_BROWSER_LOCAL_CONSOLE === '1' && q.url().match(/\/console\/assets\/([a-z]+\.(?:js|css))$/);
    if (asset) {
      const body = readFileSync(join(HERE, '../../crates/fastvideo-serve/console', asset[1]));
      await route.fulfill({ status: 200, headers: { 'content-type': asset[1].endsWith('.css') ? 'text/css' : 'text/javascript' }, body });
      return;
    }
    const h = { ...q.headers() };
    delete h.host;
    const r = await fetch(q.url(), { method: q.method(), headers: h, body: ['GET', 'HEAD'].includes(q.method()) ? undefined : q.postDataBuffer() });
    const headers = {};
    r.headers.forEach((v, k) => { if (!['content-encoding', 'content-length', 'transfer-encoding'].includes(k)) headers[k] = v; });
    await route.fulfill({ status: r.status, headers, body: Buffer.from(await r.arrayBuffer()) });
  });
}
for (let i = 0; i < o.n; i++) {
  await page.goto(o.base + '/console/models/' + o.page, { waitUntil: 'load' });
  await page.waitForSelector('[data-input=prompt]');
  await page.fill('[data-input=prompt]', o.prompt);
  for (const [k, v] of o.sets) await page.selectOption('[data-input=' + k + ']', v).catch(() => page.fill('[data-input=' + k + ']', v));
  await page.evaluate(() => { globalThis.__fvLastTrace = null; });
  await page.click('#run');
  try {
    await page.waitForFunction(() => globalThis.__fvLastTrace, null, { timeout: 300_000 });
  } catch (e) {
    // What the page shows instead (status, messages, the player's state).
    const st = await page.evaluate(() => {
      const v = document.getElementById('video');
      const t = (id) => (document.getElementById(id) || {}).textContent;
      return { status: t('result-status'), run: t('run-msg'), result: t('result-msg'), empty: t('video-empty'), src: v && v.src, ready: v && v.readyState, err: v && v.error && v.error.code };
    });
    console.error('no trace: ' + JSON.stringify(st));
    throw e;
  }
  const t = await page.evaluate(() => globalThis.__fvLastTrace);
  await new Promise((r) => setTimeout(r, 500));
  const d = await (await fetch(o.base + '/fv/v1/traces/' + t.id)).json();
  writeFileSync(join(o.out, 'trace-' + t.id + '.json'), JSON.stringify(d.events));
  const res = analyze(d.events);
  appendFileSync(join(o.out, 'runs.jsonl'), JSON.stringify({ label: o.label, trace: t.id, totalMs: res.totalMs, unaccountedMs: res.unaccountedMs, dropped: d.stats && d.stats.dropped, phases: res.phases }) + '\n');
  console.log(`== browser run ${i} trace ${t.id} (${o.label || ''}) ==`);
  console.log(textReport(res));
}
await browser.close();
