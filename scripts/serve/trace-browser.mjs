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
const browser = await chromium.launch();
const ctx = await browser.newContext();
await ctx.addInitScript(([key]) => {
  localStorage.setItem('fv.trace', '1');
  if (key) localStorage.setItem('fv.key', key);
}, [o.key || '']);
const page = await ctx.newPage();
for (let i = 0; i < o.n; i++) {
  await page.goto(o.base + '/console/models/' + o.page, { waitUntil: 'networkidle' });
  await page.waitForSelector('[data-input=prompt]');
  await page.fill('[data-input=prompt]', o.prompt);
  for (const [k, v] of o.sets) await page.selectOption('[data-input=' + k + ']', v).catch(() => page.fill('[data-input=' + k + ']', v));
  await page.evaluate(() => { globalThis.__fvLastTrace = null; });
  await page.click('#run');
  await page.waitForFunction(() => globalThis.__fvLastTrace, null, { timeout: 300_000 });
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
