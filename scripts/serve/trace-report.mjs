#!/usr/bin/env node
// The waterfall of one trace (docs/serve/tracing.md): every step with its
// start (ms from the click) and duration, the hosts' clock offsets and
// their uncertainty, the critical path and the unaccounted gaps.
//
//   node scripts/serve/trace-report.mjs trace-<id>.json          # a file trace-bench.mjs wrote
//   node scripts/serve/trace-report.mjs --base URL <trace id> [--header 'k: v']…   # from a server
//   … [--json]                                                     # the analysis as JSON
//
// From a server, `GET /fv/v1/traces/{id}` has every event the pod recorded
// plus what the edge and the browser posted to it.

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const { analyze, textReport } = await import(join(HERE, '../../crates/fastvideo-serve/console/trace.js'));

let base = null; let json = false; const headers = {}; const rest = [];
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a === '--base') base = process.argv[++i].replace(/\/+$/, '');
  else if (a === '--json') json = true;
  else if (a === '--header') { const x = process.argv[++i]; const k = x.indexOf(':'); headers[x.slice(0, k).trim()] = x.slice(k + 1).trim(); }
  else rest.push(a);
}
if (rest.length !== 1) {
  console.error('usage: trace-report.mjs <trace.json> | --base URL <trace id> [--json]');
  process.exit(2);
}
let events;
if (base) {
  const r = await fetch(base + '/fv/v1/traces/' + rest[0], { headers });
  if (!r.ok) { console.error('GET trace: ' + r.status + ' ' + (await r.text())); process.exit(1); }
  const d = await r.json();
  events = d.events;
  if (d.stats) console.error(`recorder: sent ${d.stats.sent}, dropped ${d.stats.dropped}, capacity ${d.stats.capacity}; truncated ${d.truncated}`);
} else {
  const v = JSON.parse(readFileSync(rest[0], 'utf8'));
  events = Array.isArray(v) ? v : v.events;
}
const res = analyze(events);
if (json) console.log(JSON.stringify(res, null, 2));
else console.log(textReport(res));
