#!/usr/bin/env node
// Warm-request end-to-end bench with request tracing (docs/serve/tracing.md,
// results in docs/serve/bench/e2e-warm-trace.md).
//
// Runs the console's own path (fal queue API: submit, status polls every
// --poll-ms like the console, result, then the whole video) N times after
// W warm-up requests, through any base URL: the edge (the cluster path the
// console uses) or a pod (direct). With --ab the measured runs alternate
// traced / untraced (ABAB…) so both see the same drift; client-side times
// are taken the same way in both. Traced runs ship their client events to
// the server after the run and fetch the server's events for the
// waterfall (crates/fastvideo-serve/console/trace.js `analyze`).
//
//   node scripts/serve/trace-bench.mjs --base URL --endpoint fal-ai/h3/turbo/text-to-video \
//        [--key K | --key-file F] [--auth key|bearer|none] [--header 'k: v']… \
//        [--input '{"resolution":"480P","duration":5}'] [--prompt P] [--fresh-prompt] \
//        [--n 10] [--warmup 2] [--ab] [--traced-only] [--poll-ms 700] [--out DIR] [--label L]
//
// Output: <out>/runs.jsonl (one line per run), <out>/trace-<id>.json (all
// events of a traced run), <out>/summary.json and a text summary on stdout.

import { mkdirSync, writeFileSync, appendFileSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const { ClientTrace, analyze } = await import(join(HERE, '../../crates/fastvideo-serve/console/trace.js'));

function args(argv) {
  const o = { n: 10, warmup: 2, pollMs: 700, auth: 'key', headers: [], input: '{}', prompt: 'A red fox runs through fresh snow in a pine forest at dawn, cinematic.', out: 'artifacts/trace-bench', label: '' };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const v = () => argv[++i];
    if (a === '--base') o.base = v().replace(/\/+$/, '');
    else if (a === '--endpoint') o.endpoint = v();
    else if (a === '--key') o.key = v();
    else if (a === '--key-file') o.key = readFileSync(v(), 'utf8').trim();
    else if (a === '--auth') o.auth = v();
    else if (a === '--header') o.headers.push(v());
    else if (a === '--input') o.input = v();
    else if (a === '--prompt') o.prompt = v();
    else if (a === '--fresh-prompt') o.fresh = true;
    else if (a === '--n') o.n = Number(v());
    else if (a === '--warmup') o.warmup = Number(v());
    else if (a === '--ab') o.ab = true;
    else if (a === '--traced-only') o.tracedOnly = true;
    else if (a === '--poll-ms') o.pollMs = Number(v());
    else if (a === '--out') o.out = v();
    else if (a === '--label') o.label = v();
    else throw new Error('unknown argument ' + a);
  }
  if (!o.base || !o.endpoint) throw new Error('--base and --endpoint are required');
  return o;
}

const opt = args(process.argv);
const app = opt.endpoint.split('/').slice(0, 2).join('/');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const wall = () => performance.timeOrigin + performance.now();

function baseHeaders() {
  const h = { Accept: 'application/json' };
  if (opt.key && opt.auth !== 'none') h.Authorization = (opt.auth === 'bearer' ? 'Bearer ' : 'Key ') + opt.key;
  for (const x of opt.headers) { const i = x.indexOf(':'); h[x.slice(0, i).trim()] = x.slice(i + 1).trim(); }
  return h;
}

// One HTTP exchange, recorded the same way traced or not.
async function call(method, url, { body, trace, name, times }) {
  const headers = { ...baseHeaders(), ...(trace ? trace.headers() : {}) };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const t0 = wall();
  const r = await fetch(url, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const tHead = wall();
  const text = await r.text();
  const tEnd = wall();
  let parsed = text;
  try { parsed = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (trace) trace.http(name, t0, tHead, tEnd, r.status, r.headers, parsed);
  times.push({ name, t0, tHead, tEnd, status: r.status });
  if (!r.ok) throw new Error(method + ' ' + url + ' → ' + r.status + ' ' + text.slice(0, 300));
  return { body: parsed, headers: r.headers };
}

async function one(i, traced, phase) {
  const trace = traced ? new ClientTrace({ host: 'client' }) : null;
  const times = [];
  const input = { ...JSON.parse(opt.input), prompt: opt.fresh ? opt.prompt + ' (' + Date.now() + '-' + i + ')' : opt.prompt };
  const click = wall();
  if (trace) trace.point('click', undefined, click);
  const sub = await call('POST', opt.base + '/' + opt.endpoint, { body: input, trace, name: 'submit', times });
  const id = sub.body.request_id;
  let polls = 0;
  let terminalSeen = null;
  for (;;) {
    await sleep(opt.pollMs);
    const st = await call('GET', opt.base + '/' + app + '/requests/' + encodeURIComponent(id) + '/status', { trace, name: 'poll', times });
    polls++;
    if (st.body.status === 'COMPLETED') { terminalSeen = wall(); break; }
    if (st.body.status === 'FAILED' || st.body.status === 'CANCELLED') throw new Error('job ' + id + ' ' + st.body.status);
  }
  const res = await call('GET', opt.base + '/' + app + '/requests/' + encodeURIComponent(id), { trace, name: 'result', times });
  const url = res.body && res.body.video && res.body.video.url;
  // The video, read to the end: first byte and last byte.
  const tv0 = wall();
  const vr = await fetch(url);
  const reader = vr.body.getReader();
  let first = null; let bytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (first === null) first = wall();
    bytes += value.length;
  }
  const tv1 = wall();
  if (trace) {
    trace.spanMs('video_fetch', tv0, tv1, { bytes, status: vr.status });
    if (first) { trace.spanMs('video_ttfb', tv0, first); trace.spanMs('video_bytes', first, tv1); }
    trace.point('canplay', { note: 'bench: last byte (no player)' }, tv1);
    trace.spanMs('e2e', click, tv1);
  }
  const rec = {
    i, phase, traced, id, trace: trace ? trace.id : null,
    e2e_ms: tv1 - click,
    submit_ms: times[0].tHead - times[0].t0,
    seen_terminal_ms: terminalSeen - click,
    result_ms: times[times.length - 1].tEnd - times[times.length - 1].t0,
    video_ttfb_ms: first ? first - tv0 : null,
    video_ms: tv1 - tv0, video_bytes: bytes, polls,
    timings: res.body.timings || null,
  };
  if (trace) {
    // After the run: ship the client's events, read everyone's.
    await fetch(opt.base + '/fv/v1/traces/' + trace.id + '/events', { method: 'POST', headers: { ...baseHeaders(), 'content-type': 'application/json' }, body: JSON.stringify({ events: trace.events }) }).catch(() => {});
    await sleep(300);
    let server = [];
    try {
      const r = await fetch(opt.base + '/fv/v1/traces/' + trace.id, { headers: baseHeaders() });
      const d = await r.json();
      server = d.events || [];
      rec.dropped = d.stats ? d.stats.dropped : null;
      rec.truncated = d.truncated;
    } catch (e) { rec.trace_error = String(e); }
    const mine = new Set(trace.events.map((e) => e.name + ':' + e.t_wall_ns));
    const all = [...trace.events, ...server.filter((e) => !(e.host === 'client' && mine.has(e.name + ':' + e.t_wall_ns)))];
    writeFileSync(join(opt.out, 'trace-' + trace.id + '.json'), JSON.stringify(all));
    const a = analyze(all);
    rec.unaccounted_ms = a.unaccountedMs;
    rec.phases = Object.fromEntries(a.phases.map((p) => [p.from + ' → ' + p.to, p.ms]));
    rec.steps = stepTotals(a.rows);
    rec.offsets = Object.fromEntries(Object.entries(a.offsets).map(([h, o]) => [h, { offset_ms: o.offset / 1e6, uncert_ms: o.uncert / 1e6 }]));
  }
  return rec;
}

// Per step: total duration (repeated names summed: polls, denoise steps) and count.
function stepTotals(rows) {
  const m = {};
  for (const r of rows) {
    const k = r.comp + '.' + r.name + (r.clock === 'gpu' ? ' (gpu)' : '');
    const e = (m[k] = m[k] || { ms: 0, n: 0 });
    e.ms += r.durMs; e.n += 1;
  }
  return m;
}

const q = (xs, p) => {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const k = (v.length - 1) * p; const lo = Math.floor(k); const hi = Math.ceil(k);
  return v[lo] + (v[hi] - v[lo]) * (k - lo);
};
const stat = (xs) => ({ n: xs.filter(Number.isFinite).length, median: q(xs, 0.5), p90: q(xs, 0.9), min: q(xs, 0), max: q(xs, 1) });

// Noise of a median difference: bootstrap 95 % interval of median(A) − median(B).
function bootDiff(a, b, iters = 4000) {
  if (a.length < 2 || b.length < 2) return null;
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const pick = (xs) => xs.map(() => xs[Math.floor(rnd() * xs.length)]);
  const d = [];
  for (let i = 0; i < iters; i++) d.push(q(pick(a), 0.5) - q(pick(b), 0.5));
  return { lo: q(d, 0.025), hi: q(d, 0.975) };
}

mkdirSync(opt.out, { recursive: true });
const runsFile = join(opt.out, 'runs.jsonl');
const runs = [];
console.error(`[bench] ${opt.label} base=${opt.base} endpoint=${opt.endpoint} warmup=${opt.warmup} n=${opt.n} ab=${!!opt.ab}`);
for (let i = 0; i < opt.warmup; i++) {
  const r = await one(i, false, 'warmup');
  console.error(`[warmup ${i}] e2e ${r.e2e_ms.toFixed(0)} ms`);
}
for (let i = 0; i < opt.n; i++) {
  const traced = opt.tracedOnly ? true : opt.ab ? i % 2 === 0 : true;
  const r = await one(i, traced, 'measure');
  r.label = opt.label;
  runs.push(r);
  appendFileSync(runsFile, JSON.stringify(r) + '\n');
  console.error(`[run ${i}] ${traced ? 'traced  ' : 'untraced'} e2e ${r.e2e_ms.toFixed(0)} ms, inference ${r.timings && r.timings.inference != null ? (r.timings.inference * 1e3).toFixed(0) : '?'} ms${r.trace ? ', trace ' + r.trace + ', unaccounted ' + r.unaccounted_ms.toFixed(0) + ' ms, dropped ' + r.dropped : ''}`);
}

const on = runs.filter((r) => r.traced); const off = runs.filter((r) => !r.traced);
const metric = (rs, f) => rs.map(f).filter((x) => x !== null && x !== undefined && Number.isFinite(x));
const keys = {
  e2e_ms: (r) => r.e2e_ms,
  seen_terminal_ms: (r) => r.seen_terminal_ms,
  submit_ms: (r) => r.submit_ms,
  inference_ms: (r) => (r.timings && r.timings.inference != null ? r.timings.inference * 1e3 : null),
  denoise_ms: (r) => (r.timings && r.timings.denoise != null ? r.timings.denoise * 1e3 : null),
  video_decode_ms: (r) => (r.timings && r.timings.video_decode != null ? r.timings.video_decode * 1e3 : null),
  encode_ms: (r) => (r.timings && r.timings.encode != null ? r.timings.encode * 1e3 : null),
  engine_total_ms: (r) => (r.timings && r.timings.total != null ? r.timings.total * 1e3 : null),
};
const summary = { label: opt.label, base: opt.base, endpoint: opt.endpoint, input: JSON.parse(opt.input), fresh_prompt: !!opt.fresh, n: runs.length, ab: {}, steps: {}, phases: {} };
for (const [k, f] of Object.entries(keys)) {
  const a = metric(on, f); const b = metric(off, f);
  summary.ab[k] = { traced: stat(a), untraced: stat(b), diff_median: a.length && b.length ? q(a, 0.5) - q(b, 0.5) : null, diff_ci95: bootDiff(a, b) };
}
const stepNames = new Set(on.flatMap((r) => Object.keys(r.steps || {})));
for (const s of stepNames) summary.steps[s] = { ...stat(on.map((r) => (r.steps[s] ? r.steps[s].ms : NaN))), count: q(on.map((r) => (r.steps[s] ? r.steps[s].n : NaN)), 0.5) };
const phaseNames = new Set(on.flatMap((r) => Object.keys(r.phases || {})));
for (const p of phaseNames) summary.phases[p] = stat(on.map((r) => (r.phases[p] ?? NaN)));
summary.unaccounted_ms = stat(on.map((r) => r.unaccounted_ms));
summary.dropped_max = Math.max(0, ...on.map((r) => r.dropped || 0));
summary.offsets = on.length ? on[on.length - 1].offsets : null;
writeFileSync(join(opt.out, 'summary.json'), JSON.stringify(summary, null, 2));

const f = (x) => (x === null || x === undefined ? '—' : x.toFixed(1));
console.log(`\n== ${opt.label} (${runs.length} runs: ${on.length} traced, ${off.length} untraced) ==`);
console.log('A/B (median / p90 ms; diff = traced − untraced, 95 % bootstrap CI)');
for (const [k, v] of Object.entries(summary.ab)) {
  if (!v.traced.n && !v.untraced.n) continue;
  console.log(`  ${k.padEnd(18)} on ${f(v.traced.median)} / ${f(v.traced.p90)}   off ${f(v.untraced.median)} / ${f(v.untraced.p90)}   diff ${f(v.diff_median)} [${v.diff_ci95 ? f(v.diff_ci95.lo) + ', ' + f(v.diff_ci95.hi) : '—'}]`);
}
console.log('critical path (traced runs, median / p90 ms):');
for (const [p, v] of Object.entries(summary.phases)) console.log(`  ${f(v.median).padStart(9)} / ${f(v.p90).padStart(9)}  ${p}`);
console.log(`unaccounted median ${f(summary.unaccounted_ms.median)} ms; dropped events max ${summary.dropped_max}`);
