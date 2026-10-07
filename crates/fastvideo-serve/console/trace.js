// Request tracing in the console and the bench (docs/serve/tracing.md).
//
// Two halves, both without side effects at import (the CLI tools
// scripts/serve/trace-bench.mjs and trace-report.mjs import this module in
// Node):
//
// - `ClientTrace`: one trace from the click. Marks are `performance.now()`
//   reads pushed into an array; nothing is sent, formatted or rendered
//   until the video is playable, and then only in an idle callback
//   (`navigator.sendBeacon`). Every request of the trace carries
//   `traceparent` + `x-fv-trace: 1`; each answer's `x-fv-trace-t` (the
//   pod's receive/send ns) and `x-fv-edge-t` (the edge's, ms) become
//   NTP-style clock samples.
// - `analyze(events)`: aligns the hosts' clocks (minimum-delay sample per
//   host, ± half its delay), builds the waterfall, the critical-path phases
//   and the unaccounted gaps.

const hex = (n) => {
  const b = new Uint8Array(n / 2);
  globalThis.crypto.getRandomValues(b);
  return [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
};

const nowMs = () => globalThis.performance.timeOrigin + globalThis.performance.now();
const ns = (ms) => Math.round(ms * 1e6);

// Whether the console traces its runs (the Trace switch, `?trace=1`).
export const TRACE_KEY = 'fv.trace';

export class ClientTrace {
  constructor({ host = 'client' } = {}) {
    this.id = hex(32);
    this.span = hex(16);
    this.host = host;
    this.events = [];
    this.marks = {};
  }

  // Headers for every request of this trace.
  headers() {
    return { traceparent: '00-' + this.id + '-' + this.span + '-01', 'x-fv-trace': '1' };
  }

  // A point now (or at `atMs`, a wall-clock ms).
  point(name, attrs, atMs) {
    const t = atMs ?? nowMs();
    this.marks[name] = t;
    this.events.push({ trace: this.id, host: this.host, comp: 'client', name, clock: 'host', t_wall_ns: ns(t), dur_ns: 0, attrs });
    return t;
  }

  // A span between two wall-clock ms.
  spanMs(name, startMs, endMs, attrs) {
    this.events.push({ trace: this.id, host: this.host, comp: 'client', name, clock: 'host', t_wall_ns: ns(startMs), dur_ns: Math.max(0, ns(endMs - startMs)), attrs });
  }

  now() { return nowMs(); }

  // One HTTP exchange: `t0` before fetch, `tHead` when the headers arrived,
  // `tEnd` when the body was read (wall ms). `headers` a Headers-like
  // object; `body` the parsed answer (its `status` is kept: fal's
  // IN_QUEUE / IN_PROGRESS / COMPLETED).
  http(name, t0, tHead, tEnd, status, headers, body) {
    const attrs = { status };
    if (body && typeof body === 'object' && typeof body.status === 'string') attrs.job_status = body.status;
    const get = (k) => (headers && typeof headers.get === 'function' ? headers.get(k) : null);
    const pod = get('x-fv-trace-t');
    const edge = get('x-fv-edge-t');
    const sync = [];
    if (pod && pod.includes(';')) {
      const [t1, t2] = pod.split(';').map(Number);
      if (Number.isFinite(t1) && Number.isFinite(t2)) sync.push({ peer: 'pod', t0: ns(t0), t1, t2, t3: ns(tHead) });
    }
    if (edge && edge.includes(';')) {
      const [a, b] = edge.split(';').map(Number);
      if (Number.isFinite(a) && Number.isFinite(b)) sync.push({ peer: 'edge', t0: ns(t0), t1: a * 1e6, t2: b * 1e6, t3: ns(tHead) });
    }
    if (sync.length) attrs.sync = sync;
    this.spanMs(name, t0, tHead, attrs);
    if (tEnd > tHead) this.spanMs(name + '.body', tHead, tEnd);
  }

  // Resource Timing of `url` (the video): request start, first byte, last byte.
  resource(url) {
    const perf = globalThis.performance;
    if (!perf || typeof perf.getEntriesByName !== 'function') return;
    const list = perf.getEntriesByName(url);
    const e = list[list.length - 1];
    if (!e) return;
    const o = perf.timeOrigin;
    // Cross-origin without Timing-Allow-Origin: only start and end are known.
    const first = e.responseStart > 0 ? e.responseStart : null;
    this.spanMs('video_fetch', o + e.startTime, o + e.responseEnd, { bytes: e.transferSize || e.encodedBodySize || null, ttfb_known: first !== null });
    if (first !== null) {
      this.spanMs('video_ttfb', o + e.startTime, o + first);
      this.spanMs('video_bytes', o + first, o + e.responseEnd);
    }
  }
}

// ---- analysis -----------------------------------------------------------------

// Spans that contain other steps: drawn, but not counted as "accounted".
const ENVELOPES = new Set(['client:e2e', 'engine:run', 'edge:request', 'store:terminal', 'client:wait_poll']);

// NTP estimate from (t0, t1, t2, t3): offset = remote − local, delay = round trip − remote time.
function estimate(s) {
  const offset = ((s.t1 - s.t0) + (s.t2 - s.t3)) / 2;
  const delay = (s.t3 - s.t0) - (s.t2 - s.t1);
  return { offset, delay: Math.max(0, delay) };
}

function best(samples) {
  let b = null;
  for (const s of samples) {
    const e = estimate(s);
    if (!b || e.delay < b.delay) b = { ...e, n: samples.length };
  }
  return b;
}

// events: every host's events of one trace. Returns {offsets, rows, phases, gaps, totals}.
export function analyze(events, { reference } = {}) {
  const hosts = [...new Set(events.map((e) => e.host || 'unknown'))];
  const podHost = hosts.find((h) => h.startsWith('pod:')) || null;
  const clientSamples = { pod: [], edge: [] };
  const edgePod = [];
  for (const e of events) {
    const sync = e.attrs && e.attrs.sync;
    if (!sync) continue;
    for (const s of Array.isArray(sync) ? sync : [sync]) {
      if (!s || !Number.isFinite(s.t0)) continue;
      if (e.host === 'edge' && s.peer === 'front') edgePod.push(s);
      else if (s.peer === 'pod') clientSamples.pod.push(s);
      else if (s.peer === 'edge') clientSamples.edge.push(s);
    }
  }
  // Offsets of each host relative to the reference clock (the client's when present).
  const ref = reference || (hosts.includes('client') ? 'client' : podHost || hosts[0]);
  const off = { [ref]: { offset: 0, uncert: 0, via: 'reference' } };
  const cp = best(clientSamples.pod);
  const ce = best(clientSamples.edge);
  const ep = best(edgePod);
  if (ref === 'client') {
    if (ce) off.edge = { offset: ce.offset, uncert: ce.delay / 2, via: 'client↔edge, ' + ce.n + ' samples' };
    if (podHost && cp) off[podHost] = { offset: cp.offset, uncert: cp.delay / 2, via: 'client↔pod, ' + cp.n + ' samples' };
    else if (podHost && ce && ep) off[podHost] = { offset: ce.offset + ep.offset, uncert: (ce.delay + ep.delay) / 2, via: 'client↔edge↔pod' };
  } else if (ref === podHost && ep) {
    off.edge = { offset: -ep.offset, uncert: ep.delay / 2, via: 'edge↔pod, ' + ep.n + ' samples' };
  }
  const rows = [];
  for (const e of events) {
    const o = off[e.host];
    if (!o) continue; // a host we cannot align is left out of the waterfall
    const t = e.t_wall_ns - o.offset;
    rows.push({ start: t, end: t + (e.dur_ns || 0), dur: e.dur_ns || 0, comp: e.comp, name: e.name, host: e.host, clock: e.clock || 'host', arg: e.arg, attrs: e.attrs, uncert: o.uncert });
  }
  const click = rows.find((r) => r.comp === 'client' && r.name === 'click');
  const t0 = click ? click.start : Math.min(...rows.map((r) => r.start));
  for (const r of rows) { r.startMs = (r.start - t0) / 1e6; r.durMs = r.dur / 1e6; r.endMs = r.startMs + r.durMs; r.uncertMs = r.uncert / 1e6; }
  rows.sort((a, b) => a.start - b.start || b.dur - a.dur);
  const endRow = rows.find((r) => r.comp === 'client' && r.name === 'canplay') || rows.find((r) => r.comp === 'client' && r.name === 'video_fetch');
  const endMs = endRow ? endRow.endMs : Math.max(...rows.map((r) => r.endMs));

  // Unaccounted: [0, end] not covered by any non-envelope step.
  const iv = rows.filter((r) => r.dur > 0 && !ENVELOPES.has(r.comp + ':' + r.name)).map((r) => [Math.max(0, r.startMs), Math.min(endMs, r.endMs)]).filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
  const gaps = [];
  let cur = 0;
  for (const [a, b] of iv) {
    if (a > cur + 0.05) gaps.push({ startMs: cur, endMs: a, durMs: a - cur });
    cur = Math.max(cur, b);
  }
  if (endMs > cur + 0.05) gaps.push({ startMs: cur, endMs, durMs: endMs - cur });
  for (const g of gaps) {
    const before = rows.filter((r) => r.endMs <= g.startMs + 0.01 && r.dur > 0).sort((a, b) => b.endMs - a.endMs)[0];
    const after = rows.find((r) => r.startMs >= g.endMs - 0.01);
    g.after = before ? before.comp + '.' + before.name : 'start';
    g.before = after ? after.comp + '.' + after.name : 'end';
  }
  const unaccounted = gaps.reduce((s, g) => s + g.durMs, 0);
  return { reference: ref, offsets: off, rows, gaps, unaccountedMs: unaccounted, totalMs: endMs, phases: phases(rows, endMs) };
}

// The critical path as consecutive phases between anchor events (ms from the click).
export function phases(rows, endMs) {
  const first = (pred) => rows.find(pred);
  const last = (pred) => [...rows].reverse().find(pred);
  const is = (c, n) => (r) => r.comp === c && r.name === n;
  const submit = first(is('client', 'submit'));
  const podPost = first((r) => r.comp === 'http' && r.name === 'post');
  const qwait = first(is('queue', 'wait'));
  const run = first(is('engine', 'run'));
  const gpuFirst = first((r) => r.comp === 'gpu');
  const gpuLast = last((r) => r.comp === 'gpu');
  const terminal = first(is('store', 'terminal'));
  const done = terminal ? terminal.endMs : null;
  const seen = first((r) => r.comp === 'client' && r.name === 'poll' && r.attrs && r.attrs.job_status === 'COMPLETED');
  const result = first(is('client', 'result'));
  const vf = first(is('client', 'video_fetch'));
  const vttfb = first(is('client', 'video_ttfb'));
  const canplay = first(is('client', 'canplay'));
  const a = [
    ['click', 0],
    ['submit sent', submit && submit.startMs],
    ['pod received', podPost && podPost.startMs],
    ['pod answered (queued)', podPost && podPost.endMs],
    ['submit answer at client', submit && submit.endMs],
    ['engine dequeued', qwait && qwait.endMs],
    ['GPU first mark', gpuFirst && gpuFirst.startMs],
    ['GPU last mark', gpuLast && gpuLast.endMs],
    ['engine run end', run && run.endMs],
    ['job terminal (stored)', done],
    ['client saw COMPLETED', seen && seen.endMs],
    ['result answer at client', result && result.endMs],
    ['video first byte', vttfb ? vttfb.endMs : null],
    ['video last byte', vf && vf.endMs],
    ['video playable', canplay ? canplay.startMs : endMs],
  ].filter(([, t]) => t !== null && t !== undefined && Number.isFinite(t));
  const out = [];
  for (let i = 1; i < a.length; i++) out.push({ from: a[i - 1][0], to: a[i][0], ms: a[i][1] - a[i - 1][1], atMs: a[i][1] });
  return out;
}

// Plain-text waterfall (the CLI report).
export function textReport(res, { width = 60 } = {}) {
  const lines = [];
  const scale = res.totalMs > 0 ? width / res.totalMs : 0;
  lines.push('reference clock: ' + res.reference + '; total ' + res.totalMs.toFixed(1) + ' ms; unaccounted ' + res.unaccountedMs.toFixed(1) + ' ms');
  for (const [h, o] of Object.entries(res.offsets)) lines.push('  ' + h + ': offset ' + (o.offset / 1e6).toFixed(2) + ' ms ± ' + (o.uncert / 1e6).toFixed(2) + ' ms (' + o.via + ')');
  lines.push('');
  lines.push('start ms   dur ms     step');
  for (const r of res.rows) {
    const bar = ' '.repeat(Math.max(0, Math.floor(r.startMs * scale))) + (r.dur > 0 ? '█'.repeat(Math.max(1, Math.round(r.durMs * scale))) : '|');
    lines.push(r.startMs.toFixed(1).padStart(9) + ' ' + r.durMs.toFixed(2).padStart(9) + '  ' + (r.comp + '.' + r.name + (r.arg != null ? '[' + r.arg + ']' : '') + (r.clock === 'gpu' ? ' (gpu)' : '')).padEnd(34) + ' ' + bar);
  }
  lines.push('');
  lines.push('critical path:');
  for (const p of res.phases) lines.push('  ' + p.ms.toFixed(1).padStart(9) + ' ms  ' + p.from + ' → ' + p.to);
  lines.push('');
  lines.push('unaccounted gaps (> 0.05 ms):');
  for (const g of res.gaps.filter((x) => x.durMs >= 1).sort((a, b) => b.durMs - a.durMs).slice(0, 12)) lines.push('  ' + g.durMs.toFixed(1).padStart(9) + ' ms at ' + g.startMs.toFixed(1) + ' (after ' + g.after + ', before ' + g.before + ')');
  return lines.join('\n');
}
