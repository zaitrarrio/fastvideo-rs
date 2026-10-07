// node --test scripts/serve/tests/trace-analyze.test.mjs
// The waterfall analysis of console/trace.js: clock alignment from
// NTP-style samples, the critical path and the unaccounted gaps.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const { analyze, textReport, ClientTrace } = await import(join(HERE, '../../../crates/fastvideo-serve/console/trace.js'));

const MS = 1e6;
// The pod's clock runs 5 000 ms ahead of the client's; one-way latency 20 ms.
const SKEW = 5000 * MS;
const T = 1_800_000_000_000 * MS;
const ev = (host, comp, name, startMs, durMs, attrs) => ({
  trace: 'a'.repeat(32), host, comp, name, clock: comp === 'gpu' ? 'gpu' : 'host',
  t_wall_ns: T + startMs * MS + (host.startsWith('pod') ? SKEW : 0), dur_ns: durMs * MS, attrs,
});

function sample() {
  // Client sends at 0, pod receives at 20 and answers at 30 (pod clock + skew), client sees it at 50.
  return { peer: 'pod', t0: T, t1: T + 20 * MS + SKEW, t2: T + 30 * MS + SKEW, t3: T + 50 * MS };
}

test('pod events are aligned to the client clock within the sample uncertainty', () => {
  const events = [
    ev('client', 'client', 'click', 0, 0),
    ev('client', 'client', 'submit', 0, 50, { sync: [sample()] }),
    ev('pod:x', 'http', 'post', 20, 10),
    ev('pod:x', 'queue', 'wait', 30, 5),
    ev('pod:x', 'gpu', 'denoise.step', 35, 100),
    ev('pod:x', 'store', 'terminal', 135, 15),
    ev('pod:x', 'store', 'terminal_write', 135, 15),
    ev('client', 'client', 'poll', 700, 40, { job_status: 'COMPLETED' }),
    ev('client', 'client', 'canplay', 800, 0),
  ];
  const r = analyze(events);
  assert.equal(r.reference, 'client');
  assert.ok(Math.abs(r.offsets['pod:x'].offset - SKEW) < 1e3, 'offset recovered');
  // ns wall times near 1.8e18 are doubles in JS: 256 ns resolution.
  assert.ok(Math.abs(r.offsets['pod:x'].uncert - 20 * MS) < 1e3, '± half of (rtt − server time)');
  const post = r.rows.find((x) => x.name === 'post');
  assert.ok(Math.abs(post.startMs - 20) < 1e-3);
  // Gap between the store write (ends at 150) and the poll (700) is unaccounted.
  const g = r.gaps.find((x) => x.startMs > 140 && x.startMs < 160);
  assert.ok(g && Math.abs(g.durMs - 550) < 1e-3, JSON.stringify(r.gaps));
  assert.ok(r.phases.some((p) => p.to === 'client saw COMPLETED'));
  assert.match(textReport(r), /critical path/);
});

test('a client trace records spans and samples from headers', () => {
  const t = new ClientTrace();
  t.point('click');
  const h = new Map([['x-fv-trace-t', '100;200'], ['x-fv-edge-t', '1;2']]);
  t.http('submit', 1000, 1010, 1012, 200, { get: (k) => h.get(k) ?? null }, { status: 'IN_QUEUE' });
  const s = t.events.find((e) => e.name === 'submit');
  assert.equal(s.attrs.job_status, 'IN_QUEUE');
  assert.equal(s.attrs.sync.length, 2);
  assert.equal(t.headers()['x-fv-trace'], '1');
  assert.match(t.headers().traceparent, /^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);
});
