// Live input in headless Chromium with a fake camera and microphone
// (--use-fake-device-for-media-stream): the console's Live input page
// publishes to fv-serve's loopback echo model (FV_ECHO_MODEL=1) and the
// echo comes back with its overlay, over both transports:
//   - native WHIP ingest (POST /fv/v1/streams/ingest, send-receive peer);
//   - the Reactor runtime (PublishTrack input_video / input_audio).
// Checks, per transport: the page reaches "streaming"; the model shows
// input frames (the stream status / state_update counters); the <video>
// shows the magenta overlay border around a picture whose centre matches
// the fake camera's centre; the overlay's frame counter advances; Stop
// ends the session. The server requires an API key (duplex sessions are
// authenticated); the page reads it from local storage.
//
//   FV_SERVE_BIN=target/debug/fv-serve node tests/console/live_echo.cjs
//
// tests/console/run.sh runs it after smoke.cjs (same Playwright/Chromium).

'use strict';

const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');

const BIN = process.env.FV_SERVE_BIN || path.join(__dirname, '..', '..', 'target', 'debug', 'fv-serve');
const KEY = 'fv_live_echo_test_key';

function step(msg) { process.stdout.write('  - ' + msg + '\n'); }

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
    s.on('error', reject);
  });
}

async function startServer() {
  const port = await freePort();
  const origin = 'http://127.0.0.1:' + port;
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-live-echo-'));
  const cfg = path.join(state, 'fv.toml');
  fs.writeFileSync(cfg, '[webrtc]\npublic_ip = "127.0.0.1"\n');
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port, FV_STATE_DIR: state, FV_JOB_STORE: 'memory', FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'live-echo', FV_API_KEYS: crypto.createHash('sha256').update(KEY).digest('hex'),
    FV_ECHO_MODEL: '1', FV_REACTOR_MODEL: 'fv-echo', RUST_LOG: 'warn,fastvideo_webrtc::ingest=info',
  });
  const srv = { origin, state, logs: '', exited: null };
  srv.proc = spawn(BIN, ['--config', cfg], { env, stdio: ['ignore', 'pipe', 'pipe'] });
  srv.proc.stdout.on('data', (d) => { srv.logs += d; });
  srv.proc.stderr.on('data', (d) => { srv.logs += d; });
  srv.proc.on('exit', (code) => { srv.exited = code; });
  for (let i = 0; i < 600; i++) {
    if (srv.exited !== null) throw new Error('fv-serve exited with ' + srv.exited + '\n' + srv.logs.slice(-3000));
    try { const r = await fetch(origin + '/health'); if (r.ok) return srv; } catch { /* not yet */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('fv-serve did not become healthy');
}

async function stopServer(srv) {
  srv.proc.kill('SIGTERM');
  await new Promise((r) => { if (srv.exited !== null) r(); else { srv.proc.on('exit', r); setTimeout(r, 5000); } });
  fs.rmSync(srv.state, { recursive: true, force: true });
}

// What the two <video>s show: sizes, the remote border pixel, the mean
// colour of both centres (a large square: the fake camera's moving pie
// must not dominate), and the overlay's 32-cell frame counter.
async function sample(page) {
  return page.evaluate(() => {
    const grab = (v) => {
      if (!v || v.videoWidth < 64) return null;
      const c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      const g = c.getContext('2d');
      g.drawImage(v, 0, 0);
      return { c, g, w: c.width, h: c.height };
    };
    const mean = (f, cx, cy, r) => {
      const d = f.g.getImageData(cx - r, cy - r, 2 * r, 2 * r).data;
      const s = [0, 0, 0];
      for (let i = 0; i < d.length; i += 4) { s[0] += d[i]; s[1] += d[i + 1]; s[2] += d[i + 2]; }
      const n = d.length / 4;
      return s.map((x) => Math.round(x / n));
    };
    const o = {};
    const r = grab(document.querySelector('#remote'));
    const l = grab(document.querySelector('#local'));
    if (r) {
      const b = Math.max(4, Math.floor(r.h / 40)) & ~1;
      const px = r.g.getImageData(Math.floor(b / 2), Math.floor(r.h / 2), 1, 1).data;
      o.remote = { w: r.w, h: r.h, border: [px[0], px[1], px[2]], centre: mean(r, Math.floor(r.w / 2), Math.floor(r.h / 2), Math.floor(r.h / 4)) };
      const inner = r.w - 2 * b;
      let idx = 0;
      for (let bit = 0; bit < 32; bit++) {
        const x = b + Math.floor(bit * inner / 32 + inner / 64);
        const p = r.g.getImageData(x, b + b, 1, 1).data;
        idx = idx * 2 + ((p[0] + p[1] + p[2]) / 3 > 127 ? 1 : 0);
      }
      o.remote.index = idx;
    }
    if (l) o.local = { w: l.w, h: l.h, centre: mean(l, Math.floor(l.w / 2), Math.floor(l.h / 2), Math.floor(l.h / 4)) };
    o.state = document.body.dataset.liveState;
    o.inputFrames = Number(document.body.dataset.inputFrames || 0);
    return o;
  });
}

function check(cond, what, s) {
  if (!cond) throw new Error('live echo: ' + what + (s ? ' :: ' + JSON.stringify(s) : ''));
}

const close = (a, b, tol) => a.every((x, i) => Math.abs(x - b[i]) <= tol);

async function run(page, transport) {
  await page.selectOption('#transport', transport);
  await page.click('#start');
  await page.waitForFunction(() => document.body.dataset.liveState === 'streaming', null, { timeout: 30_000 });
  // The model has shown the camera (counters from the stream status or state_update).
  await page.waitForFunction(() => Number(document.body.dataset.inputFrames || 0) >= 20, null, { timeout: 60_000 });
  await page.waitForFunction(() => document.querySelector('#remote').videoWidth > 0, null, { timeout: 10_000 });
  await new Promise((r) => setTimeout(r, 1500));
  const a = await sample(page);
  await new Promise((r) => setTimeout(r, 1000));
  const b = await sample(page);
  check(b.remote && b.local, 'both videos play', b);
  check(b.remote.w === 640 && b.remote.h === 360, 'the echo is 640x360', b);
  check(close(b.remote.border, [255, 0, 255], 70), 'the overlay border is magenta', b);
  check(close(b.remote.centre, b.local.centre, 60), 'the echo shows the camera (centre colours match)', b);
  check(b.remote.centre[1] > b.remote.centre[0] + 40, 'the echo is the fake camera\'s green, not the grey waiting card', b);
  check(b.remote.index > a.remote.index, 'the overlay counter advances', { a, b });
  step(transport + ': streaming, ' + b.inputFrames + ' input frames shown; border ' + JSON.stringify(b.remote.border)
    + ', centre ' + JSON.stringify(b.remote.centre) + ' vs camera ' + JSON.stringify(b.local.centre) + ', counter ' + a.remote.index + ' -> ' + b.remote.index);
  await page.click('#stop');
  await page.waitForFunction(() => document.body.dataset.liveState === 'stopped', null, { timeout: 30_000 });
  step(transport + ': stopped');
}

(async () => {
  const srv = await startServer();
  const browser = await chromium.launch({
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  let page;
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ['camera', 'microphone'] });
    page = await context.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.addInitScript((key) => { try { localStorage.setItem('fv.key', key); } catch { /* no storage */ } }, KEY);
    // Without a key, duplex ingest is refused.
    const refused = await fetch(srv.origin + '/fv/v1/streams/ingest?model=fv-echo', { method: 'POST', headers: { 'Content-Type': 'application/sdp' }, body: 'v=0' });
    check(refused.status === 401, 'WHIP ingest without a key is refused (' + refused.status + ')');
    const r2 = await fetch(srv.origin + '/start_session', { method: 'POST' });
    check(r2.status === 401, 'a Reactor duplex session without a key is refused (' + r2.status + ')');
    step('no key: WHIP ingest and Reactor start_session answer 401');

    await page.goto(srv.origin + '/console/live');
    await page.waitForFunction(() => [...document.querySelectorAll('#model option')].some((o) => o.value === 'fv-echo'), null, { timeout: 30_000 });
    const facts = await page.textContent('#model-facts');
    check(/vp8\/h264 up to 1280×720/.test(facts), 'the model facts show the input caps', facts);
    step('Live input page lists fv-echo: ' + facts);
    await page.selectOption('#res', '640x360');
    await page.fill('#scene', 'a test studio');
    await run(page, 'whip');
    await run(page, 'reactor');
    if (errors.length) throw new Error('browser errors:\n' + errors.join('\n'));
    console.log('live echo: OK');
  } catch (e) {
    if (page) {
      try {
        const log = await page.evaluate(() => [...document.querySelectorAll('#live-log div')].map((d) => d.textContent).join('\n')
          + '\n' + document.querySelector('#live-msg').textContent + '\n' + document.querySelector('#live-stats').textContent);
        process.stderr.write('--- page events ---\n' + log.slice(-4000) + '\n');
      } catch { /* page gone */ }
    }
    process.stderr.write('--- fv-serve log ---\n' + srv.logs.slice(-4000) + '\n');
    throw e;
  } finally {
    await browser.close();
    await stopServer(srv);
  }
})().catch((e) => { console.error(e); process.exit(1); });
