// Director playback in headless Chromium when generation is slower than
// real time (the console's director page against fv-serve's fake engine).
//
// The fake engine builds each 5 s chunk in FV_DIRECTOR_PLAYBACK_RTF x 5 s
// (default 3: 15 s), the way H3 takes ~22 s per 10 s chunk at 480p and over
// a minute at 768p. Checks, with no prompt sent:
//   - before chunk 0 is built the page says it is generating chunk 0 and
//     the video element has no picture (the stream carries silence only);
//   - chunk 0 shows as soon as it is built (burned-in frame index advancing
//     in the <video>), with no prompt or other client event;
//   - after chunk 0's 5 s the page says it waits for chunk 1 while the last
//     frame holds;
//   - chunk 1 then plays (the frame index restarts), with the decoder
//     keeping up (framesDecoded at ~24 fps, keyframes decoded);
// then a prompt changes nothing about playout, and stop clears the status.
//
//   FV_SERVE_BIN=target/debug/fv-serve node tests/console/director_playback.cjs
//
// tests/console/run.sh runs it after smoke.cjs (same Playwright/Chromium).

'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');

const BIN = process.env.FV_SERVE_BIN || path.join(__dirname, '..', '..', 'target', 'debug', 'fv-serve');
const RTF = Number(process.env.FV_DIRECTOR_PLAYBACK_RTF || 3);
const CHUNK_S = 5;
const GEN_MS = RTF * CHUNK_S * 1000;

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
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-director-playback-'));
  const cfg = path.join(state, 'fv.toml');
  fs.writeFileSync(cfg, `[engine.fake]\nrtf = ${RTF}\n\n[director]\nchunk_seconds = ${CHUNK_S}.0\n`);
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port, FV_STATE_DIR: state, FV_JOB_STORE: 'memory', FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'director-playback', FV_AUTH_MODE: 'none', RUST_LOG: 'warn',
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

// What the <video> shows: the fake engine's burned-in frame index (32
// black/white cells across the top, one cell = width/32 px), plus the
// inbound video stats.
async function sample(page) {
  return page.evaluate(async () => {
    const v = document.querySelector('#director-video');
    const o = { w: v.videoWidth, h: v.videoHeight, index: null };
    if (v.videoWidth >= 32 && v.videoHeight >= 32) {
      const c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      const g = c.getContext('2d');
      g.drawImage(v, 0, 0);
      const cw = Math.floor(c.width / 32);
      const d = g.getImageData(0, 0, c.width, cw).data;
      let idx = 0;
      for (let b = 0; b < 32; b++) {
        const i = (Math.floor(cw / 2) * c.width + b * cw + Math.floor(cw / 2)) * 4;
        idx = idx * 2 + (d[i] > 127 ? 1 : 0);
      }
      o.index = idx;
    }
    const pc = window.__pcs[window.__pcs.length - 1];
    if (pc) {
      (await pc.getStats()).forEach((x) => {
        if (x.type === 'inbound-rtp' && x.kind === 'video') {
          o.framesDecoded = x.framesDecoded; o.keyFramesDecoded = x.keyFramesDecoded; o.pliCount = x.pliCount;
        }
      });
    }
    const p = document.querySelector('#director-playback');
    o.phase = p.dataset.phase; o.chunk = p.dataset.chunk; o.text = p.textContent;
    return o;
  });
}

async function waitPhase(page, phase, chunk, timeout) {
  await page.waitForFunction(([ph, ch]) => {
    const p = document.querySelector('#director-playback');
    return p && p.dataset.phase === ph && (ch == null || p.dataset.chunk === String(ch));
  }, [phase, chunk], { timeout });
}

function check(cond, what, s) {
  if (!cond) throw new Error('director playback: ' + what + (s ? ' :: ' + JSON.stringify(s) : ''));
}

(async () => {
  const srv = await startServer();
  const browser = await chromium.launch();
  let page;
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    page = await context.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.addInitScript(() => {
      const Native = window.RTCPeerConnection;
      window.__pcs = [];
      window.RTCPeerConnection = function (...args) { const pc = new Native(...args); window.__pcs.push(pc); return pc; };
      window.RTCPeerConnection.prototype = Native.prototype;
      Object.setPrototypeOf(window.RTCPeerConnection, Native);
    });
    await page.goto(srv.origin + '/console/models/minimax/h3-turbo/director');
    await page.waitForSelector('#director-start');
    await page.click('#director-start');
    const t0 = Date.now();

    // Before chunk 0: generating, and no picture.
    await waitPhase(page, 'generating', 0, 30_000);
    await new Promise((r) => setTimeout(r, Math.min(4000, GEN_MS / 3)));
    let s = await sample(page);
    check(s.phase === 'generating' && /Generating chunk 0/.test(s.text), 'status while chunk 0 generates', s);
    check(s.w === 0 && !s.framesDecoded, 'no picture before chunk 0 is built', s);
    step('before chunk 0: "' + s.text + '", no video frames');

    // Chunk 0 shows once built, with no prompt sent.
    await waitPhase(page, 'playing', 0, GEN_MS + 60_000);
    const shownAt = Date.now() - t0;
    await page.waitForFunction(() => document.querySelector('#director-video').videoWidth > 0, null, { timeout: 5000 });
    await new Promise((r) => setTimeout(r, 800));
    const a = await sample(page);
    await new Promise((r) => setTimeout(r, 1000));
    const b = await sample(page);
    const sent = await page.evaluate(() => [...document.querySelectorAll('#director-log div')].filter((d) => d.textContent.includes('"type":"→ prompt"')).length);
    check(sent === 0, 'no prompt was sent before chunk 0 showed');
    check(a.index != null && b.index > a.index && b.index - a.index >= 15, 'chunk 0 frames advance in the <video>', { a, b });
    check(b.keyFramesDecoded >= 1, 'a keyframe was decoded', b);
    step('chunk 0 plays ' + (shownAt / 1000).toFixed(1) + ' s after Start with no prompt (frame ' + a.index + ' -> ' + b.index + ')');

    // After its 5 s: waiting for chunk 1, the last frame holding.
    await waitPhase(page, 'waiting', 1, 15_000);
    await new Promise((r) => setTimeout(r, 1200));
    const h0 = await sample(page);
    await new Promise((r) => setTimeout(r, 1000));
    const h1 = await sample(page);
    check(/Waiting for chunk 1/.test(h1.text), 'status while chunk 1 is late', h1);
    check(h0.index === h1.index && h1.index >= CHUNK_S * 24 - 10, 'the last frame of chunk 0 holds', { h0, h1 });
    step('after chunk 0: "' + h1.text + '", frame ' + h1.index + ' held');

    // Chunk 1 plays when it is ready: the frame index restarts and advances.
    await waitPhase(page, 'playing', 1, GEN_MS + 60_000);
    await new Promise((r) => setTimeout(r, 1500));
    const c0 = await sample(page);
    await new Promise((r) => setTimeout(r, 1000));
    const c1 = await sample(page);
    check(c0.index < h1.index && c1.index > c0.index, 'chunk 1 frames replace the held frame and advance', { h1, c0, c1 });
    const fps = (c1.framesDecoded - c0.framesDecoded) / 1.0;
    check(fps > 18, 'the decoder keeps up with chunk 1 (' + fps + ' frames/s)', { c0, c1 });
    step('chunk 1 plays (frame ' + c0.index + ' -> ' + c1.index + ', ' + fps + ' frames decoded/s, ' + c1.keyFramesDecoded + ' keyframes, ' + c1.pliCount + ' PLIs)');

    // A prompt is only direction for a later chunk: playout goes on.
    await page.fill('#director-next', 'They reach the lamp room.');
    await page.click('#director-send');
    await page.waitForFunction(
      () => [...document.querySelectorAll('#director-timeline li')].some((li) => li.textContent.startsWith('v2') && /pending|applied/.test(li.lastChild.textContent)),
      null, { timeout: 10_000 });
    const p0 = await sample(page);
    await new Promise((r) => setTimeout(r, 600));
    const p1 = await sample(page);
    check(p0.phase === 'playing' || p0.phase === 'waiting', 'status after a prompt', p0);
    if (p0.phase === 'playing' && p1.phase === 'playing') check(p1.index > p0.index, 'playout continues after a prompt', { p0, p1 });
    step('prompt v2 queued; playout unaffected');

    await page.click('#director-stop');
    await page.waitForFunction(() => document.querySelector('#director-state').textContent === 'closed', null, { timeout: 30_000 });
    const end = await sample(page);
    check(end.phase === 'idle' && end.text === '', 'status cleared after stop', end);
    step('stopped: status cleared');
    if (errors.length) throw new Error('browser errors:\n' + errors.join('\n'));
    console.log('director playback: OK');
  } catch (e) {
    if (page) {
      try {
        const log = await page.evaluate(() => [...document.querySelectorAll('#director-log div')].map((d) => d.textContent.slice(0, 300)).join('\n'));
        process.stderr.write('--- director events ---\n' + log.slice(-4000) + '\n');
      } catch { /* page gone */ }
    }
    process.stderr.write('--- fv-serve log ---\n' + srv.logs.slice(-4000) + '\n');
    throw e;
  } finally {
    await browser.close();
    await stopServer(srv);
  }
})().catch((e) => { console.error(e); process.exit(1); });
