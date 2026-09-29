// Script avatar page (/console/avatar) in headless Chromium: a photo and a
// script in, the Reactor avatar mode streams speech and video window by
// window over WebRTC.
//
// Two modes:
//   - fake (default): spawns `fv-serve` on the fake engine in avatar mode
//     (FV_REACTOR_MODE=avatar, fake LTX, 256x128) with a synthetic photo, and
//     checks the page end to end: generation_started, every window built,
//     started and streamed in order, generation_complete at the take's
//     length, video decoded and audio received in the <video>, pause and
//     resume, and the timing table filled in.
//   - live: FV_AVATAR_ORIGIN=http://127.0.0.1:8000 drives a real server
//     (an LTX-2.5 pod in avatar mode). FV_AVATAR_IMAGE is the photo; without
//     it a portrait is generated first with a native text-to-video job
//     (FV_KEY). The received stream is recorded (MediaRecorder, WebM) and
//     the events and per-window timings are written to FV_AVATAR_OUT.
//
//   FV_SERVE_BIN=target/debug/fv-serve node tests/console/avatar.cjs
//
// tests/console/run.sh runs the fake mode; scripts/serve/e2e/pod-clients.sh
// `avatar` runs the live mode on a GPU pod.

'use strict';

const { spawn, execFileSync } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const zlib = require('node:zlib');
const { chromium } = require('playwright');

const LIVE = process.env.FV_AVATAR_ORIGIN || '';
const OUT = process.env.FV_AVATAR_OUT || fs.mkdtempSync(path.join(os.tmpdir(), 'fv-avatar-'));
const TIMEOUT_MS = Number(process.env.FV_AVATAR_TIMEOUT_MS || (LIVE ? 900000 : 120000));
const SCRIPT = process.env.FV_AVATAR_SCRIPT || (LIVE
  ? 'Hello, and welcome to the show. Today I want to tell you about streaming video, one window at a time. '
    + 'Every window is generated while the one before it plays, and each one starts from the last frame of the previous one. '
    + 'That keeps the picture steady, and the voice close, from the first sentence to the last. '
    + 'Thank you for watching, and see you next time.'
  : 'Hello and welcome. This is the fake avatar. It speaks in two windows. Goodbye for now.');
const PROMPT = process.env.FV_AVATAR_PROMPT || (LIVE
  ? 'A woman in her thirties speaks warmly and clearly straight to the camera, head and shoulders framed, '
    + 'soft window light, plain light grey wall behind her. Her voice is a calm, friendly mezzo with clear diction.'
  : '');

function step(msg) { process.stdout.write('  - ' + msg + '\n'); }
function fail(msg) { throw new Error(msg); }

// A small PNG (a face-ish gradient) without an image library.
function png(w, h) {
  const crcTable = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
  const crc = (buf) => { let c = -1; for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td));
    return Buffer.concat([len, td, c]);
  };
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const o = y * (w * 3 + 1) + 1 + x * 3;
      const d = Math.hypot(x - w / 2, y - h / 2) / (h / 2);
      raw[o] = d < 0.7 ? 220 : 60; raw[o + 1] = d < 0.7 ? 180 : 70; raw[o + 2] = d < 0.7 ? 150 : 90;
    }
  }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
    s.on('error', reject);
  });
}

async function startFake() {
  const bin = process.env.FV_SERVE_BIN || path.join(__dirname, '..', '..', 'target', 'debug', 'fv-serve');
  const port = await freePort();
  const origin = 'http://127.0.0.1:' + port;
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-avatar-serve-'));
  const cfg = path.join(state, 'fv.toml');
  fs.writeFileSync(cfg, '[engine.fake]\nrtf = 0.3\n\n[reactor]\nmode = "avatar"\nmodel = "fake-ltx-turbo"\navatar_size = "256x128"\nh264 = "off"\n');
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port, FV_STATE_DIR: state, FV_JOB_STORE: 'memory', FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'avatar-console', FV_AUTH_MODE: 'none', RUST_LOG: process.env.RUST_LOG || "warn",
  });
  const srv = { origin, state, logs: '', exited: null };
  srv.proc = spawn(bin, ['--config', cfg], { env, stdio: ['ignore', 'pipe', 'pipe'] });
  srv.proc.stdout.on('data', (d) => { srv.logs += d; });
  srv.proc.stderr.on('data', (d) => { srv.logs += d; });
  srv.proc.on('exit', (code) => { srv.exited = code; });
  for (let i = 0; i < 600; i++) {
    if (srv.exited !== null) fail('fv-serve exited with ' + srv.exited + '\n' + srv.logs.slice(-3000));
    try { const r = await fetch(origin + '/health'); if (r.ok) return srv; } catch { /* not yet */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  fail('fv-serve did not become healthy');
}

async function stopServer(srv) {
  if (!srv) return;
  srv.proc.kill('SIGTERM');
  await new Promise((r) => { if (srv.exited !== null) r(); else { srv.proc.on('exit', r); setTimeout(r, 5000); } });
  fs.rmSync(srv.state, { recursive: true, force: true });
}

// Live mode without FV_AVATAR_IMAGE: a portrait from a short text-to-video
// job on the same server (first frame), so the run needs no photo fixture.
async function portrait(origin) {
  const key = process.env.FV_KEY || '';
  const h = { 'Content-Type': 'application/json', ...(key ? { Authorization: 'Bearer ' + key } : {}) };
  const body = {
    model: process.env.FV_AVATAR_PORTRAIT_MODEL || 'ltx-turbo', size: '1280x768', num_frames: 9, seed: 5,
    prompt: 'A photorealistic portrait photo of a woman in her thirties looking straight into the camera with a slight smile, '
      + 'head and shoulders centred, soft window light, plain light grey wall, sharp focus, still.',
  };
  const t0 = Date.now();
  let r = await fetch(origin + '/fv/v1/jobs', { method: 'POST', headers: h, body: JSON.stringify(body) });
  if (!r.ok) fail('portrait job: ' + r.status + ' ' + (await r.text()));
  const id = (await r.json()).id;
  let s;
  for (;;) {
    s = await (await fetch(origin + '/fv/v1/jobs/' + id, { headers: h })).json();
    if (['succeeded', 'failed', 'cancelled'].includes(s.status)) break;
    if (Date.now() - t0 > 600000) fail('portrait job timed out');
    await new Promise((res) => setTimeout(res, 2000));
  }
  if (s.status !== 'succeeded') fail('portrait job ' + s.status + ': ' + JSON.stringify(s.error));
  const mp4 = path.join(OUT, 'portrait.mp4');
  const v = await fetch(s.output.url.startsWith('http') ? s.output.url : origin + s.output.url, { headers: h });
  fs.writeFileSync(mp4, Buffer.from(await v.arrayBuffer()));
  const img = path.join(OUT, 'portrait.png');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', mp4, '-vf', 'select=eq(n\\,4)', '-frames:v', '1', img]);
  step(`portrait generated in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  return img;
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  let srv = null;
  const origin = LIVE || (srv = await startFake()).origin;
  let image = process.env.FV_AVATAR_IMAGE;
  if (!image && LIVE) image = await portrait(origin);
  if (!image) { image = path.join(OUT, 'face.png'); fs.writeFileSync(image, png(160, 120)); }
  const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
  let ok = false;
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => process.stdout.write('  ! page error: ' + e.message + '\n'));
    const chunks = [];
    await page.exposeFunction('__saveChunk', (b64) => { chunks.push(Buffer.from(b64, 'base64')); });
    await page.goto(origin + '/console/avatar');
    await page.setInputFiles('#image', image);
    await page.fill('#script', SCRIPT);
    await page.fill('#prompt', PROMPT);
    await page.fill('#seed', '42');
    // Fake: a take longer than its one speech window, so idle windows follow.
    await page.fill('#duration', process.env.FV_AVATAR_DURATION || (LIVE ? '0' : '14'));
    step('page loaded; starting the take');
    const tStart = Date.now();
    await page.click('#start');
    // Record what arrives, from the moment the stream exists.
    await page.waitForFunction(() => !!(window.__avatar && window.__avatar.stream && window.__avatar.stream.getTracks().length >= 2), null, { timeout: 60000 });
    await page.evaluate(() => {
      const rec = new MediaRecorder(window.__avatar.stream, { mimeType: 'video/webm;codecs=vp8,opus' });
      rec.ondataavailable = async (ev) => {
        if (!ev.data.size) return;
        const b = new Uint8Array(await ev.data.arrayBuffer());
        let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
        window.__saveChunk(btoa(s));
      };
      rec.start(1000);
      window.__avatar.rec = rec;
    });
    const has = (t) => page.evaluate((t) => window.__avatar.log.some((e) => e.type === t), t);
    try {
      await page.waitForFunction(() => window.__avatar.log.some((e) => ['generation_started', 'command_error'].includes(e.type)), null, { timeout: 120000 });
    } catch (e) {
      fail('no generation_started; page says: ' + (await page.textContent('#take-msg')) + '; log: '
        + JSON.stringify(await page.evaluate(() => window.__avatar.log)) + '\n' + (srv ? srv.logs.slice(-3000) : ''));
    }
    if (!(await has('generation_started'))) {
      fail('no generation_started: ' + JSON.stringify(await page.evaluate(() => window.__avatar.log.filter((e) => e.type !== 'state_update'))));
    }
    step('generation_started');
    if (!LIVE) {
      // Pause and resume once the first window plays.
      await page.waitForFunction(() => window.__avatar.log.some((e) => e.type === 'window_started'), null, { timeout: 60000 });
      await page.click('#pause');
      await page.waitForFunction(() => window.__avatar.log.some((e) => e.type === 'generation_paused'), null, { timeout: 10000 });
      await page.waitForFunction(() => !document.getElementById('resume').disabled, null, { timeout: 10000 });
      await page.click('#resume');
      await page.waitForFunction(() => window.__avatar.log.some((e) => e.type === 'generation_resumed'), null, { timeout: 10000 });
      step('paused and resumed');
    }
    await page.waitForFunction(
      () => window.__avatar.log.some((e) => ['generation_complete', 'generation_failed'].includes(e.type)),
      null, { timeout: TIMEOUT_MS, polling: 500 },
    );
    const wall = (Date.now() - tStart) / 1000;
    // Let the last second reach the recorder, then flush it.
    await new Promise((r) => setTimeout(r, 1500));
    const stats = await page.evaluate(async () => {
      const rec = window.__avatar.rec;
      await new Promise((res) => { rec.onstop = res; rec.stop(); });
      const v = document.getElementById('video');
      return { width: v.videoWidth, height: v.videoHeight, decoded: v.getVideoPlaybackQuality ? v.getVideoPlaybackQuality().totalVideoFrames : null };
    });
    await new Promise((r) => setTimeout(r, 500));
    const result = await page.evaluate(() => ({
      log: window.__avatar.log.filter((e) => e.type !== 'state_update' || e.dir === '→'),
      state: window.__avatar.state,
      windows: window.__avatar.windows,
      table: [...document.querySelectorAll('#windows tr')].map((tr) => [...tr.children].map((td) => td.textContent)),
    }));
    const failed = result.log.find((e) => e.type === 'generation_failed');
    if (failed) fail('generation_failed: ' + JSON.stringify(failed.data));
    const started = result.log.find((e) => e.type === 'generation_started').data;
    const progress = result.log.filter((e) => e.type === 'window_progress').map((e) => e.data);
    const complete = result.log.find((e) => e.type === 'generation_complete').data;
    if (progress.length !== started.total_windows) fail(`${progress.length} window_progress for ${started.total_windows} windows`);
    progress.forEach((p, i) => { if (p.window_index !== i) fail('window order: ' + JSON.stringify(progress)); });
    if (result.table.length !== started.total_windows) fail('timing table rows: ' + JSON.stringify(result.table));
    const effective = result.state.effective_seconds;
    if (Math.abs(complete.seconds_sent - effective) > 0.2) fail(`sent ${complete.seconds_sent} s of ${effective} s`);
    if (!(stats.width > 0)) {
      const rtp = await page.evaluate(async () => {
        const out = [];
        const pc = window.__avatar.pc;
        if (pc) (await pc.getStats()).forEach((r) => { if (['inbound-rtp', 'codec', 'transport'].includes(r.type)) out.push(r); });
        return { out, tracks: window.__avatar.stream.getTracks().map((t) => [t.kind, t.readyState, t.muted]) };
      });
      fail('the <video> shows no picture: ' + JSON.stringify(stats) + ' ' + JSON.stringify(rtp) + '\n' + (srv ? srv.logs.slice(-4000) : ''));
    }
    const webm = path.join(OUT, 'avatar.webm');
    fs.writeFileSync(webm, Buffer.concat(chunks));
    const windows = Object.entries(result.windows).map(([i, w]) => ({
      index: Number(i), kind: w.built && w.built.kind, seconds: w.built && w.built.seconds,
      build_s: w.built && w.built.build_seconds, rtf: w.built && w.built.rtf,
      started_s: w.started && w.started.since_start_seconds, stalled_s: w.started && w.started.stalled_seconds,
    }));
    const summary = {
      origin: LIVE ? 'live' : 'fake', total_windows: started.total_windows, take_seconds: effective,
      seconds_sent: complete.seconds_sent, first_frame_s: result.state.first_frame_seconds,
      stalls: result.state.stalls, stalled_s: result.state.stalled_seconds, wall_s: wall,
      video: stats, recording_bytes: fs.statSync(webm).size, windows,
    };
    fs.writeFileSync(path.join(OUT, 'events.json'), JSON.stringify(result, null, 1));
    fs.writeFileSync(path.join(OUT, 'summary.json'), JSON.stringify(summary, null, 1));
    step(`complete: ${summary.total_windows} windows, ${summary.seconds_sent} s sent, first frame after ${summary.first_frame_s} s, `
      + `${summary.stalls} stalls (${summary.stalled_s} s), wall ${wall.toFixed(1)} s`);
    for (const w of windows) step(`window ${w.index} ${w.kind}: ${w.seconds} s built in ${w.build_s} s (rtf ${w.rtf}), started at ${w.started_s} s, waited ${w.stalled_s} s`);
    process.stdout.write('SUMMARY ' + JSON.stringify(summary) + '\n');
    await page.click('#end');
    ok = true;
  } finally {
    await browser.close();
    await stopServer(srv);
    if (!LIVE && ok) fs.rmSync(OUT, { recursive: true, force: true });
  }
}

main().then(() => { process.stdout.write('avatar console: PASS\n'); }, (e) => {
  process.stdout.write('avatar console: FAIL ' + (e && e.stack || e) + '\n');
  process.exit(1);
});
