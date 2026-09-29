// Headless browser smoke test of the fv-serve console (WP-20).
//
// Starts `fv-serve` (built with `--features fake`) on a free port with no
// FV_ADMIN_TOKEN, reads the generated admin token from its WARN banner, and
// drives Chromium through: mint an API key on /console/admin, run
// text-to-video, upload an image and run image-to-video, see video results,
// the API snippets, the history, and a live director session over WebRTC
// (start, 1344x768 video with one video and one audio track playing, a
// second prompt applied, stop).
//
// No config file: the director's encoder is `auto`, which resolves to
// OpenH264 on a machine without NVENC, so the binary must be built with
// `encoders` (tests/console/run.sh builds `--features fake,encoders`).
//
//   FV_SERVE_BIN=target/debug/fv-serve node tests/console/smoke.cjs
//
// Needs the `playwright` npm package (resolved through NODE_PATH) and a
// Chromium under PLAYWRIGHT_BROWSERS_PATH; tests/console/run.sh sets both.
// Never prints the admin token or minted keys.
//
// Every page shows the server status strip (GET /fv/v1/status): the local
// pool's dot, the details panel, and the model page's pool badge; a pool
// reported down (status mocked in the browser) makes Run warn first and
// submit on the second click.
//
// A second fv-serve with FV_AUTH_MODE=none then checks the keyless console:
// capabilities report `auth.mode = none`, no key field, pill or banner, and
// text-to-video and a director session run with no key and no
// `Authorization` header on any request.
//
// Against a running server (the WP-18 GPU E2E): FV_CONSOLE_ORIGIN=<origin>
// with FV_ADMIN_TOKEN set skips starting fv-serve; FV_CONSOLE_TIMEOUT_MS
// (default 60 s) bounds each wait (real generations need minutes).

'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const zlib = require('node:zlib');
const { chromium } = require('playwright');

const BIN = process.env.FV_SERVE_BIN || path.join(__dirname, '..', '..', 'target', 'debug', 'fv-serve');
const REMOTE = process.env.FV_CONSOLE_ORIGIN || '';
const TIMEOUT = Number(process.env.FV_CONSOLE_TIMEOUT_MS || 60_000);

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
    s.on('error', reject);
  });
}

// A solid-colour RGB PNG (no image libraries needed).
function png(width, height, [r, g, b]) {
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (buf) => { let c = 0xffffffff; for (const x of buf) c = crcTable[(c ^ x) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td));
    return Buffer.concat([len, td, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const row = Buffer.alloc(1 + width * 3);
  for (let x = 0; x < width; x++) { row[1 + x * 3] = r; row[2 + x * 3] = g; row[3 + x * 3] = b; }
  const raw = Buffer.concat(Array.from({ length: height }, () => row));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
  ]);
}

// FV_CONSOLE_SHOTS=<dir>: also save screenshots (for reviewing the pages).
const SHOTS = process.env.FV_CONSOLE_SHOTS || '';
async function shot(page, name) {
  if (SHOTS) await page.screenshot({ path: path.join(SHOTS, name + '.png'), fullPage: true });
}

function step(msg) { process.stdout.write('  - ' + msg + '\n'); }

// Starts fv-serve with `extra` env on a free port; resolves once /health answers.
async function startServer(extra) {
  const port = await freePort();
  const origin = 'http://127.0.0.1:' + port;
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-console-smoke-'));
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port, FV_STATE_DIR: state, FV_JOB_STORE: 'memory', FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'console-smoke', RUST_LOG: 'warn',
  }, extra);
  const srv = { origin, state, logs: '', exited: null };
  srv.proc = spawn(BIN, [], { env, stdio: ['ignore', 'pipe', 'pipe'] });
  srv.proc.stdout.on('data', (d) => { srv.logs += d; });
  srv.proc.stderr.on('data', (d) => { srv.logs += d; });
  srv.proc.on('exit', (code) => { srv.exited = code; });
  for (let i = 0; i < 600; i++) {
    if (srv.exited !== null) throw new Error('fv-serve exited with ' + srv.exited);
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

// FV_AUTH_MODE=none: the console asks for no key and sends none.
async function authNone(browser, redact) {
  const srv = await startServer({ FV_AUTH_MODE: 'none' });
  try {
    const origin = srv.origin;
    const caps = await (await fetch(origin + '/fv/v1/capabilities')).json();
    if (!caps.auth || caps.auth.mode !== 'none') throw new Error('auth none: capabilities auth = ' + JSON.stringify(caps.auth));
    step('auth none: /fv/v1/capabilities reports auth.mode = none without a key');

    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await context.newPage();
    page.setDefaultTimeout(TIMEOUT);
    const errors = [];
    const authed = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(m.text()); });
    page.on('request', (r) => { if (r.headers().authorization) authed.push(r.method() + ' ' + new URL(r.url()).pathname); });

    await page.goto(origin + '/console');
    await page.waitForSelector('#conn-state.ok');
    await page.waitForFunction(() => document.querySelector('#conn').textContent === 'no key needed');
    if (await page.isVisible('#apikey')) throw new Error('auth none: the API key field is shown');
    if (await page.isVisible('#forget')) throw new Error('auth none: "Forget key" is shown');
    await shot(page, '08-auth-none-home');
    step('auth none: home connects with no key field');

    // A pool reported down: Run warns first, the second click submits.
    const real = await (await fetch(origin + '/fv/v1/status')).json();
    const down = JSON.parse(JSON.stringify(real));
    for (const p of down.pools) { p.state = 'down'; p.available = false; for (const w of p.workers) w.state = 'down'; }
    for (const m of Object.values(down.models)) m.state = 'down';
    down.state = 'down';
    await page.route('**/fv/v1/status', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(down) }));
    const submits = [];
    page.on('request', (r) => { if (r.method() === 'POST' && new URL(r.url()).pathname === '/minimax/h3-turbo/text-to-video') submits.push(r); });
    await page.goto(origin + '/console/models/minimax/h3-turbo/text-to-video');
    await page.waitForSelector('[data-input="prompt"]');
    await page.waitForSelector('#pool-state[data-state="down"] .dot.bad');
    await page.waitForSelector('#status-strip [data-pool="local"][data-state="down"] .dot.bad');
    await page.fill('[data-input="prompt"]', 'A lighthouse at dusk, waves below.');
    await page.click('#run');
    await page.waitForSelector('#run-msg.bad');
    if (!/down/.test(await page.textContent('#run-msg'))) throw new Error('down pool: no warning before Run');
    await new Promise((r) => setTimeout(r, 300));
    if (submits.length) throw new Error('down pool: submitted without the second click');
    await shot(page, '09-pool-down-warning');
    await page.click('#run');
    await page.waitForSelector('#result-status.s-COMPLETED', { timeout: TIMEOUT });
    if (submits.length !== 1) throw new Error('down pool: expected one submit after the second click, got ' + submits.length);
    await page.unroute('**/fv/v1/status');
    step('pool reported down: red badge, Run warned, second click submitted');

    await page.goto(origin + '/console/models/minimax/h3-turbo/text-to-video');
    await page.waitForSelector('[data-input="prompt"]');
    await page.waitForFunction(() => document.querySelector('#conn').textContent === 'no key needed');
    await page.waitForSelector('#pool-state[data-state="ready"]');
    if (await page.isVisible('#key-banner')) throw new Error('auth none: the key banner is shown');
    await page.fill('[data-input="prompt"]', 'A lighthouse at dusk, waves below.');
    await page.click('#run');
    await page.waitForSelector('#result-status.s-COMPLETED', { timeout: TIMEOUT });
    await page.waitForFunction(() => !!document.querySelector('#video')?.getAttribute('src'), null, { timeout: 30_000 });
    step('auth none: text-to-video ran without a key or banner');

    await page.click('[data-task="director"]');
    await page.waitForURL(/director$/);
    await page.waitForSelector('#director-start');
    await page.click('#director-start');
    await page.waitForFunction(() => document.querySelector('#director-state').textContent === 'streaming', null, { timeout: TIMEOUT });
    await page.click('#director-stop');
    await page.waitForFunction(() => document.querySelector('#director-state').textContent === 'closed', null, { timeout: 30_000 });
    step('auth none: director session started and stopped without a key');

    if (authed.length) throw new Error('auth none: requests sent Authorization: ' + authed.join(', '));
    if (errors.length) throw new Error('auth none: browser errors:\n' + errors.join('\n'));
    await context.close();
  } catch (e) {
    process.stderr.write('--- fv-serve (auth none) log ---\n' + redact(srv.logs).slice(-4000) + '\n');
    throw e;
  } finally {
    await stopServer(srv);
  }
}

async function main() {
  if (!REMOTE && !fs.existsSync(BIN)) throw new Error('fv-serve binary not found at ' + BIN + ' (cargo build -p fastvideo-serve --features fake)');
  const port = REMOTE ? 0 : await freePort();
  const origin = REMOTE ? REMOTE.replace(/\/$/, '') : 'http://127.0.0.1:' + port;
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-console-smoke-'));
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port,
    FV_STATE_DIR: state,
    FV_JOB_STORE: 'memory',
    FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'console-smoke',
    RUST_LOG: 'warn',
  });
  const server = REMOTE ? null : spawn(BIN, [], { env, stdio: ['ignore', 'pipe', 'pipe'] });
  let logs = '';
  let exited = null;
  if (server) {
    server.stdout.on('data', (d) => { logs += d; });
    server.stderr.on('data', (d) => { logs += d; });
    server.on('exit', (code) => { exited = code; });
  }
  const secrets = [];
  const redact = (s) => secrets.reduce((acc, x) => acc.split(x).join('<redacted>'), s);

  let browser;
  try {
    // Ready and the admin banner logged.
    let admin = REMOTE ? process.env.FV_ADMIN_TOKEN : null;
    if (REMOTE && !admin) throw new Error('FV_CONSOLE_ORIGIN needs FV_ADMIN_TOKEN');
    for (let i = 0; i < 600 && !admin; i++) {
      if (exited !== null) throw new Error('fv-serve exited with ' + exited);
      const m = logs.match(/fvadm_[A-Za-z0-9_-]{43}/);
      if (m) admin = m[0];
      else await new Promise((r) => setTimeout(r, 100));
    }
    if (!admin) throw new Error('no generated admin token in the fv-serve log');
    secrets.push(admin);
    if (!REMOTE && (logs.match(new RegExp(admin, 'g')) || []).length !== 1) throw new Error('admin token must be logged exactly once');
    step(REMOTE ? 'target ' + origin + ' (admin token from FV_ADMIN_TOKEN)' : 'fv-serve up on ' + origin + ' (generated admin token found in the WARN banner)');
    for (let i = 0; i < 600; i++) {
      try { const r = await fetch(origin + '/health'); if (r.ok) break; } catch { /* not yet */ }
      await new Promise((r) => setTimeout(r, 100));
    }

    browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await context.newPage();
    page.setDefaultTimeout(TIMEOUT);
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(m.text()); });

    // Home: no key yet, models listed from /fal/schema.
    await page.goto(origin + '/console');
    await page.waitForSelector('[data-endpoint="minimax/h3-max/reference-to-video"]');
    await shot(page, '01-home');
    step('home lists the fal apps');

    // Status strip: the local pool's dot, and the panel on click.
    if (!REMOTE) {
      await page.waitForSelector('#status-strip [data-pool="local"][data-state="ready"] .dot.ok');
      await page.click('#status-strip');
      await page.waitForSelector('#status-panel:not([hidden]) tr[data-pool="local"] [data-worker="local"]');
      const panel = await page.textContent('#status-panel');
      if (/127\.0\.0\.1|https?:\/\//.test(panel)) throw new Error('status panel shows an address: ' + panel.slice(0, 200));
      await shot(page, '01b-status-panel');
      await page.click('#status-strip');
      step('status strip: local pool ready (green), details panel lists the worker');
    } else {
      await page.waitForSelector('#status-strip .status-pool');
      step('status strip lists ' + (await page.$$('#status-strip .status-pool')).length + ' pools');
    }

    // Admin: wrong token refused, right token lists, mint a key.
    await page.goto(origin + '/console/admin');
    await page.fill('#admintoken', 'fvadm_wrong');
    await page.click('#admin-save');
    await page.waitForSelector('#admin-state.bad');
    await page.fill('#admintoken', admin);
    await page.click('#admin-save');
    await page.waitForSelector('#admin-state.ok');
    await page.fill('#keyname', 'playwright');
    await page.click('#mint');
    await page.waitForSelector('#minted:not([hidden])');
    const key = (await page.textContent('#minted-key')).trim();
    if (!/^fv_[A-Za-z0-9_-]{43}$/.test(key)) throw new Error('unexpected minted key shape');
    secrets.push(key);
    await page.waitForSelector('#keys tbody tr');
    await shot(page, '02-admin-minted');
    await page.click('#use-minted');
    step('minted a key on /console/admin and stored it in the browser');

    // Home: the key is accepted.
    await page.goto(origin + '/console');
    await page.waitForSelector('#conn-state.ok');
    step('key accepted by /fv/v1/capabilities');

    // Text to video.
    await page.goto(origin + '/console/models/minimax/h3-turbo/text-to-video');
    await page.waitForSelector('[data-input="prompt"]');
    await page.waitForSelector('#pool-state:not([hidden])');
    step('model page: pool badge "' + (await page.textContent('#pool-state')).trim() + '" next to Run');
    await page.fill('[data-input="prompt"]', 'A red fox trots across fresh snow at dawn, low tracking shot.');
    await page.selectOption('[data-input="aspect_ratio"]', '9:16');
    await page.click('#run');
    await page.waitForSelector('#result-status.s-COMPLETED', { timeout: TIMEOUT });
    // The result (and the player src) is fetched after the status turns
    // COMPLETED; over a real network that takes a moment.
    await page.waitForFunction(() => !!document.querySelector('#video')?.getAttribute('src'), null, { timeout: 30_000 });
    const src1 = await page.getAttribute('#video', 'src');
    const isMedia = (u) => !!u && (REMOTE ? /^https?:\/\//.test(u) : u.includes('/files/'));
    if (!isMedia(src1)) throw new Error('text-to-video: no video URL in the player (' + String(src1).split('?')[0].slice(0, 120) + ')');
    const v1 = await fetch(src1);
    if (!v1.ok || (await v1.arrayBuffer()).byteLength === 0) throw new Error('text-to-video: video URL does not serve');
    await shot(page, '03-t2v-result');
    const out1 = JSON.parse(await page.textContent('#output-json'));
    if (!out1.video || out1.video.content_type !== 'video/mp4') throw new Error('text-to-video: output JSON lacks video');
    step('text-to-video completed; player src ' + new URL(src1).pathname.split('/').slice(0, 3).join('/') + '/…');

    // API tab reflects the inputs.
    await page.click('#tab-api');
    const curl = await page.textContent('#snippet');
    if (!curl.includes('/minimax/h3-turbo/text-to-video') || !curl.includes('"9:16"')) throw new Error('API snippet does not reflect the inputs');
    await page.click('[data-lang="python"]');
    if (!(await page.textContent('#snippet')).includes('fal_client.subscribe')) throw new Error('python snippet missing');
    await shot(page, '04-api-tab');
    await page.click('#tab-playground');
    step('API tab: cURL / Python snippets carry the chosen inputs');

    // Image to video with an uploaded image.
    await page.click('[data-task="image-to-video"]');
    await page.waitForURL(/image-to-video$/);
    await page.waitForSelector('[data-field-file="image_url"]', { state: 'attached' });
    const img = path.join(state, 'first-frame.png');
    fs.writeFileSync(img, png(320, 180, [200, 90, 40]));
    await page.setInputFiles('[data-field-file="image_url"]', img);
    await page.waitForSelector('[data-field="image_url"] .media-item:not(.busy) img');
    await page.fill('[data-input="prompt"]', 'The camera slowly pulls back from the orange wall.');
    await page.click('#run');
    await page.waitForSelector('#result-status.s-COMPLETED', { timeout: TIMEOUT });
    await page.waitForFunction((prev) => { const s = document.querySelector('#video')?.getAttribute('src'); return !!s && s !== prev; }, src1, { timeout: 30_000 }).catch(() => {});
    const src2 = await page.getAttribute('#video', 'src');
    if (!isMedia(src2) || src2 === src1) throw new Error('image-to-video: no new video URL');
    const r2 = await fetch(src2);
    if (!r2.ok) throw new Error('image-to-video: video URL does not serve');
    await shot(page, '05-i2v-result');
    step('image-to-video with an uploaded image completed');

    // History holds both requests for this app.
    const rows = await page.$$('#history tbody tr');
    if (rows.length < 2) throw new Error('history should list both requests, got ' + rows.length);
    step('history lists ' + rows.length + ' requests');

    // Validation errors show fal's message.
    await page.fill('[data-input="prompt"]', ' ');
    await page.click('#run');
    await page.waitForSelector('#run-msg.bad');

    // Director: a live WebRTC session against the fake engine. The page
    // probes the signalling routes (POST /wma/ice) on load.
    const probed = page.waitForResponse((r) => new URL(r.url()).pathname === '/wma/ice');
    await page.click('[data-task="director"]');
    await page.waitForURL(/director$/);
    await page.waitForSelector('#director-start');
    const probe = await probed;
    if (probe.status() !== 200) throw new Error('director: POST /wma/ice answered ' + probe.status() + ' (built without webrtc?)');
    if (await page.isVisible('#director-unavailable')) throw new Error('director: shown as unavailable');
    await page.click('#director-start');
    await page.waitForFunction(() => document.querySelector('#director-state').textContent === 'streaming', null, { timeout: TIMEOUT });
    await page.waitForFunction(() => {
      const v = document.querySelector('#director-video');
      return v && v.videoWidth > 0 && v.currentTime > 1;
    }, null, { timeout: TIMEOUT });
    const vinfo = await page.evaluate(() => {
      const v = document.querySelector('#director-video');
      return { w: v.videoWidth, h: v.videoHeight, audio: v.srcObject.getAudioTracks().length, video: v.srcObject.getVideoTracks().length };
    });
    if (vinfo.w !== 1344 || vinfo.h !== 768) throw new Error('director: expected 1344x768 video, got ' + JSON.stringify(vinfo));
    if (vinfo.video !== 1 || vinfo.audio !== 1) throw new Error('director: expected one video and one audio track, got ' + JSON.stringify(vinfo));
    step('director streaming: ' + vinfo.w + 'x' + vinfo.h + ', ' + vinfo.video + ' video + ' + vinfo.audio + ' audio track, playing');
    await page.fill('#director-next', 'The keeper reaches the lamp room and looks out to sea.');
    await page.click('#director-send');
    await page.waitForFunction(
      () => [...document.querySelectorAll('#director-timeline li')].some((li) => li.textContent.startsWith('v2') && /applied/.test(li.lastChild.textContent)),
      null,
      { timeout: TIMEOUT },
    );
    step('director: second prompt (v2) applied');
    await shot(page, '06-director-streaming');
    await page.click('#director-stop');
    await page.waitForFunction(() => document.querySelector('#director-state').textContent === 'closed', null, { timeout: 30_000 });
    step('director: stopped (state closed)');

    // Phone width: no horizontal scrolling.
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(origin + '/console/models/minimax/h3-max/reference-to-video');
    await page.waitForSelector('[data-drop="reference_image_urls"]');
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    await shot(page, '07-r2v-phone');
    if (overflow > 1) throw new Error('horizontal overflow at 390 px: ' + overflow + ' px');
    step('reference-to-video form fits a 390 px viewport');

    // Revoke from the admin page: the key stops working.
    await page.goto(origin + '/console/admin');
    await page.waitForSelector('#keys tbody tr [data-revoke]');
    page.once('dialog', (d) => d.accept());
    await page.click('#keys tbody tr [data-revoke]');
    await page.waitForSelector('#keys tbody tr .pill.bad');
    const refused = await fetch(origin + '/minimax/h3-turbo/text-to-video', {
      method: 'POST', headers: { Authorization: 'Key ' + key, 'Content-Type': 'application/json' }, body: '{"prompt":"x"}',
    });
    if (refused.status !== 401) throw new Error('revoked key still accepted: ' + refused.status);
    step('revoked on /console/admin; the key is refused');

    if (errors.length) throw new Error('browser errors:\n' + errors.join('\n'));
    if (!REMOTE) await authNone(browser, redact);
    process.stdout.write('console smoke: OK\n');
  } catch (e) {
    process.stderr.write('console smoke FAILED: ' + redact(String(e && e.stack || e)) + '\n');
    process.stderr.write('--- fv-serve log (secrets redacted) ---\n' + redact(logs).slice(-6000) + '\n');
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (server) {
      server.kill('SIGTERM');
      await new Promise((r) => { if (exited !== null) r(); else { server.on('exit', r); setTimeout(r, 5000); } });
    }
    fs.rmSync(state, { recursive: true, force: true });
  }
}

main();
