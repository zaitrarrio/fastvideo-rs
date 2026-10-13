// Stream outputs in the console (docs/serve/stream-outputs.md), in headless
// Chromium against `fv-serve --features fake,full` with a fake Cloudflare
// Stream API in this script (its live inputs' WHIP ingest is a mock that
// answers but never connects ICE; its WHEP playback a mock that records the
// viewer's offer):
//   - the destination selector -> the request's `output` (Cloudflare
//     default, cloudflare-srt, your own RTMPS with a stream key, your own
//     WHIP with a token), and credentials never reach localStorage or the
//     page;
//   - the player per output: WHEP for the Cloudflare default (the viewer
//     POSTs to the live input's webRTCPlayback URL), HLS (the vendored
//     hls.js) for Cloudflare SRT, the pasted HLS URL for an RTMPS
//     destination of your own (an ffmpeg-made playlist that really plays);
//   - playerFor() for every playback shape;
//   - the director page's output copy: `x_fv_output` in configure (the
//     page log shows it redacted), `configured` echoes it.
//
//   FV_SERVE_BIN=target/debug/fv-serve node tests/console/stream_outputs.cjs

'use strict';

const { spawn, execFileSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');

const BIN = process.env.FV_SERVE_BIN || path.join(__dirname, '..', '..', 'target', 'debug', 'fv-serve');
const T = 60_000;
const CF_TOKEN = 'cf-console-test-token';
const SECRETS = ['user-stream-key-123', 'user-whip-token-456', CF_TOKEN];

function step(msg) { process.stdout.write('  - ' + msg + '\n'); }
function check(cond, what, extra) {
  if (!cond) throw new Error('stream outputs: ' + what + (extra !== undefined ? ' :: ' + JSON.stringify(extra) : ''));
}

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
    s.on('error', reject);
  });
}

// An SDP answer that is valid but never connects (enough for WHIP/WHEP).
function answer(offer) {
  const media = [];
  let cur = null;
  for (const l of offer.split(/\r?\n/)) {
    if (l.startsWith('m=')) { cur = { m: l, attrs: [] }; media.push(cur); } else if (cur) cur.attrs.push(l);
  }
  const mids = media.map((x) => (x.attrs.find((a) => a.startsWith('a=mid:')) || 'a=mid:0').slice(6));
  const fp = [...crypto.randomBytes(32)].map((b) => b.toString(16).padStart(2, '0').toUpperCase()).join(':');
  const out = ['v=0', 'o=- 1 1 IN IP4 127.0.0.1', 's=-', 't=0 0', 'a=group:BUNDLE ' + mids.join(' ')];
  media.forEach((x, i) => {
    const [kind, , proto, pt] = x.m.slice(2).split(' ');
    const dir = (x.attrs.find((a) => /^a=(sendonly|recvonly)$/.test(a)) || 'a=recvonly') === 'a=sendonly' ? 'a=recvonly' : 'a=sendonly';
    out.push('m=' + kind + ' 9 ' + proto + ' ' + pt, 'c=IN IP4 127.0.0.1', 'a=mid:' + mids[i], dir, 'a=rtcp-mux',
      'a=ice-ufrag:mock' + i, 'a=ice-pwd:mockmockmockmockmockmock', 'a=fingerprint:sha-256 ' + fp, 'a=setup:active',
      ...x.attrs.filter((a) => a.startsWith('a=rtpmap:' + pt + ' ') || a.startsWith('a=fmtp:' + pt + ' ')),
      'a=candidate:1 1 udp 2130706431 127.0.0.1 9 typ host');
  });
  return out.join('\r\n') + '\r\n';
}

// The fake Cloudflare API, its WHIP ingest and WHEP playback, and a static
// HLS fixture, on one port.
async function fakeCloudflare(hlsDir) {
  const seen = { creates: [], deletes: [], whip: [], whep: [] };
  const port = await freePort();
  const base = 'http://127.0.0.1:' + port;
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (d) => { body += d; });
    req.on('end', () => {
      const u = req.url;
      let m;
      if (req.method === 'POST' && /\/client\/v4\/accounts\/acct1\/stream\/live_inputs$/.test(u)) {
        const b = JSON.parse(body);
        seen.creates.push({ auth: req.headers.authorization, body: b });
        const uid = String(seen.creates.length).padStart(32, '0');
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ success: true, errors: [], messages: [], result: {
          uid, meta: b.meta, recording: b.recording,
          rtmps: { url: 'rtmps://127.0.0.1:1/live/', streamKey: 'cf-rtmps-key-000000000000000' },
          srt: { url: 'srt://127.0.0.1:1', streamId: 'cfstreamid', passphrase: 'cf-srt-passphrase-0000' },
          webRTC: { url: base + '/cfkey000000000000000000000000000000/webRTC/publish' },
          webRTCPlayback: { url: base + '/' + uid + '/webRTC/play' },
        } }));
      } else if (req.method === 'DELETE' && (m = /live_inputs\/(\w+)$/.exec(u))) {
        seen.deletes.push(m[1]);
        res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"success":true,"errors":[],"result":null}');
      } else if (req.method === 'PUT' && /live_inputs\//.test(u)) {
        res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"success":true,"errors":[],"result":null}');
      } else if (/\/webRTC\/publish$/.test(u) && req.method === 'POST') {
        seen.whip.push(u);
        res.writeHead(201, { 'content-type': 'application/sdp', location: u + '/r1' }); res.end(answer(body));
      } else if (/\/webRTC\/play$/.test(u) && req.method === 'POST') {
        seen.whep.push(u);
        res.writeHead(201, { 'content-type': 'application/sdp', 'access-control-allow-origin': '*', 'access-control-expose-headers': 'location', location: u + '/r1' });
        res.end(answer(body));
      } else if (req.method === 'OPTIONS') {
        res.writeHead(204, { 'access-control-allow-origin': '*', 'access-control-allow-methods': 'POST, DELETE', 'access-control-allow-headers': 'content-type, authorization' }); res.end();
      } else if (req.method === 'DELETE') {
        res.writeHead(200, { 'access-control-allow-origin': '*' }); res.end();
      } else if ((m = /^\/hls\/([\w.-]+)$/.exec(u)) && fs.existsSync(path.join(hlsDir, m[1]))) {
        const type = m[1].endsWith('.m3u8') ? 'application/vnd.apple.mpegurl' : 'video/mp4';
        res.writeHead(200, { 'content-type': type, 'access-control-allow-origin': '*' });
        res.end(fs.readFileSync(path.join(hlsDir, m[1])));
      } else {
        res.writeHead(404, { 'access-control-allow-origin': '*' }); res.end();
      }
    });
  });
  await new Promise((r) => server.listen(port, '127.0.0.1', r));
  return { server, seen, base };
}

async function startServer(cfBase) {
  const port = await freePort();
  const origin = 'http://127.0.0.1:' + port;
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-stream-outputs-'));
  const cfg = path.join(state, 'fv.toml');
  fs.writeFileSync(cfg, ['[webrtc]', 'public_ip = "127.0.0.1"', '', '[director]', 'encoder = "openh264"', ''].join('\n'));
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port, FV_STATE_DIR: state, FV_JOB_STORE: 'memory', FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'stream-outputs', FV_AUTH_MODE: 'none', RUST_LOG: 'warn', FV_STREAM_STUN: 'none',
    FV_CF_ACCOUNT_ID: 'acct1', FV_CF_STREAM_API_TOKEN: CF_TOKEN, FV_CF_STREAM_API_BASE: cfBase + '/client/v4',
    FV_CF_STREAM_NAME_PREFIX: 'fv-test-', FV_OUTPUTS_ALLOW_PRIVATE: '1', FV_STREAM_ENCODER: 'x264-test',
    // The fake API's URLs name no customer-<code> host: HLS URLs use this one.
    FV_CF_STREAM_CUSTOMER_DOMAIN: 'customer-test.invalid',
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

// A 4 s VOD playlist (320x240) for the HLS player: VP9 in fMP4 segments,
// because Playwright's Chromium has no H.264 for Media Source Extensions
// (Cloudflare's own HLS is H.264, which real browsers play).
function makeHls() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-hls-fixture-'));
  execFileSync(process.env.FV_FFMPEG || 'ffmpeg', ['-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=24', '-t', '4',
    '-c:v', 'libvpx-vp9', '-deadline', 'realtime', '-cpu-used', '8', '-pix_fmt', 'yuv420p', '-g', '24',
    '-f', 'hls', '-hls_time', '1', '-hls_playlist_type', 'vod', '-hls_segment_type', 'fmp4', '-hls_fmp4_init_filename', 'init.mp4',
    '-hls_segment_filename', path.join(dir, 'seg%d.m4s'), path.join(dir, 'index.m3u8')]);
  return dir;
}

// Starts a native stream with the form as filled; returns the POST body.
async function startNative(page) {
  const req = page.waitForRequest((r) => r.method() === 'POST' && r.url().endsWith('/fv/v1/streams'), { timeout: T });
  await page.click('#start');
  const body = (await req).postDataJSON();
  await page.waitForFunction(() => document.body.dataset.stream && document.body.dataset.player, null, { timeout: T });
  return body;
}

async function stopNative(page) {
  await page.click('#stop');
  await page.waitForFunction(() => document.body.dataset.streamState === 'stopped', null, { timeout: T });
}

async function noSecrets(page, where) {
  const leak = await page.evaluate((secrets) => {
    const ls = JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage });
    const text = document.body.innerText;
    return secrets.filter((s) => ls.includes(s) || text.includes(s));
  }, SECRETS);
  check(leak.length === 0, where + ': credentials leaked into storage or the page', leak);
}

async function streamPage(page, srv, cf) {
  await page.goto(srv.origin + '/console/stream?model=fake-sfwan');
  await page.waitForFunction(() => [...document.querySelectorAll('#model option')].some((o) => o.value === 'fake-sfwan'), null, { timeout: T });
  await page.selectOption('#transport', 'native');
  await page.waitForFunction(() => !document.querySelector('#out-kind option[value="cloudflare"]').disabled, null, { timeout: T });
  check((await page.inputValue('#out-kind')) === 'cloudflare', 'Cloudflare is the default destination');
  for (const id of ['#out-token', '#out-key', '#out-passphrase']) {
    check((await page.getAttribute(id, 'type')) === 'password', id + ' is a password input');
  }

  // 1. The Cloudflare default: WHEP playback of the live input.
  let body = await startNative(page);
  check(JSON.stringify(body.output) === JSON.stringify({ kind: 'cloudflare' }), 'default request shape', body);
  check((await page.evaluate(() => document.body.dataset.player)) === 'whep', 'WHEP player for the Cloudflare default');
  await page.waitForFunction(() => /WHEP/.test(document.getElementById('playback-info').textContent), null, { timeout: T });
  for (let i = 0; i < 100 && !cf.seen.whep.length; i++) await new Promise((r) => setTimeout(r, 100));
  check(cf.seen.whep.length === 1 && /\/webRTC\/play$/.test(cf.seen.whep[0]), 'the viewer POSTed to the live input\'s webRTCPlayback URL', cf.seen.whep);
  check(cf.seen.creates[0].body.meta.name.startsWith('fv-test-fvstream_'), 'live input name', cf.seen.creates[0].body);
  await page.waitForFunction(() => /native stream → cloudflare/.test(document.getElementById('stream-stats').textContent), null, { timeout: T });
  await stopNative(page);
  step('Cloudflare default: output {kind: cloudflare}, WHEP player on webRTCPlayback');

  // 2. Cloudflare over SRT: HLS through the vendored hls.js.
  await page.selectOption('#out-kind', 'cloudflare-srt');
  check(await page.isHidden('#out-url'), 'no URL for a Cloudflare kind');
  body = await startNative(page);
  check(JSON.stringify(body.output) === JSON.stringify({ kind: 'cloudflare-srt' }), 'cloudflare-srt request shape', body);
  const p2 = await page.evaluate(() => [document.body.dataset.player, document.getElementById('video').dataset.hlsEngine, !!window.Hls]);
  check(p2[0] === 'hls' && p2[1] === 'hls.js' && p2[2], 'HLS player (hls.js) for Cloudflare SRT', p2);
  check(/HLS · https:\/\/customer-test\.invalid\/0+2\/manifest\/video\.m3u8/.test(await page.textContent('#playback-info')), 'the live input\'s HLS URL', await page.textContent('#playback-info'));
  check(cf.seen.creates[1].body.recording.mode === 'automatic', 'an SRT input records (HLS needs it)', cf.seen.creates[1].body);
  await stopNative(page);
  step('cloudflare-srt: output {kind: cloudflare-srt}, HLS player (hls.js) on hls_url');

  // 3. Your own RTMPS with a stream key, played from a pasted HLS URL.
  await page.selectOption('#out-kind', 'rtmps');
  await page.fill('#out-url', 'rtmp://127.0.0.1:1/live');
  await page.fill('#out-key', 'user-stream-key-123');
  await page.fill('#out-play', cf.base + '/hls/index.m3u8');
  body = await startNative(page);
  check(JSON.stringify(body.output) === JSON.stringify({ kind: 'rtmps', url: 'rtmp://127.0.0.1:1/live', stream_key: 'user-stream-key-123' }), 'rtmps request shape', body.output);
  await page.waitForFunction(() => { const v = document.getElementById('video'); return v.videoWidth > 0 && v.currentTime > 0.3; }, null, { timeout: T });
  const v3 = await page.evaluate(() => { const v = document.getElementById('video'); return [document.body.dataset.player, v.videoWidth, v.videoHeight]; });
  check(v3[0] === 'hls' && v3[1] === 320 && v3[2] === 240, 'the pasted HLS playlist plays', v3);
  await page.waitForFunction(() => /rtmps \(rtmp:\/\/127\.0\.0\.1:1\/live\)/.test(document.getElementById('stream-stats').textContent), null, { timeout: T });
  await noSecrets(page, 'stream page (rtmps)');
  await stopNative(page);
  step('rtmps of your own: output {kind, url, stream_key}, the pasted HLS URL plays 320x240, no key on the page or in storage');

  // 4. Your own WHIP with a token and VP8: request shape only.
  await page.selectOption('#out-kind', 'whip');
  await page.fill('#out-url', cf.base + '/userkey00000000000000000000000000/webRTC/publish');
  await page.fill('#out-token', 'user-whip-token-456');
  await page.selectOption('#out-codec', 'vp8');
  await page.fill('#out-play', '');
  body = await startNative(page);
  check(body.output.kind === 'whip' && body.output.token === 'user-whip-token-456' && body.output.codec === 'vp8', 'whip request shape', { ...body.output, token: 'set' });
  check((await page.evaluate(() => document.body.dataset.player)) === 'none', 'no player without a playback URL');
  check(/paste its WHEP or HLS URL/.test(await page.textContent('#playback-info')), 'the page asks for a playback URL');
  await noSecrets(page, 'stream page (whip)');
  await stopNative(page);
  step('whip of your own: output {kind, url, token, codec: vp8}; no known player -> asks for a WHEP / HLS URL');

  // playerFor for every playback shape.
  const choices = await page.evaluate(async () => {
    const { playerFor } = await import('/console/assets/outputs.js');
    return [
      playerFor({ format: 'whep', whep_url: 'https://c/x/webRTC/play' }, ''),
      playerFor({ format: 'hls', hls_url: 'https://c/x/manifest/video.m3u8', dash_url: 'https://c/x/manifest/video.mpd' }, ''),
      playerFor({ format: 'whep', whep_url: 'https://c/w', hls_url: 'https://c/h.m3u8' }, ''),
      playerFor({}, 'https://relay/live/whep'),
      playerFor({}, 'https://relay/live/index.m3u8?x=1'),
      playerFor({}, ''),
    ].map((c) => c.kind);
  });
  check(JSON.stringify(choices) === JSON.stringify(['whep', 'hls', 'whep', 'whep', 'hls', null]), 'playerFor', choices);
  step('playerFor: WHEP / HLS / pasted / none');
}

async function directorPage(page, srv) {
  await page.goto(srv.origin + '/console/models/minimax/h3-turbo/director');
  await page.waitForSelector('#director-start');
  await page.click('#director-output summary');
  await page.waitForFunction(() => !document.querySelector('#director-out-kind option[value="cloudflare"]').disabled, null, { timeout: T });
  check((await page.inputValue('#director-out-kind')) === '', 'no copy by default');
  await page.selectOption('#director-out-kind', 'whip');
  await page.fill('#director-out-url', 'http://127.0.0.1:9/live/whip');
  await page.fill('#director-out-token', 'user-whip-token-456');
  await page.click('#director-start');
  await page.waitForFunction(() => /Output copy: whip → http:\/\/127\.0\.0\.1:9\/live\/whip/.test(document.getElementById('director-copy-status').textContent), null, { timeout: T });
  const log = await page.textContent('#director-log');
  check(/"x_fv_output":\{"kind":"whip","url":"http:\/\/127\.0\.0\.1:9\/live\/whip","token":"set"\}/.test(log), 'the configure carried x_fv_output (redacted in the log)', log.slice(0, 2000));
  await noSecrets(page, 'director page');
  step('director: x_fv_output in configure (token redacted in the page log), configured echoes the copy');
  await page.click('#director-stop');
}

(async () => {
  const hlsDir = makeHls();
  const cf = await fakeCloudflare(hlsDir);
  const srv = await startServer(cf.base);
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 1000 } })).newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    process.stdout.write('stream outputs:\n');
    await streamPage(page, srv, cf);
    await directorPage(page, srv);
    check(errors.length === 0, 'page errors', errors);
    // Every Cloudflare input the streams created was released.
    for (let i = 0; i < 50 && cf.seen.deletes.length < 1; i++) await new Promise((r) => setTimeout(r, 100));
    check(cf.seen.deletes.length >= 1, 'the WebRTC live input was deleted', cf.seen);
    check(cf.seen.creates.every((c) => c.auth === 'Bearer ' + CF_TOKEN), 'the API token went to Cloudflare only as a bearer');
    check(!srv.logs.includes(CF_TOKEN) && !srv.logs.includes('user-stream-key-123') && !srv.logs.includes('user-whip-token-456'), 'no secret in the server log');
    process.stdout.write('stream outputs: ok\n');
  } catch (e) {
    process.stdout.write(srv.logs.slice(-4000) + '\n');
    throw e;
  } finally {
    await browser.close();
    await stopServer(srv);
    cf.server.close();
    fs.rmSync(hlsDir, { recursive: true, force: true });
  }
})().catch((e) => { console.error(e); process.exit(1); });
