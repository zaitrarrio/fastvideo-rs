// The console pages and fields added for the UI gaps (docs/serve/console.md),
// in headless Chromium against `fv-serve --features fake,full` (keys mode,
// the Reactor runtime on the causal `fake-sfwan`, the fal app
// `fastvideo/fake-sfwan` for a causal director):
//   - Models: served models with tier and recipe, the mounted APIs
//     (`protocols` of /fv/v1/capabilities), no H3 tabs on an app that serves
//     no endpoint, the causal director tag.
//   - A model page: the result's x-fv-tier / x-fv-recipe, and a draft
//     result (x-fv-quality: draft, added in the browser: the fake engine has
//     no draft tier) flagged above the video and in the history.
//   - Reference to video: x-fv-min/max-references (refused at 0, the drop
//     zones close at 12); the API tab offers the mounted APIs (MiniMax for
//     H3, no LTX) with their own bodies.
//   - The clip director: schema-driven resolution, the session's model,
//     tier and recipe, session_info, a configure the session cannot take
//     (driving audio) refused before it is sent, a script-only prompt, the
//     chunk size selector (the schema's `chunk_duration`, or a mocked one on
//     a server without it) narrowed per resolution and sent in configure.
//   - The causal director (fastvideo/fake-sfwan): text-only 480p / 16:9
//     form, chunks and an applied prompt; a licence (mocked: LongLive is not
//     in the fake set) shown as a non-commercial banner.
//   - Live stream: the Reactor transport (832x480 video, a prompt switch
//     applied, stats, stop) and the native POST /fv/v1/streams transport to
//     a mock WHIP endpoint in this script (create, a prompt switch through
//     /commands, stats, Stop -> DELETE).
//   - Native API: tier aliases, a job with steps / guidance (tier and recipe
//     in the result), the OpenAI path with flow_shift on a model that honours it.
//
//   FV_SERVE_BIN=target/debug/fv-serve node tests/console/ui_gaps.cjs
//   FV_CONSOLE_SHOTS=<dir> also saves screenshots.

'use strict';

const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');

const BIN = process.env.FV_SERVE_BIN || path.join(__dirname, '..', '..', 'target', 'debug', 'fv-serve');
const KEY = 'fv_ui-gaps-test-key';
const SHOTS = process.env.FV_CONSOLE_SHOTS || '';
const T = 60_000;

function step(msg) { process.stdout.write('  - ' + msg + '\n'); }
function check(cond, what, extra) {
  if (!cond) throw new Error('ui gaps: ' + what + (extra !== undefined ? ' :: ' + JSON.stringify(extra) : ''));
}
async function shot(page, name) {
  if (SHOTS) await page.screenshot({ path: path.join(SHOTS, name + '.png'), fullPage: true });
}

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
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-ui-gaps-'));
  const cfg = path.join(state, 'fv.toml');
  fs.writeFileSync(cfg, [
    '[webrtc]', 'public_ip = "127.0.0.1"', 'ice_servers = [{ urls = ["stun:127.0.0.1:9"] }]', '',
    '[director]', 'encoder = "openh264"', 'vp8_fallback = true', '',
    '[protocols]', 'fal_apps = ["minimax/h3-max", "minimax/h3-turbo", "minimax/h3-draft", "fastvideo/fake-sfwan"]', '',
  ].join('\n'));
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port, FV_STATE_DIR: state, FV_JOB_STORE: 'memory', FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'ui-gaps', FV_API_KEYS: crypto.createHash('sha256').update(KEY).digest('hex'),
    FV_REACTOR_MODEL: 'fake-sfwan', FV_STREAM_STUN: 'none', RUST_LOG: 'warn',
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

// A WHIP endpoint that answers every offer (RFC 9725 shape: 201, an SDP
// answer, a Location) with a syntactically valid answer whose ICE never
// connects: enough for the stream to be created, to publish, to take
// commands and to be stopped. `hits` records the requests.
function mockWhip() {
  const hits = [];
  const answer = (offer) => {
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
      out.push('m=' + kind + ' 9 ' + proto + ' ' + pt, 'c=IN IP4 127.0.0.1', 'a=mid:' + mids[i], 'a=recvonly', 'a=rtcp-mux',
        'a=ice-ufrag:mock' + i, 'a=ice-pwd:mockmockmockmockmockmock', 'a=fingerprint:sha-256 ' + fp, 'a=setup:active',
        ...x.attrs.filter((a) => a.startsWith('a=rtpmap:' + pt + ' ') || a.startsWith('a=fmtp:' + pt + ' ')),
        'a=candidate:1 1 udp 2130706431 127.0.0.1 9 typ host');
    });
    return out.join('\r\n') + '\r\n';
  };
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (d) => { body += d; });
    req.on('end', () => {
      hits.push(req.method + ' ' + req.url);
      if (req.method === 'POST') { res.writeHead(201, { 'content-type': 'application/sdp', location: '/whip/res1' }); res.end(answer(body)); } else { res.writeHead(200); res.end(); }
    });
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, hits, url: 'http://127.0.0.1:' + server.address().port + '/whip' })));
}

const resolutionReady = () => { const r = document.getElementById('director-resolution'); return !!r && !r.disabled; };
const directorState = (s) => document.getElementById('director-state') && document.getElementById('director-state').textContent === s;

async function home(page, origin) {
  await page.goto(origin + '/console');
  await page.waitForSelector('#served tr[data-model="fake-sfwan"]', { timeout: T });
  const row = (id) => page.$eval('#served tr[data-model="' + id + '"]', (r) => [...r.children].map((c) => c.textContent));
  const sf = await row('fake-sfwan');
  check(/attention dense · vae tiny · 4 steps/.test(sf[2]) && /causal-4step/.test(sf[2]), 'the recipe of fake-sfwan', sf);
  check(/Live stream/.test(sf[4]) && /120 s default, 300 s max/.test(sf[4]), 'the causal model links Live stream with its length rule', sf);
  const turbo = await row('fake-h3-turbo');
  check(turbo[1].startsWith('turbo') && /4step-vsa/.test(turbo[2]), 'tier and recipe of fake-h3-turbo', turbo);
  const apis = await page.textContent('#server-apis');
  check(/native/.test(apis) && /fal/.test(apis) && /minimax/.test(apis) && /reactor/.test(apis), 'the mounted APIs', apis);
  // minimax/h3-draft serves nothing here: no H3 tabs invented for it.
  const draft = await page.$eval('[data-app="minimax/h3-draft"]', (c) => ({ items: c.querySelectorAll('li').length, unserved: !!c.querySelector('[data-unserved]') }));
  check(draft.items === 0 && draft.unserved, 'an app with no served endpoint lists none', draft);
  const causal = await page.$eval('[data-app="fastvideo/fake-sfwan"]', (c) => c.textContent);
  check(/live · causal/.test(causal), 'the causal director tag', causal);
  step('Models: tier, recipe, length rule, APIs (' + apis + '), no stale H3 tabs');
  await shot(page, 'home');
}

async function modelPage(page, origin) {
  await page.goto(origin + '/console/models/minimax/h3-turbo/text-to-video');
  await page.waitForSelector('[data-input="prompt"]', { timeout: T });
  await page.fill('[data-input="prompt"]', 'a paper boat on a pond');
  await page.click('#run');
  await page.waitForSelector('#video:not([hidden])', { timeout: T });
  const tier = await page.textContent('[data-fact="tier"]');
  const recipe = await page.textContent('[data-fact="recipe"]');
  check(tier.startsWith('turbo') && recipe === '4step-vsa', 'the result shows x-fv-tier and x-fv-recipe', { tier, recipe });
  check(await page.isHidden('#result-quality'), 'a turbo result is not flagged as draft');
  step('model page: result tier ' + tier + ', recipe ' + recipe);
  // A draft tier's result (x-fv-quality: draft): added to the real response.
  await page.route('**/minimax/h3-turbo/requests/*', async (route) => {
    if (route.request().url().includes('/status')) return route.continue();
    const r = await route.fetch();
    await route.fulfill({ response: r, headers: { ...r.headers(), 'x-fv-tier': 'draft', 'x-fv-quality': 'draft' } });
  });
  await page.fill('[data-input="prompt"]', 'a draft preview');
  await page.click('#run');
  await page.waitForSelector('#result-quality:not([hidden])', { timeout: T });
  check(/draft quality/.test(await page.textContent('[data-fact="tier"]')), 'the draft pill next to the tier');
  await page.waitForSelector('#history [data-quality="draft"]', { timeout: T });
  step('model page: a draft result is flagged above the video and in the history');
  await shot(page, 'model-draft');
  await page.unroute('**/minimax/h3-turbo/requests/*');
}

async function references(page, origin) {
  await page.goto(origin + '/console/models/minimax/h3-max/reference-to-video');
  await page.waitForSelector('[data-ref-count]', { timeout: T });
  await page.fill('[data-input="prompt"]', 'two friends at a cafe');
  await page.click('#run');
  check(/at least 1 reference/.test(await page.textContent('#run-msg')), 'Run is refused with no reference', await page.textContent('#run-msg'));
  const add = async (field, url) => {
    await page.fill('[data-field-url="' + field + '"]', url);
    await page.click('[data-field="' + field + '"] .urlrow button');
  };
  for (let i = 0; i < 9; i++) await add('reference_image_urls', 'data:image/png;base64,iVBORw0KGgo=' + i);
  for (let i = 0; i < 3; i++) await add('reference_video_urls', 'https://example.invalid/v' + i + '.mp4');
  const n = await page.getAttribute('[data-ref-count]', 'data-ref-count');
  const full = await page.getAttribute('[data-field="reference_audio_urls"]', 'data-full');
  check(n === '12' && full === 'true', 'the reference lists close at x-fv-max-references (12)', { n, full });
  step('reference to video: refused at 0 references, closed at 12 in all');
  await page.click('#tab-api');
  await page.waitForSelector('#protocols [data-proto="minimax"]', { timeout: T });
  const protos = await page.$$eval('#protocols [data-proto]', (x) => x.map((y) => y.dataset.proto));
  check(protos.includes('native') && !protos.includes('ltx') && !protos.includes('openai_videos') && !protos.includes('reactor'), 'the APIs that can run H3 reference to video (the Reactor streams fake-sfwan)', protos);
  await page.click('[data-proto="minimax"]');
  const mm = await page.textContent('#snippet');
  check(/\/v2\/video_generation/.test(mm) && /MiniMax-H3-Max/.test(mm) && /"reference_image"/.test(mm), 'the MiniMax snippet', mm.slice(0, 400));
  await page.click('[data-proto="native"]');
  const nat = await page.textContent('#snippet');
  check(/\/fv\/v1\/jobs/.test(nat) && /"reference_urls"/.test(nat), 'the native snippet', nat.slice(0, 400));
  step('API tab: ' + protos.join(', ') + ' with their own request bodies');
  await shot(page, 'api-snippets');
}

async function clipDirector(page, origin) {
  const schema = await (await fetch(origin + '/fal/schema/minimax/h3-turbo/director')).json();
  const served = schema.properties && schema.properties.chunk_duration;
  if (!served) {
    // A server without the chunk size (before wip/director-chunk): the same shape, mocked.
    // 480p is listed too so the per-resolution narrowing runs (Start goes back to the default).
    const resolution = { ...schema.properties.resolution, enum: [...new Set(['480p', ...schema.properties.resolution.enum])] };
    await page.route('**/fal/schema/minimax/h3-turbo/director', (r) => r.fulfill({ json: { ...schema, properties: { ...schema.properties, resolution, chunk_duration: {
      type: 'integer', enum: [5, 10], default: 5, description: 'Seconds per chunk.',
      'x-fv-labels': { 5: '5 s (prompt changes land sooner)', 10: '10 s (fewer joins)' }, 'x-fv-options-by-resolution': { '768p': [5, 10], '480p': [5] },
    } } } }));
  }
  await page.goto(origin + '/console/models/minimax/h3-turbo/director');
  await page.waitForFunction(resolutionReady, null, { timeout: T });
  const res = await page.$$eval('#director-resolution option', (o) => o.map((x) => x.value));
  check(res.length && schema.properties.resolution.enum.every((r) => res.includes(r)), 'resolutions from the schema', res);
  await page.waitForSelector('#director-chunk-duration', { timeout: T });
  const opts = () => page.$$eval('#director-chunk-duration option', (o) => o.map((x) => x.value).join(','));
  const by = (served || {})['x-fv-options-by-resolution'] || { '768p': [5, 10], '480p': [5] };
  const narrow = Object.entries(by).find(([, v]) => v.length === 1);
  if (narrow && res.includes(narrow[0])) {
    await page.selectOption('#director-resolution', narrow[0]);
    check((await opts()) === String(narrow[1][0]), 'the chunk sizes narrow per resolution', await opts());
    step('director: at ' + narrow[0] + ' the chunk size narrows to ' + (await opts()));
    await page.selectOption('#director-resolution', schema.properties.resolution.default);
  }
  await page.selectOption('#director-chunk-duration', '10');
  step('director: resolutions ' + res.join('/') + ' from the schema; chunk size options ' + (await opts()) + (served ? '' : ' (mocked)'));

  // A configure the session cannot take is stopped before it is sent.
  await page.click('#director-more summary');
  await page.fill('#director-audio', 'https://example.invalid/voice.wav');
  await page.click('#director-start');
  await page.waitForFunction(() => /no driving audio/.test(document.getElementById('director-msg').textContent), null, { timeout: T });
  step('director: driving audio refused from session_info (audio_conditioning false) before configure');
  await page.fill('#director-audio', '');
  await page.waitForFunction(() => !document.getElementById('director-start').disabled, null, { timeout: T });
  await page.click('#director-start');
  if (!served) {
    // A server without the field answers invalid_message; the configure sent is what is checked.
    await page.waitForFunction(() => /"type":"→ configure"/.test(document.getElementById('director-log').textContent), null, { timeout: T });
    const sent = await page.$$eval('#director-log div', (d) => d.map((x) => x.textContent).find((t) => t.includes('"type":"→ configure"')));
    check(/"chunk_duration":10/.test(sent), 'configure carries chunk_duration', sent);
    step('director: configure carries chunk_duration 10 (this server has no chunk size yet)');
    await page.click('#director-stop');
    await page.waitForFunction(() => !document.getElementById('director-start').disabled, null, { timeout: T });
    await page.unroute('**/fal/schema/minimax/h3-turbo/director');
    await page.reload();
    await page.waitForFunction(resolutionReady, null, { timeout: T });
    await page.click('#director-start');
  }
  await page.waitForFunction(directorState, 'streaming', { timeout: T });
  await page.waitForSelector('#director-stats dd', { timeout: T });
  const stats = await page.textContent('#director-stats');
  check(/fake-h3-turbo/.test(stats) && /turbo/.test(stats) && /4step-vsa/.test(stats), 'the session\'s model, tier and recipe (x-fv-* headers)', stats);
  if (served) check(/chunk length10 s/.test(stats), 'the chunk length the session runs (configured.chunk_duration)', stats);
  const info = await page.textContent('#director-session-info');
  check(/audio conditioningno/.test(info) && /scripts/.test(info), 'session_info facts', info);
  // A script is sent on its own.
  await page.click('#director-next-more summary');
  await page.fill('#director-next', 'they reach the lamp room');
  await page.click('#director-next-script-add-beat');
  await page.fill('#director-next-script-beats [data-beat="offset"]', '2');
  await page.fill('#director-next-script-beats [data-beat="prompt"]', 'the lamp turns on');
  await page.click('#director-send');
  check(/sent on its own/.test(await page.textContent('#director-msg')), 'a script with a prompt is refused in the page');
  await page.fill('#director-next', '');
  await page.click('#director-send');
  await page.waitForFunction(() => /1 beats \(replace\)/.test(document.getElementById('director-timeline').textContent), null, { timeout: T });
  await page.waitForTimeout(1500);
  const tl = await page.textContent('#director-timeline');
  check(!/rejected/.test(tl), 'the script-only prompt is accepted', tl);
  step('director: model/tier/recipe from the session headers, session_info, a script-only prompt accepted');
  await shot(page, 'director-clip');
  await page.click('#director-stop');
  await page.waitForFunction(() => !document.getElementById('director-start').disabled, null, { timeout: T });
}

async function causalDirector(page, origin) {
  await page.goto(origin + '/console/models/fastvideo/fake-sfwan/director');
  await page.waitForFunction(resolutionReady, null, { timeout: T });
  const form = await page.evaluate(() => ({
    mode: document.querySelector('[data-director-mode]').dataset.directorMode,
    image: document.querySelector('[data-cond="image"]').hidden,
    audio: document.querySelector('[data-cond="audio"]').hidden,
    end: document.querySelector('[data-cond="end-image"]').hidden,
    res: [...document.querySelectorAll('#director-resolution option')].map((o) => o.value).join(','),
    aspect: [...document.querySelectorAll('#director-aspect option')].map((o) => o.value).join(','),
    chunk: !!document.getElementById('director-chunk-duration'),
  }));
  check(form.mode === 'causal' && form.image && form.audio && form.end && form.res === '480p' && form.aspect === 'auto,16:9' && !form.chunk, 'the causal form: text only, 480p, 16:9, no chunk size', form);
  await page.click('#director-start');
  await page.waitForFunction(directorState, 'streaming', { timeout: T });
  await page.waitForFunction(() => /chunks/.test(document.getElementById('director-stats').textContent), null, { timeout: T });
  await page.fill('#director-next', 'the road turns inland');
  await page.click('#director-send');
  await page.waitForFunction(() => /the road turns inlandapplied/.test(document.getElementById('director-timeline').textContent), null, { timeout: T });
  const size = await page.evaluate(() => { const v = document.getElementById('director-video'); return [v.videoWidth, v.videoHeight]; });
  check(size[0] === 832 && size[1] === 480, 'the causal stream plays at 832x480', size);
  step('causal director: text-only 480p form, chunks, prompt applied, ' + size.join('x'));
  await shot(page, 'director-causal');
  await page.click('#director-stop');
  await page.waitForFunction(() => !document.getElementById('director-start').disabled, null, { timeout: T });
  // A LongLive app advertises its licence (the fake set has no LongLive: mocked).
  const licence = 'LongLive-1.3B weights (NVIDIA, HF card: CC-BY-NC-SA 4.0): non-commercial use only (research and evaluation).';
  const real = await (await fetch(origin + '/fal/schema/fastvideo/fake-sfwan/director')).json();
  await page.route('**/fal/schema/fastvideo/fake-sfwan/director', (r) => r.fulfill({ json: { ...real, 'x-fv-licence': licence } }));
  await page.reload();
  await page.waitForSelector('[data-licence="non-commercial"]', { timeout: T });
  step('causal director: the licence shows as a non-commercial banner');
  await page.unroute('**/fal/schema/fastvideo/fake-sfwan/director');
}

async function liveStream(page, origin, whip) {
  await page.goto(origin + '/console/stream');
  await page.waitForFunction(() => [...document.querySelectorAll('#model option')].some((o) => o.value === 'fake-sfwan'), null, { timeout: T });
  const facts = await page.textContent('#model-facts');
  check(/12-frame blocks at 16 fps/.test(facts) && /120 s by default, at most 300 s/.test(facts), 'the causal model facts', facts);
  check((await page.inputValue('#transport')) === 'reactor', 'the Reactor transport is offered for the Reactor\'s model');
  check(!(await page.isHidden('#director-link')), 'the causal director link');
  await page.click('#start');
  await page.waitForFunction(() => document.body.dataset.streamState === 'streaming', null, { timeout: T });
  await page.waitForFunction(() => Number(document.body.dataset.block || 0) > 2, null, { timeout: T });
  await page.fill('#next', 'a pine forest at dawn');
  await page.click('#switch');
  await page.waitForFunction(() => /a pine forest at dawnapplied/.test(document.getElementById('timeline').textContent), null, { timeout: T });
  const v = await page.evaluate(() => { const x = document.getElementById('video'); return [x.videoWidth, x.videoHeight, x.currentTime]; });
  check(v[0] === 832 && v[1] === 480 && v[2] > 0, 'the stream plays', v);
  const stats = await page.textContent('#stream-stats');
  check(/a pine forest at dawn/.test(stats) && /fps/.test(stats), 'the stream stats', stats);
  step('Live stream (Reactor): ' + v[0] + 'x' + v[1] + ', prompt switch applied');
  await shot(page, 'stream-reactor');
  await page.click('#stop');
  await page.waitForFunction(() => document.body.dataset.streamState === 'stopped', null, { timeout: T });

  // Native POST /fv/v1/streams to the mock WHIP endpoint.
  await page.selectOption('#transport', 'whip');
  await page.fill('#whip-url', whip.url);
  await page.fill('#max-seconds', '60');
  await page.click('#start');
  await page.waitForFunction(() => document.body.dataset.stream, null, { timeout: T });
  await page.waitForFunction(() => /publishing|streaming/.test(document.getElementById('stream-stats').textContent), null, { timeout: T });
  check(whip.hits.some((h) => h.startsWith('POST')), 'fv-serve published to the WHIP endpoint', whip.hits);
  await page.fill('#next', 'a harbour at night');
  await page.click('#switch');
  await page.waitForFunction(() => /a harbour at nightapplied/.test(document.getElementById('timeline').textContent), null, { timeout: T });
  const st = await page.textContent('#stream-stats');
  check(/fvstream_/.test(st) && /pacer/.test(st) && /\/ 60 s/.test(st), 'the native stream stats', st);
  step('Live stream (native): POST /fv/v1/streams published to WHIP, set_prompt through /commands');
  await shot(page, 'stream-native');
  const id = await page.evaluate(() => document.body.dataset.stream);
  await page.click('#stop');
  await page.waitForFunction(() => document.body.dataset.streamState === 'stopped', null, { timeout: T });
  const after = await (await fetch(origin + '/fv/v1/streams/' + id, { headers: { Authorization: 'Bearer ' + KEY } })).json();
  check(after.status.state === 'closed' && after.status.end_reason === 'stopped', 'Stop deleted the stream', after.status);
  step('Live stream (native): Stop -> DELETE, the stream closed (stopped)');
}

async function nativePage(page, origin) {
  await page.goto(origin + '/console/native?model=ltx-turbo');
  await page.waitForSelector('[data-input="prompt"]', { timeout: T });
  const models = await page.$$eval('#model option', (o) => o.map((x) => x.value));
  check(models.includes('ltx-turbo') && models.includes('fake-ltx-turbo') && !models.includes('fake-sfwan'), 'served models and tier aliases (no causal model)', models);
  const tasks = await page.$$eval('#task option', (o) => o.map((x) => x.value));
  check(['t2v', 'i2v', 'a2v', 'retake', 'extend'].every((t) => tasks.includes(t)), 'the LTX tasks', tasks);
  await page.fill('[data-input="prompt"]', 'a red fox in the snow');
  await page.fill('[data-input="steps"]', '4');
  await page.fill('[data-input="guidance"]', '3');
  await page.selectOption('[data-input="aspect_ratio"]', '16:9');
  await page.selectOption('[data-input="short_edge"]', '480');
  const body = JSON.parse(await page.textContent('#body-json'));
  check(body.model === 'ltx-turbo' && body.steps === 4 && body.guidance === 3 && body.short_edge === 480, 'the native body', body);
  await page.click('#run');
  await page.waitForFunction(() => document.body.dataset.job, null, { timeout: T });
  check((await page.evaluate(() => document.body.dataset.job)) === 'succeeded', 'the native job succeeded', await page.textContent('#facts'));
  const tier = await page.textContent('[data-fact="tier"]');
  const recipe = await page.textContent('[data-fact="recipe"]');
  check(tier === 'turbo' && recipe === 'two-stage-sol', 'tier and recipe of the native job', { tier, recipe });
  step('Native API: ltx-turbo job with steps and guidance -> ' + tier + ' / ' + recipe);
  await shot(page, 'native');
  await page.selectOption('#task', 'retake');
  const retake = await page.$$eval('[data-block]', (b) => b.map((x) => x.dataset.block));
  check(['video_url', 'start_s', 'end_s', 'retake_mode'].every((f) => retake.includes(f)) && !retake.includes('seconds'), 'the retake fields', retake);
  // flow_shift is an OpenAI /v1/videos field; fake-wan honours it.
  await page.selectOption('#model', 'fake-wan');
  await page.selectOption('#api', 'openai');
  await page.waitForSelector('[data-input="flow_shift"]', { timeout: T });
  await page.fill('[data-input="prompt"]', 'a lighthouse in fog');
  await page.fill('[data-input="flow_shift"]', '5');
  await page.click('#run');
  await page.waitForFunction(() => document.body.dataset.job, null, { timeout: T });
  check((await page.evaluate(() => document.body.dataset.job)) === 'completed', 'the OpenAI job with flow_shift completed', await page.textContent('#job-json'));
  step('Native API: flow_shift through OpenAI /v1/videos on fake-wan');
}

(async () => {
  const srv = await startServer();
  const whip = await mockWhip();
  const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
  let page;
  let section = 'start';
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
    page = await context.newPage();
    page.setDefaultTimeout(T);
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.addInitScript((key) => { try { localStorage.setItem('fv.key', key); } catch { /* no storage */ } }, KEY);
    for (const [name, f] of [['home', home], ['model page', modelPage], ['references', references], ['clip director', clipDirector],
      ['causal director', causalDirector], ['live stream', (p, o) => liveStream(p, o, whip)], ['native API', nativePage]]) {
      section = name;
      await f(page, srv.origin);
    }
    if (errors.length) throw new Error('browser errors:\n' + errors.join('\n'));
    console.log('ui gaps: OK');
  } catch (e) {
    if (page) {
      try {
        const log = await page.evaluate(() => ['#director-log', '#stream-log', '#director-msg', '#stream-msg', '#run-msg', '#result-msg']
          .map((s) => document.querySelector(s)).filter(Boolean).map((n) => n.textContent.slice(-6000)).join('\n'));
        process.stderr.write('--- failed in: ' + section + ' (' + page.url() + ') ---\n' + log + '\n');
      } catch { /* page gone */ }
    }
    process.stderr.write('--- fv-serve log ---\n' + srv.logs.slice(-4000) + '\n');
    throw e;
  } finally {
    await browser.close();
    whip.server.close();
    await stopServer(srv);
  }
})().catch((e) => { console.error(e); process.exit(1); });
