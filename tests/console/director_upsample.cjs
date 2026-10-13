// Director prompt upsampling in headless Chromium (docs/serve/prompt-upsampling.md):
// the console's director page against fv-serve's fake engine and a fake
// OpenAI-compatible LLM (a local HTTP server answering /v1/chat/completions
// with `segment` JSON). Checks:
//   - session_info advertises the upsampler; the form's toggle, style and
//     entities go out in `configure`;
//   - the LLM gets the spec's request (bearer key, model, strict
//     json_schema `segment`, the premise as the world bible);
//   - each chunk's `segment_expanded` shows in the page, with the entity's
//     descriptor and the style spliced into the prompt;
//   - a direction whose call fails falls back to the direction's text
//     (shown as `fallback`) and the stream goes on.
//
//   FV_SERVE_BIN=target/debug/fv-serve node tests/console/director_upsample.cjs
//
// tests/console/run.sh runs it (fv-serve built with `full`, which has the
// HTTP client the upsampler needs).

'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');

const BIN = process.env.FV_SERVE_BIN || path.join(__dirname, '..', '..', 'target', 'debug', 'fv-serve');

function step(msg) { process.stdout.write('  - ' + msg + '\n'); }
function check(cond, what, s) {
  if (!cond) throw new Error('director upsample: ' + what + (s ? ' :: ' + JSON.stringify(s).slice(0, 2000) : ''));
}

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
    s.on('error', reject);
  });
}

// The fake LLM: every call answers the next beat; a direction containing
// "please fail" gets a 500 (both hedged attempts), so the director falls back.
function startLlm() {
  const llm = { calls: [], n: 0 };
  llm.server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (d) => { body += d; });
    req.on('end', () => {
      let j = null;
      try { j = JSON.parse(body); } catch { /* recorded as null */ }
      llm.calls.push({ path: req.url, auth: req.headers.authorization || null, body: j });
      // Only chat completions (fv-serve's `/tokenize` feature check gets a 404).
      if (req.url !== '/v1/chat/completions') {
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end('{"error":"not found"}');
        return;
      }
      const user = (j && j.messages && j.messages[1] && j.messages[1].content) || '';
      if (/please fail/i.test(user.split('NEW DIRECTION:').pop())) {
        res.writeHead(500, { 'content-type': 'application/json' });
        res.end('{"error":"boom"}');
        return;
      }
      const n = llm.n++;
      const seg = {
        beat_summary: 'beat ' + n,
        entity_ids: ['mara', 'tower'],
        new_entities: n === 0 ? [{ id: 'gull', kind: 'prop', name: 'Gull', descriptor: 'a grey herring gull' }] : [],
        action: 'Mara climbs the stairs, step ' + n + '.',
        camera: 'slow tracking shot',
        lighting: 'lamp glow',
        audio: 'wind and rain',
        dialogue: [],
      };
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: JSON.stringify(seg) } }] }));
    });
  });
  return new Promise((resolve) => llm.server.listen(0, '127.0.0.1', () => { llm.port = llm.server.address().port; resolve(llm); }));
}

async function startServer(llmPort) {
  const port = await freePort();
  const origin = 'http://127.0.0.1:' + port;
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-director-upsample-'));
  const cfg = path.join(state, 'fv.toml');
  fs.writeFileSync(cfg, '[engine.fake]\nrtf = 0.3\n\n[director]\nchunk_seconds = 5.0\n');
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('FV_') && k !== 'FV_SERVE_BIN') delete env[k];
  Object.assign(env, {
    FV_BIND: '127.0.0.1:' + port, FV_STATE_DIR: state, FV_JOB_STORE: 'memory', FV_ENGINE: 'fake',
    FV_URL_SIGNING_KEY: 'director-upsample', FV_AUTH_MODE: 'none', RUST_LOG: 'warn,fastvideo_fal=info',
    FV_UPSAMPLER_BASE_URL: 'http://127.0.0.1:' + llmPort + '/v1', FV_UPSAMPLER_MODEL: 'fake-llm',
    FV_UPSAMPLER_API_KEY: 'test-key', FV_UPSAMPLER_TIMEOUT_MS: '5000',
    NO_PROXY: '127.0.0.1,localhost', no_proxy: '127.0.0.1,localhost',
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

const segs = (page) => page.evaluate(() => [...document.querySelectorAll('#director-segments li')].map((li) => ({
  text: li.textContent, prompt: li.querySelector('.segment-prompt').textContent, fallback: li.dataset.fallback === 'true',
})));

(async () => {
  const llm = await startLlm();
  const srv = await startServer(llm.port);
  const browser = await chromium.launch();
  let page;
  try {
    page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(srv.origin + '/console/models/minimax/h3-turbo/director');
    await page.waitForSelector('#director-start');
    await page.fill('#director-prompt', 'A lighthouse keeper on a stormy island.');
    await page.click('#director-upsampling summary');
    await page.check('#director-upsample');
    await page.fill('#director-style', '35mm film, muted teal palette.');
    await page.fill('#director-entities', JSON.stringify([
      { id: 'mara', kind: 'character', name: 'Mara', descriptor: 'a wiry woman in a yellow oilskin coat' },
      { id: 'tower', kind: 'location', name: 'Lighthouse', descriptor: 'a white lighthouse on black rocks' },
    ]));
    await page.click('#director-start');
    await page.waitForFunction(() => document.querySelector('#director-state').textContent === 'streaming', null, { timeout: 60_000 });

    // The first segment (chunk 0) shows in the page.
    await page.waitForFunction(() => document.querySelectorAll('#director-segments li').length >= 1, null, { timeout: 60_000 });
    let s = await segs(page);
    const first = s[s.length - 1];
    check(!first.fallback && /chunk 0/.test(first.text) && /beat 0/.test(first.text), 'chunk 0 segment', s);
    // H3: the Context-IR layout (docs/serve/prompt-upsampling.md §Assembly).
    check(first.prompt.startsWith('integrated_multimodal_description: a white lighthouse on black rocks. Mara: a wiry woman in a yellow oilskin coat. Mara climbs the stairs, step 0.'), 'location, cast and action lead the description', first);
    check(first.prompt.endsWith('35mm film, muted teal palette.\noverall_soundscape: wind and rain No spoken words.\nnon_diegetic_music: None.'), 'style closes the description, then soundscape and music', first);
    check(/planned/.test(first.text), 'the planned count is shown', first);
    check(/New: Gull \(prop\)/.test(first.text), 'the new entity is listed', first);
    step('chunk 0 upsampled: "' + first.prompt.slice(0, 90) + '…"');

    // The LLM saw the spec's request.
    const chat = llm.calls.filter((c) => c.path === '/v1/chat/completions');
    const c0 = chat[0];
    check(c0.path === '/v1/chat/completions' && c0.auth === 'Bearer test-key', 'endpoint and bearer key', c0);
    check(c0.body.model === 'fake-llm' && c0.body.response_format.type === 'json_schema' && c0.body.response_format.json_schema.name === 'segment', 'model and schema', c0.body);
    check(c0.body.messages[1].content.startsWith('WORLD BIBLE:\nA lighthouse keeper on a stormy island.'), 'the premise is the world bible', c0.body.messages[1]);
    check(/- mara \(character, "Mara"\)/.test(c0.body.messages[1].content), 'the configured cast is in the registry', c0.body.messages[1]);
    const facts = await page.textContent('#director-session-info');
    check(/prompt upsampler/.test(facts) && /available/.test(facts) && /h3 layout/.test(facts), 'session info lists the upsampler', facts);
    check(c0.body.messages[0].content.includes('"music" is the non-diegetic score'), 'the music rule is in the system prompt', c0.body.messages[0]);
    step('LLM request: bearer key, model fake-llm, json_schema `segment`, ' + llm.calls.length + ' call(s) so far');

    // A direction whose call fails: its text is used (never the premise), shown as fallback.
    await page.fill('#director-next', 'Please fail now.');
    await page.click('#director-send');
    await page.waitForFunction(() => [...document.querySelectorAll('#director-segments li')].some((li) => li.dataset.fallback === 'true'), null, { timeout: 90_000 });
    s = await segs(page);
    const fb = s.find((x) => x.fallback);
    check(fb.prompt === 'Please fail now.' && /upsampler LLM 500/.test(fb.text), 'the fallback is the direction', fb);
    step('failed direction fell back to "' + fb.prompt + '"');

    // The stream goes on: a later chunk is upsampled again.
    await page.fill('#director-next', 'The storm clears at dawn.');
    await page.click('#director-send');
    await page.waitForFunction(() => {
      const li = [...document.querySelectorAll('#director-segments li')];
      return li.length >= 3 && li[0].dataset.fallback === 'false';
    }, null, { timeout: 90_000 });
    const chats = llm.calls.filter((c) => c.body && c.body.messages);
    const dirCall = chats.find((c) => /NEW DIRECTION:\nThe storm clears at dawn\.$/.test(c.body.messages[1].content));
    check(dirCall, 'the direction reached the LLM', chats.map((c) => c.body.messages[1].content.split('NEW DIRECTION:').pop()));
    step('upsampling resumed after the fallback (' + (await segs(page)).length + ' segments, ' + llm.calls.length + ' LLM calls)');

    await page.click('#director-stop');
    await page.waitForFunction(() => document.querySelector('#director-state').textContent === 'closed', null, { timeout: 30_000 });
    check(!errors.length, 'no page errors', errors);
    step('stopped');
  } catch (e) {
    process.stdout.write(srv.logs.slice(-4000) + '\n');
    if (page) await page.screenshot({ path: path.join(os.tmpdir(), 'director-upsample-failure.png') }).catch(() => {});
    throw e;
  } finally {
    await browser.close();
    await stopServer(srv);
    llm.server.close();
  }
})().catch((e) => { console.error(e); process.exit(1); });
