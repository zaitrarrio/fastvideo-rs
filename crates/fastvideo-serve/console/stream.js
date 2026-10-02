// Live stream: a causal model (SF-Wan, LongLive; `caps.stream.causal` in
// `GET /fv/v1/capabilities`) generating one continuous video that prompt
// switches steer, over one of two transports.
//
// Reactor runtime (when its `[reactor] model` is this model, `GET /schema`):
//   POST /start_session {seed, max_seconds} -> recvonly WebRTC (rtc.js
//   reactorWatch) -> `set_prompt` with the opening prompt, then
//   `set_prompt` / `set_paused` / `reset` / `get_state` (state_update every
//   second) on the `data` channel; POST /stop_session.
// Native stream (design §5.1-5.2, crates/fastvideo-serve/src/streams.rs):
//   POST /fv/v1/streams {model, whip_url, whip_token, whip_target, prompt,
//   seed, max_seconds} publishes H.264 to a WHIP endpoint (a relay such as
//   MediaMTX or Cloudflare Stream); this page plays the relay's WHEP URL.
//   GET /fv/v1/streams/{id} every second (state, pacer, TTFF, session),
//   POST …/commands {type: set_prompt | set_paused | reset | get_state},
//   DELETE /fv/v1/streams/{id}.
// A causal session always has a length (`stream_limits`: the default and
// the ceiling; a reset restarts the clock within the ceiling).

import {
  $, el, request, setMsg, topbar, loadAuthMode, needsKey, loadCapabilities, loadCatalog, mountedProtocols, streamKind,
  recipeText, licenceBanner, modelHref,
} from './common.js';
import { reactorWatch, reactorModel, whepPlay } from './rtc.js';

topbar('stream');

let models = [];
let reactorServes = null; // the model id the Reactor runtime streams
let protocols = null;
let apps = [];
let live = null; // {kind, close(), command(type, data), poll}
let version = 0;
const switches = new Map(); // version -> {text, li}

function log(line) {
  const box = $('stream-log');
  box.append(el('div', {}, new Date().toISOString().slice(11, 23) + ' ' + line));
  while (box.childElementCount > 200) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

function setState(text, kind) {
  const p = $('stream-state');
  p.textContent = text;
  p.className = 'pill' + (kind ? ' ' + kind : '');
  document.body.dataset.streamState = text;
}

const current = () => models.find((m) => m.caps.id === $('model').value);
const answers = (m, name) => !!name && (m.caps.id === name || (m.caps.served_names || []).includes(name));
// The fal app whose model is this one (its licence, its causal director).
const appFor = (m) => apps.find((a) => answers(m, a.model));

function showStats(rows) {
  $('stream-stats').replaceChildren(...rows.filter(([, v]) => v !== null && v !== undefined && v !== '')
    .flatMap(([k, v]) => [el('dt', {}, k), el('dd', { 'data-stat': k }, v)]));
}

function showModel() {
  const m = current();
  $('licence').replaceChildren();
  $('director-link').hidden = true;
  if (!m) { $('model-facts').textContent = ''; return; }
  const c = m.caps.stream.causal || {};
  const l = m.stream_limits;
  const parts = [];
  if (c.block_frames) parts.push(c.block_frames + '-frame blocks at ' + (c.target_fps || '?') + ' fps');
  if (m.caps.canvas && m.caps.canvas.short_edges) parts.push(m.caps.canvas.short_edges.join('/') + 'p');
  if (m.caps.recipe) parts.push('recipe ' + m.caps.recipe + (recipeText(m.recipe) ? ' (' + recipeText(m.recipe) + ')' : ''));
  if (l) parts.push(l.default_max_s + ' s by default, at most ' + l.hard_max_s + ' s' + (l.reset_restarts_clock ? ' (a reset restarts the clock)' : ''));
  $('model-facts').textContent = parts.join(' · ');
  const app = appFor(m);
  const licence = m.licence || (app && app.licence);
  if (licence) $('licence').replaceChildren(licenceBanner(licence));
  if (app && app.director_mode === 'causal') {
    $('director-link').hidden = false;
    $('director-link').replaceChildren('Also as a fal director session (scripts, timeline): ', el('a', { href: modelHref(app.id, 'director') }, app.id + '/director'), '.');
  }
  // Transports: the Reactor runtime only when it streams this model; the
  // native stream when the build can publish.
  const reactorOk = answers(m, reactorServes);
  const whipOk = !protocols || protocols.streams !== false;
  const t = $('transport');
  t.querySelector('[value="reactor"]').disabled = !reactorOk;
  t.querySelector('[value="whip"]').disabled = !whipOk;
  if (t.selectedOptions[0] && t.selectedOptions[0].disabled) t.value = reactorOk ? 'reactor' : 'whip';
  $('transport-note').textContent = !reactorOk
    ? (reactorServes ? 'The Reactor runtime streams ' + reactorServes + ' on this server ([reactor] model), not this model.' : 'This server does not mount the Reactor runtime.')
      + (whipOk ? '' : ' This build cannot publish native streams either (it needs the webrtc and http-client features).')
    : '';
  showTransport();
}

function showTransport() {
  $('whip-fields').hidden = $('transport').value !== 'whip';
}

async function load() {
  const mode = await loadAuthMode();
  $('key-banner').hidden = !needsKey();
  void mode;
  const caps = await loadCapabilities();
  if (!caps) {
    setMsg('stream-msg', 'Could not read GET /fv/v1/capabilities (is the API key set and the native API mounted?).', 'bad');
    return;
  }
  protocols = mountedProtocols(caps);
  models = (caps.models || []).filter((m) => streamKind(m) === 'causal');
  reactorServes = protocols && protocols.reactor === false ? null : await reactorModel();
  try { apps = (await loadCatalog()).apps || []; } catch { apps = []; }
  $('model').replaceChildren(...models.map((m) => el('option', { value: m.caps.id }, m.caps.id + ((m.caps.served_names || []).filter((n) => n !== m.caps.id).length ? ' (' + m.caps.served_names.filter((n) => n !== m.caps.id).join(', ') + ')' : ''))));
  const want = new URLSearchParams(location.search).get('model');
  const pick = models.find((m) => answers(m, want));
  if (pick) $('model').value = pick.caps.id;
  if (!models.length) setMsg('stream-msg', 'This server serves no causal streaming model (SF-Wan, LongLive).', 'bad');
  showModel();
}

// ---- timeline ----------------------------------------------------------------

function addSwitch(text, st) {
  version += 1;
  const li = el('li', { 'data-version': String(version) }, el('span', { class: 'v' }, 'v' + version), el('span', { class: 't' }, text), el('span', { class: 'pill' }, st));
  switches.set(version, { text, li });
  $('timeline').prepend(li);
  return version;
}
function markSwitch(v, st, kind) {
  const s = switches.get(v);
  if (!s) return;
  const pill = s.li.lastChild;
  pill.textContent = st; pill.className = 'pill' + (kind ? ' ' + kind : '');
  s.state = st;
}
// The session reports its prompt: every switch up to the one showing is applied.
function promptNow(prompt) {
  let hit = null;
  for (const [v, s] of switches) if (s.text === prompt) hit = v;
  if (hit === null) return;
  for (const [v, s] of switches) if (v <= hit && s.state !== 'applied' && !String(s.state).startsWith('refused')) markSwitch(v, 'applied', 'ok');
}

// ---- transports --------------------------------------------------------------

function attachVideo() {
  const out = new MediaStream();
  $('video').srcObject = out;
  return (track) => { out.addTrack(track); log('← ' + track.kind + ' track'); };
}

function stateRows(s) {
  if (!s) return [];
  return [
    ['prompt', s.prompt],
    ['paused', s.paused ? 'yes' : 'no'],
    ['block', s.block_index != null ? String(s.block_index) : null],
    ['generation', s.unique_fps != null ? Number(s.unique_fps).toFixed(1) + ' fps' : null],
    ['seed', s.seed != null ? String(s.seed) : null],
  ];
}

async function startReactor(m, prompt, params) {
  const onTrack = attachVideo();
  const w = await reactorWatch({
    params,
    onTrack,
    log,
    onState: (s) => {
      log('peer ' + s);
      if (s === 'connected') setState('streaming', 'ok');
      if (s === 'failed') stop('the peer connection failed');
    },
    onMessage: (msg) => {
      if (msg.type === 'state_update') {
        const s = msg.data || {};
        showStats([['transport', 'Reactor runtime'], ['model', m.caps.id], ...stateRows(s)]);
        document.body.dataset.block = String(s.block_index || 0);
        if (s.prompt) promptNow(s.prompt);
      } else if (msg.type === 'command_error') {
        const d = msg.data || {};
        log('← ' + d.command + ' refused: ' + d.reason);
        for (const [v, s] of switches) if (s.state === 'sent') markSwitch(v, 'refused: ' + d.reason, 'bad');
        setMsg('stream-msg', d.command + ': ' + d.reason, 'bad');
      } else if (msg.type === 'sessionEnded') {
        stop('session ended: ' + ((msg.data && msg.data.reason) || ''));
      }
    },
  });
  w.command('set_prompt', { prompt });
  const poll = setInterval(() => w.command('get_state', {}), 1000);
  return {
    kind: 'reactor', poll,
    command: async (type, data) => { w.command(type, data || {}); log('→ ' + type + (data && data.prompt ? ' ' + JSON.stringify(data.prompt) : '')); return null; },
    close: () => w.close(),
  };
}

async function startWhip(m, prompt, params) {
  const whip = $('whip-url').value.trim();
  if (!whip) throw new Error('enter the WHIP endpoint the server publishes to');
  const body = { model: m.caps.id, whip_url: whip, prompt, ...params };
  if ($('whip-token').value.trim()) body.whip_token = $('whip-token').value.trim();
  if ($('whip-target').value) body.whip_target = $('whip-target').value;
  const s = await request('POST', '/fv/v1/streams', { auth: 'bearer', body });
  const path = '/fv/v1/streams/' + encodeURIComponent(s.id);
  document.body.dataset.stream = s.id;
  log('→ POST /fv/v1/streams: ' + s.id + ' (' + s.mode + ', ' + s.max_seconds + ' s)');
  let player = null;
  const whep = $('whep-url').value.trim();
  if (whep) {
    try {
      player = await whepPlay(whep, { onTrack: attachVideo(), onState: (st) => log('WHEP ' + st) });
    } catch (e) { log('WHEP: ' + e.message); setMsg('stream-msg', 'Playback: ' + e.message, 'bad'); }
  }
  const poll = setInterval(async () => {
    try {
      const v = await request('GET', path, { auth: 'bearer' });
      const st = v.status || {};
      const p = v.pacer || {};
      const t = v.ttff || null;
      showStats([
        ['transport', 'native stream → WHIP'],
        ['stream', v.id],
        ['state', st.state + (st.end_reason ? ' (' + st.end_reason + ')' : '')],
        ['error', st.error],
        ['output', [st.video_codec, st.encoder].filter(Boolean).join(' · ') + (st.frames_sent ? ' · ' + st.frames_sent + ' frames, ' + st.keyframes_sent + ' keyframes sent' : '')],
        ['pacer', p.ticks != null ? Number(p.effective_fps || 0).toFixed(1) + ' fps out, ' + Number(p.unique_fps || 0).toFixed(1) + ' fps generated, '
          + Number(p.video_seconds || 0).toFixed(1) + ' / ' + (v.max_seconds || '?') + ' s, ' + (p.underruns || 0) + ' underruns' : null],
        ['first frame', st.first_frame_ms != null ? st.first_frame_ms + ' ms' : null],
        ['ttff', t ? JSON.stringify(t) : null],
        ...stateRows(v.session),
      ]);
      setState(st.state === 'streaming' ? 'streaming' : st.state, st.state === 'streaming' ? 'ok' : st.state === 'closed' ? '' : 'warn');
      if (v.session && v.session.prompt) promptNow(v.session.prompt);
      if (st.state === 'closed') stop('the stream ended: ' + (st.end_reason || 'closed') + (st.error ? ' (' + st.error + ')' : ''), true);
    } catch (e) { log('status: ' + e.message); }
  }, 1000);
  return {
    kind: 'whip', poll,
    command: async (type, data) => {
      log('→ ' + type + (data && data.prompt ? ' ' + JSON.stringify(data.prompt) : ''));
      const r = await request('POST', path + '/commands', { auth: 'bearer', body: data ? { type, data } : { type } });
      const reply = r && r.reply;
      if (reply && reply.type === 'command_error') throw new Error((reply.data && reply.data.reason) || 'refused');
      if (reply && reply.type === 'state_update' && reply.data && reply.data.prompt) promptNow(reply.data.prompt);
      return reply;
    },
    close: async (ended) => {
      if (player) await player.close();
      if (!ended) { try { await request('DELETE', path, { auth: 'bearer' }); } catch (e) { log('DELETE: ' + e.message); } }
    },
  };
}

// ---- page ----------------------------------------------------------------------

function controls(on) {
  $('start').disabled = on;
  for (const id of ['stop', 'switch', 'pause', 'reset']) $(id).disabled = !on;
}

async function start() {
  if (live) return;
  setMsg('stream-msg', '');
  const m = current();
  if (!m) { setMsg('stream-msg', 'Pick a causal model.', 'bad'); return; }
  await loadAuthMode();
  if (needsKey()) { setMsg('stream-msg', 'Set an API key first.', 'bad'); return; }
  const prompt = $('prompt').value.trim();
  if (!prompt) { setMsg('stream-msg', 'Enter an opening prompt.', 'bad'); return; }
  const params = {};
  if ($('seed').value.trim()) params.seed = Number($('seed').value.trim());
  if ($('max-seconds').value) params.max_seconds = Number($('max-seconds').value);
  switches.clear(); $('timeline').replaceChildren(); version = 0; $('stream-stats').replaceChildren();
  setState('starting', 'warn');
  controls(true);
  $('stop').disabled = true;
  try {
    const v = addSwitch(prompt, 'sent');
    live = $('transport').value === 'whip' ? await startWhip(m, prompt, params) : await startReactor(m, prompt, params);
    live.paused = false;
    markSwitch(v, 'sent', 'warn');
    controls(true);
    setState('connecting', 'warn');
  } catch (e) {
    setState('failed', 'bad');
    setMsg('stream-msg', e.message, 'bad');
    controls(false);
    live = null;
  }
}

async function stop(reason, ended) {
  const l = live;
  live = null;
  if (!l) return;
  clearInterval(l.poll);
  await l.close(ended);
  const v = $('video').srcObject;
  if (v && v.getTracks) v.getTracks().forEach((t) => t.stop());
  $('video').srcObject = null;
  setState('stopped');
  if (reason) setMsg('stream-msg', reason);
  controls(false);
  $('pause').textContent = 'Pause';
}

$('start').onclick = start;
$('stop').onclick = () => stop('stopped');
$('switch').onclick = async () => {
  const text = $('next').value.trim();
  if (!text || !live) return;
  const v = addSwitch(text, 'sent');
  markSwitch(v, 'sent', 'warn');
  try {
    await live.command('set_prompt', { prompt: text });
    $('next').value = '';
  } catch (e) { markSwitch(v, 'refused: ' + e.message, 'bad'); setMsg('stream-msg', e.message, 'bad'); }
};
$('next').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) $('switch').click(); });
$('pause').onclick = async () => {
  if (!live) return;
  live.paused = !live.paused;
  try { await live.command('set_paused', { paused: live.paused }); } catch (e) { setMsg('stream-msg', e.message, 'bad'); }
  $('pause').textContent = live.paused ? 'Resume' : 'Pause';
};
$('reset').onclick = async () => {
  if (!live) return;
  try { await live.command('reset', null); log('reset: the rollout restarts from the current prompt'); } catch (e) { setMsg('stream-msg', e.message, 'bad'); }
};
$('model').onchange = showModel;
$('transport').onchange = showTransport;
window.addEventListener('beforeunload', () => { if (live) live.close(); });

load();
