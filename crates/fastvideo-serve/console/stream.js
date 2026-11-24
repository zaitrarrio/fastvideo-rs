// Live stream: a causal model (SF-Wan, LongLive; `caps.stream.causal` in
// `GET /fv/v1/capabilities`) generating one continuous video that prompt
// switches steer, over one of two transports; or a clip model (H3, LTX, Wan;
// `caps.stream.clip`) playing a queue of clips over the native stream only:
// the opening prompt is the first clip, "Enqueue clip" adds one (`enqueue`
// {prompt, seconds}), Pause / Resume toggle autoplay (`set_autoplay`), Reset
// drops the queue (`reset`); the stream object's `session` is the clip
// player's state and its `events` (clip_started, clip_finished) mark the
// timeline.
//
// Reactor runtime (when its `[reactor] model` is this model, `GET /schema`):
//   POST /start_session {seed, max_seconds} -> recvonly WebRTC (rtc.js
//   reactorWatch) -> `set_prompt` with the opening prompt, then
//   `set_prompt` / `set_paused` / `reset` / `get_state` (state_update every
//   second) on the `data` channel; POST /stop_session.
// Native stream (design §5.1-5.2, crates/fastvideo-serve/src/streams.rs,
// docs/serve/stream-outputs.md):
//   POST /fv/v1/streams {model, output, prompt, seed, max_seconds}: the
//   destination form (outputs.js) builds `output` (Cloudflare Stream by
//   default, WHIP, RTMPS, SRT; credentials never stored here); the stream
//   object's `playback` picks the player: WHEP for WebRTC outputs, HLS for
//   RTMPS / SRT, else the playback URL pasted in the form.
//   GET /fv/v1/streams/{id} every second (state, pacer, TTFF, session),
//   POST …/commands {type: set_prompt | set_paused | reset | get_state},
//   DELETE /fv/v1/streams/{id}.
// A causal session always has a length (`stream_limits`: the default and
// the ceiling; a reset restarts the clock within the ceiling).

import {
  $, el, request, setMsg, topbar, loadAuthMode, needsKey, loadCapabilities, loadCatalog, mountedProtocols, streamKind,
  recipeText, licenceBanner, modelHref, admit, BUSY,
} from './common.js';
import { reactorWatch, reactorModel } from './rtc.js';
import { outputForm, loadOutputs, playOutput } from './outputs.js';

topbar('stream');

let models = [];
let reactorServes = null; // the model id the Reactor runtime streams
let protocols = null;
let apps = [];
let live = null; // {kind, close(), command(type, data), poll}
const dest = outputForm('out');
$('native-fields').append(dest.node);
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
const kindOf = (m) => (m ? streamKind(m) : null);
const isClip = () => kindOf(current()) === 'clip';
const answers = (m, name) => !!name && (m.caps.id === name || (m.caps.served_names || []).includes(name));
// The fal app whose model is this one (its licence, its causal director).
const appFor = (m) => apps.find((a) => answers(m, a.model));

// The engine is still busy (a session that just ended releases it a moment later).
function waiting(s) {
  setState('waiting', 'warn');
  setMsg('stream-msg', 'The engine is still busy (another streaming session is ending): retrying (' + s + ' s).');
}

function showStats(rows) {
  $('stream-stats').replaceChildren(...rows.filter(([, v]) => v !== null && v !== undefined && v !== '')
    .flatMap(([k, v]) => [el('dt', {}, k), el('dd', { 'data-stat': k }, v)]));
}

function showModel() {
  const m = current();
  $('licence').replaceChildren();
  $('director-link').hidden = true;
  if (!m) { $('model-facts').textContent = ''; return; }
  const clip = kindOf(m) === 'clip';
  const c = m.caps.stream.causal || {};
  const k = m.caps.stream.clip || {};
  const l = m.stream_limits;
  const parts = [];
  if (clip) parts.push('clip queue' + (k.min_s != null ? ', clips of ' + k.min_s + ' to ' + k.max_s + ' s' : ''));
  if (c.block_frames) parts.push(c.block_frames + '-frame blocks at ' + (c.target_fps || '?') + ' fps');
  if (m.caps.canvas && m.caps.canvas.short_edges) parts.push(m.caps.canvas.short_edges.join('/') + 'p');
  if (m.caps.recipe) parts.push('recipe ' + m.caps.recipe + (recipeText(m.recipe) ? ' (' + recipeText(m.recipe) + ')' : ''));
  if (l) parts.push(l.default_max_s + ' s by default, at most ' + l.hard_max_s + ' s' + (l.reset_restarts_clock ? ' (a reset restarts the clock)' : ''));
  $('model-facts').textContent = parts.join(' · ');
  // The controls read as the session kind does.
  $('length-label').textContent = clip ? 'Clip length (s)' : 'Length (s)';
  $('max-seconds').placeholder = clip ? "the model's default" : 'server default';
  $('switch').textContent = clip ? 'Enqueue clip' : 'Switch prompt';
  $('next-hint').textContent = clip
    ? 'Queued behind the clips already generating; each clip starts from the last frame of the one before (a scene cut on Reset).'
    : 'Applied at the next block boundary (a LongLive model re-caches its KV window once per switch).';
  if (!live) $('pause').textContent = 'Pause';
  const app = appFor(m);
  const licence = m.licence || (app && app.licence);
  if (licence) $('licence').replaceChildren(licenceBanner(licence));
  if (app && app.director_mode === 'causal') {
    $('director-link').hidden = false;
    $('director-link').replaceChildren('Also as a fal director session (scripts, timeline): ', el('a', { href: modelHref(app.id, 'director') }, app.id + '/director'), '.');
  }
  // Transports: the Reactor runtime only when it streams this model; the
  // native stream when the build can publish.
  // A clip model streams natively only (the Reactor runtime runs causal models).
  const reactorOk = !clip && answers(m, reactorServes);
  const whipOk = !protocols || protocols.streams !== false;
  const t = $('transport');
  t.querySelector('[value="reactor"]').disabled = !reactorOk;
  t.querySelector('[value="native"]').disabled = !whipOk;
  if (t.selectedOptions[0] && t.selectedOptions[0].disabled) t.value = reactorOk ? 'reactor' : 'native';
  $('transport-note').textContent = !reactorOk
    ? (clip ? 'A clip model streams as a native stream (a clip queue published to an output).'
      : reactorServes ? 'The Reactor runtime streams ' + reactorServes + ' on this server ([reactor] model), not this model.' : 'This server does not mount the Reactor runtime.')
      + (whipOk ? '' : ' This build cannot publish native streams either (it needs the webrtc and http-client features).')
    : '';
  showTransport();
}

function showTransport() {
  $('native-fields').hidden = $('transport').value !== 'native';
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
  models = (caps.models || []).filter((m) => ['causal', 'clip'].includes(streamKind(m)));
  reactorServes = protocols && protocols.reactor === false ? null : await reactorModel();
  dest.setInfo(await loadOutputs());
  try { apps = (await loadCatalog()).apps || []; } catch { apps = []; }
  $('model').replaceChildren(...models.map((m) => el('option', { value: m.caps.id }, m.caps.id + ' · ' + streamKind(m) + ((m.caps.served_names || []).filter((n) => n !== m.caps.id).length ? ' (' + m.caps.served_names.filter((n) => n !== m.caps.id).join(', ') + ')' : ''))));
  const want = new URLSearchParams(location.search).get('model');
  const pick = models.find((m) => answers(m, want));
  if (pick) $('model').value = pick.caps.id;
  if (!models.length) setMsg('stream-msg', 'This server serves no streaming model (causal: SF-Wan, LongLive; clip: H3, LTX, Wan).', 'bad');
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

// A clip stream's events mark the queued clips: started → playing, finished → played.
const seenEvents = new Set();
function clipEvents(events) {
  for (const e of events || []) {
    const clip = e && e.data && e.data.clip;
    if (!clip || !clip.clip_id) continue;
    const key = e.type + ':' + clip.clip_id;
    if (seenEvents.has(key)) continue;
    seenEvents.add(key);
    const st = e.type === 'clip_started' ? 'playing' : e.type === 'clip_finished' ? 'played' : e.type === 'clip_generated' ? 'generated' : null;
    if (!st) continue;
    for (const [v, sw] of switches) {
      if (sw.text === clip.prompt && sw.state !== 'played' && !String(sw.state).startsWith('refused')) { markSwitch(v, st, st === 'played' ? '' : 'ok'); break; }
    }
  }
}

// ---- transports --------------------------------------------------------------

function attachVideo() {
  const out = new MediaStream();
  $('video').srcObject = out;
  return (track) => { out.addTrack(track); log('← ' + track.kind + ' track'); };
}

function stateRows(s) {
  if (!s) return [];
  if (s.clip_seconds != null || s.playout_queued != null) {
    return [
      ['playing', s.playing ? 'yes' + (s.playing_clip_id ? ' (' + String(s.playing_clip_id).slice(0, 8) + ')' : '') : 'no'],
      ['queue', (s.generation_queued || 0) + ' generating, ' + (s.playout_queued || 0) + ' ready to play, ' + (s.clips_played || 0) + ' played'],
      ['clip length', s.clip_seconds != null ? Number(s.clip_seconds).toFixed(1) + ' s (' + s.clip_seconds_min + ' to ' + s.clip_seconds_max + ')' : null],
      ['autoplay', s.autoplay ? 'on' : 'off'],
      ['canvas', s.width ? s.width + '×' + s.height + (s.aspect ? ' ' + s.aspect : '') : null],
      ['seed', s.seed != null ? String(s.seed) : null],
    ];
  }
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
    onWait: waiting,
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

async function startNative(m, prompt, params) {
  const output = dest.read();
  const clip = kindOf(m) === 'clip';
  // Causal: the opening prompt and the session length. Clip: the first clip
  // (with its length when given); `max_seconds` is not a clip session's.
  const body = clip
    ? { model: m.caps.id, ...(params.max_seconds ? { clips: [{ prompt, seconds: params.max_seconds }] } : { prompt }), ...(params.seed != null ? { seed: params.seed } : {}), output }
    : { model: m.caps.id, prompt, ...params, output };
  const s = await admit(() => request('POST', '/fv/v1/streams', { auth: 'bearer', body }), { isBusy: (e, failed) => failed && BUSY.has(e.status), onWait: waiting });
  const path = '/fv/v1/streams/' + encodeURIComponent(s.id);
  document.body.dataset.stream = s.id;
  document.body.dataset.output = (s.output && s.output.kind) || '';
  log('→ POST /fv/v1/streams: ' + s.id + ' (' + s.mode + ', ' + s.max_seconds + ' s) → ' + ((s.output && s.output.kind) || 'output'));
  // The player follows the output's format (WHEP for WebRTC, HLS for RTMPS / SRT).
  // It connects in the background: a clip model's first frame comes seconds
  // after the stream opens (its first clip builds), and the output is not
  // playable before it, so the commands and the status poll must not wait.
  let player = null;
  const playing = playOutput($('video'), s.playback, dest.pasted(), { log }).then((p) => {
    player = p;
    document.body.dataset.player = p.kind || 'none';
    $('playback-info').textContent = p.kind ? 'Player: ' + p.kind.toUpperCase() + ' · ' + p.url : 'Player: ' + p.why;
  }).catch((e) => {
    document.body.dataset.player = 'failed';
    log('playback: ' + e.message);
    setMsg('stream-msg', 'Playback: ' + e.message, 'bad');
  });
  const poll = setInterval(async () => {
    try {
      const v = await request('GET', path, { auth: 'bearer' });
      const st = v.status || {};
      const p = v.pacer || {};
      const t = v.ttff || null;
      const o = v.output || {};
      showStats([
        ['transport', 'native stream → ' + (o.kind || 'output') + (o.url ? ' (' + o.url + ')' : '')],
        ['stream', v.id],
        ['state', st.state + (st.end_reason ? ' (' + st.end_reason + ')' : '')],
        ['error', st.error],
        ['output', [st.output_state, st.video_codec, st.encoder, st.width ? st.width + '×' + st.height : null].filter(Boolean).join(' · ')
          + (st.frames_sent ? ' · ' + st.frames_sent + ' frames' + (st.keyframes_sent ? ', ' + st.keyframes_sent + ' keyframes' : '') + ' sent' : '')],
        ['output error', st.output_error],
        ['pacer', p.ticks != null ? Number(p.effective_fps || 0).toFixed(1) + ' fps out, ' + Number(p.unique_fps || 0).toFixed(1) + ' fps generated, '
          + Number(p.video_seconds || 0).toFixed(1) + ' / ' + (v.max_seconds || '?') + ' s, ' + (p.underruns || 0) + ' underruns' : null],
        ['first frame', st.first_frame_ms != null ? st.first_frame_ms + ' ms' : null],
        ['ttff', t ? Object.entries(t).filter(([, x]) => x != null).map(([k, x]) => k.replace(/_ms$/, '').replace(/_/g, ' ') + ' ' + Math.round(Number(x)) + ' ms').join(', ') : null],
        ...stateRows(v.session),
      ]);
      setState(st.state === 'streaming' ? 'streaming' : st.state, st.state === 'streaming' ? 'ok' : st.state === 'closed' ? '' : 'warn');
      if (v.session && v.session.prompt) promptNow(v.session.prompt);
      if (clip) clipEvents(v.events);
      if (st.state === 'closed') stop('the stream ended: ' + (st.end_reason || 'closed') + (st.error ? ' (' + st.error + ')' : ''), true);
    } catch (e) { log('status: ' + e.message); }
  }, 1000);
  return {
    kind: 'native', poll,
    command: async (type, data) => {
      log('→ ' + type + (data && data.prompt ? ' ' + JSON.stringify(data.prompt) : ''));
      const r = await request('POST', path + '/commands', { auth: 'bearer', body: data === undefined || data === null ? { type } : { type, data } });
      const reply = r && r.reply;
      if (reply && reply.type === 'command_error') throw new Error((reply.data && reply.data.reason) || 'refused');
      if (reply && reply.type === 'state_update' && reply.data && reply.data.prompt) promptNow(reply.data.prompt);
      return reply;
    },
    close: async (ended) => {
      await playing;
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
  if (!m) { setMsg('stream-msg', 'Pick a streaming model.', 'bad'); return; }
  if (kindOf(m) === 'clip' && $('transport').value !== 'native') $('transport').value = 'native';
  await loadAuthMode();
  if (needsKey()) { setMsg('stream-msg', 'Set an API key first.', 'bad'); return; }
  const prompt = $('prompt').value.trim();
  if (!prompt) { setMsg('stream-msg', 'Enter an opening prompt.', 'bad'); return; }
  const params = {};
  if ($('seed').value.trim()) params.seed = Number($('seed').value.trim());
  if ($('max-seconds').value) params.max_seconds = Number($('max-seconds').value);
  switches.clear(); seenEvents.clear(); $('timeline').replaceChildren(); version = 0; $('stream-stats').replaceChildren(); $('stream-log').replaceChildren();
  for (const k of ['stream', 'player', 'output']) delete document.body.dataset[k];
  delete $('video').dataset.hlsEngine;
  setState('starting', 'warn');
  controls(true);
  $('stop').disabled = true;
  try {
    const v = addSwitch(prompt, 'sent');
    live = $('transport').value === 'native' ? await startNative(m, prompt, params) : await startReactor(m, prompt, params);
    live.paused = false;
    markSwitch(v, 'sent', 'warn');
    controls(true);
    setState('connecting', 'warn');
    setMsg('stream-msg', '');
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
  $('playback-info').textContent = '';
  setState('stopped');
  if (reason) setMsg('stream-msg', reason);
  controls(false);
  $('pause').textContent = 'Pause';
}

$('start').onclick = start;
$('stop').onclick = () => stop('stopped');
$('switch').onclick = async () => {
  const text = $('next').value.trim();
  if (!live) { setMsg('stream-msg', 'The stream is still starting.'); return; }
  if (!text) return;
  const v = addSwitch(text, 'sent');
  markSwitch(v, 'sent', 'warn');
  try {
    if (isClip()) {
      const secs = Number($('max-seconds').value);
      await live.command('enqueue', secs > 0 ? { prompt: text, seconds: secs } : { prompt: text });
      markSwitch(v, 'queued', 'warn');
    } else await live.command('set_prompt', { prompt: text });
    $('next').value = '';
  } catch (e) { markSwitch(v, 'refused: ' + e.message, 'bad'); setMsg('stream-msg', e.message, 'bad'); }
};
$('next').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) $('switch').click(); });
$('pause').onclick = async () => {
  if (!live) return;
  live.paused = !live.paused;
  try {
    // Causal: pause the rollout. Clip: stop autoplay (the queue keeps generating; Resume plays on).
    if (isClip()) await live.command('set_autoplay', !live.paused);
    else await live.command('set_paused', { paused: live.paused });
  } catch (e) { setMsg('stream-msg', e.message, 'bad'); }
  $('pause').textContent = live.paused ? 'Resume' : 'Pause';
};
$('reset').onclick = async () => {
  if (!live) return;
  try { await live.command('reset', null); log(isClip() ? 'reset: the queue is dropped (the next clip is a scene cut)' : 'reset: the rollout restarts from the current prompt'); } catch (e) { setMsg('stream-msg', e.message, 'bad'); }
};
$('model').onchange = showModel;
$('transport').onchange = showTransport;
window.addEventListener('beforeunload', () => { if (live) live.close(); });

load();
