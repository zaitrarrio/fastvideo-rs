// fal director (WMA) client for the console: design §5.6, research-fal §8.
//
// Everything that talks to the director lives in this module. Flow:
//   1. POST /wma/ice {app_id}                      -> {ice_servers}
//   2. RTCPeerConnection, client-created data channel "control", recvonly
//      video + audio, full (non-trickle) offer after ICE gathering
//   3. POST /wma/session {app_id, sdp, type:"offer"} -> {session_id, sdp, type:"answer"}
//   4. POST /wma/session/heartbeat {session_id} every 5 s -> {alive}
//   5. on channel open the server sends `session_info`; the client checks the
//      configure against it (`check(info, cfg)`; INFO_WAIT_MS at most: a
//      server that sends none gets the configure unchecked, logged), then
//      sends `configure` (prompt_version 1) and waits for `configured`
//   6. `prompt` messages with prompt_version 2, 3, … (replan true/false)
//   7. `stop` -> `stream_exhausted`, then close
// 404 / 405 / 501 on the signalling routes means this server does not
// mount the director (built without `webrtc`, or `fal_director` off): the
// page says so instead of failing.

import {
  el, request, setMsg, loadAuthMode, needsKey, poolBadge, poolWarning, loadCatalog, metaHeaders, draftPill, licenceBanner, upload,
} from './common.js';
import { gathered } from './rtc.js';

const HEARTBEAT_MS = 5000;
// How long `configure` waits for `session_info`. The server sends it the
// moment the control channel opens, but a busy machine can take seconds:
// a short wait sent the configure unchecked, and what the session cannot
// take came back as the server's own error instead of the page's check.
const INFO_WAIT_MS = 10000;
const UNAVAILABLE = new Set([404, 405, 501]);

function field(label, node, hint) {
  return el('div', {}, el('label', {}, label), node, hint ? el('p', { class: 'hint' }, hint) : null);
}

export class DirectorClient {
  constructor({ appId, onEvent, onState, onTrack, check }) {
    this.appId = appId;
    // `check(session_info, configure)` -> an error text that cancels the session, or null.
    this.check = check || null;
    this.onEvent = onEvent || (() => {});
    this.onState = onState || (() => {});
    this.onTrack = onTrack || (() => {});
    this.pc = null; this.dc = null; this.sessionId = null; this.hb = null;
    this.version = 0; this.configured = false; this.pendingConfigure = null;
  }

  // Whether the server answers the director signalling routes.
  static async probe(appId) {
    try {
      await request('POST', '/wma/ice', { auth: 'key', body: { app_id: appId } });
      return { available: true };
    } catch (e) {
      if (UNAVAILABLE.has(e.status)) return { available: false, reason: 'not-implemented' };
      if (e.status === 401) return { available: true, reason: 'unauthorized', message: e.message };
      return { available: false, reason: 'error', message: e.message };
    }
  }

  async start(configure) {
    this.onState('connecting');
    const ice = await request('POST', '/wma/ice', { auth: 'key', body: { app_id: this.appId } });
    const pc = new RTCPeerConnection({ iceServers: (ice && ice.ice_servers) || [] });
    this.pc = pc;
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    pc.ontrack = (ev) => this.onTrack(ev.streams[0] || new MediaStream([ev.track]));
    pc.onconnectionstatechange = () => {
      this.onEvent({ type: 'peer', state: pc.connectionState });
      if (pc.connectionState === 'failed') this.close('peer connection failed');
    };
    const dc = pc.createDataChannel('control');
    this.dc = dc;
    this.pendingConfigure = { ...configure, type: 'configure', prompt_version: 1, protocol_version: 1 };
    this.version = 1;
    this.info = null; this.configureSent = false;
    dc.onopen = () => { this.onState('configuring'); this.infoWait = setTimeout(() => {
      this.onEvent({ type: 'info_timeout', message: 'no session_info within ' + INFO_WAIT_MS / 1000 + ' s: configure sent unchecked' });
      this.sendConfigure();
    }, INFO_WAIT_MS); };
    dc.onmessage = (ev) => this.handle(ev.data);
    dc.onclose = () => this.close('control channel closed');

    await pc.setLocalDescription(await pc.createOffer());
    await gathered(pc);
    const r = await request('POST', '/wma/session', {
      auth: 'key', full: true, body: { app_id: this.appId, sdp: pc.localDescription.sdp, type: 'offer' },
    });
    const answer = r.body;
    // The model behind the session (`x-fv-model`, `x-fv-tier`, `x-fv-recipe`).
    this.meta = metaHeaders(r.headers);
    this.onEvent({ type: 'meta', ...this.meta });
    this.sessionId = answer.session_id;
    await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });
    this.onEvent({ type: 'session', session_id: this.sessionId });
    this.hb = setInterval(() => this.heartbeat(), HEARTBEAT_MS);
  }

  async heartbeat() {
    if (!this.sessionId) return;
    try {
      const r = await request('POST', '/wma/session/heartbeat', { auth: 'key', body: { session_id: this.sessionId } });
      if (r && r.alive === false) this.close('session ended by the server');
    } catch (e) { this.onEvent({ type: 'heartbeat_error', message: e.message }); }
  }

  // `configure`, once: after `session_info` (or its timeout) and the check.
  sendConfigure() {
    if (this.configureSent || !this.dc) return;
    this.configureSent = true;
    clearTimeout(this.infoWait);
    const err = this.check && this.info ? this.check(this.info, this.pendingConfigure) : null;
    if (err) { this.onEvent({ type: 'error', code: 'not_offered', error: err }); this.close(err); return; }
    try { this.send(this.pendingConfigure); } catch (e) { this.close(e.message); }
  }

  send(msg) {
    if (!this.dc || this.dc.readyState !== 'open') throw new Error('control channel is not open');
    this.dc.send(JSON.stringify(msg));
    this.onEvent({ ...msg, type: '→ ' + msg.type });
  }

  // A `prompt` message: `fields` may carry `prompt`, `end_image_url`,
  // `audio_url`, `audio_behavior`, `script`, `script_mode` (messages.rs).
  prompt(text, { replan = true, ...fields } = {}) {
    if (!this.configured) throw new Error('wait for `configured` before sending prompts');
    this.version += 1;
    const m = { type: 'prompt', prompt_version: this.version, replan };
    if (text) m.prompt = text;
    for (const [k, v] of Object.entries(fields)) if (v !== undefined && v !== null && v !== '') m[k] = v;
    this.send(m);
    return this.version;
  }

  stop() { try { this.send({ type: 'stop' }); } catch { this.close('stopped'); } }

  handle(raw) {
    let m;
    try { m = JSON.parse(raw); } catch { this.onEvent({ type: 'invalid', raw }); return; }
    if (m.type === 'session_info') { this.info = m; this.onEvent(m); this.sendConfigure(); return; }
    if (m.type === 'configured') { this.configured = true; this.onState('streaming'); }
    if (m.type === 'stream_exhausted') { this.onEvent(m); this.close('stream exhausted (' + m.reason + ')'); return; }
    if (m.type === 'error' && m.code && !['invalid_message', 'not_configured', 'immutable_settings'].includes(m.code) && m.prompt_version == null) {
      this.onEvent(m); this.close('error: ' + (m.error || m.code)); return;
    }
    this.onEvent(m);
  }

  close(reason) {
    if (this.hb) clearInterval(this.hb);
    clearTimeout(this.infoWait);
    this.hb = null;
    try { this.dc && this.dc.close(); } catch { /* ignore */ }
    try { this.pc && this.pc.close(); } catch { /* ignore */ }
    const was = this.pc;
    this.pc = null; this.dc = null; this.sessionId = null; this.configured = false;
    if (was) this.onState('closed', reason);
  }
}

// What the viewer is shown, reconstructed from the control messages (the
// director sends no playback-state message). Nothing plays until chunk 0 is
// built, which takes a model-dependent time (about 20 s for a 10 s chunk on
// h3-turbo at 480p, over a minute at 768p): the stream carries silence and
// no video meanwhile. Each chunk then plays for its `playback_seconds`,
// from its arrival or from the end of the one before it, and when the next
// one is not ready the last frame holds (fal's
// `freeze_video_and_silence_audio_until_ready`) until it is. When
// generation is slower than real time that hold follows every chunk, and it
// is not a stall: without this status line the page looked stuck until an
// unrelated event (the next prompt) happened to coincide with the next chunk.
export class PlaybackTracker {
  constructor() { this.reset(); }

  reset() {
    this.phase = 'idle';
    this.since = 0;
    this.segments = []; // [{index, start, end}] in ms
    this.est = null;
    this.play = null;
  }

  configured(now) {
    this.phase = 'generating';
    this.since = now;
  }

  chunk(m, now) {
    const last = this.segments[this.segments.length - 1];
    const start = Math.max(now, last ? last.end : now);
    const seconds = Number(m.playback_seconds) || 0;
    this.segments.push({ index: m.chunk_index, start, end: start + seconds * 1000 });
    if (this.segments.length > 8) this.segments.shift();
    if (m.next_generation_estimate_seconds != null) this.est = Number(m.next_generation_estimate_seconds);
    this.play = seconds;
    this.phase = 'streaming';
  }

  // {phase: idle|generating|playing|waiting, chunk, text} at `now`.
  view(now) {
    const s = (ms) => Math.max(0, Math.round(ms / 1000));
    if (this.phase === 'idle') return { phase: 'idle', chunk: null, text: '' };
    if (this.phase === 'generating') {
      return {
        phase: 'generating', chunk: 0,
        text: 'Generating chunk 0 (' + s(now - this.since) + ' s so far). Video starts when it is ready; there is no picture before that.',
      };
    }
    const cur = this.segments.find((g) => now >= g.start && now < g.end);
    if (cur) {
      return { phase: 'playing', chunk: cur.index, text: 'Playing chunk ' + cur.index + ' (' + s(cur.end - now) + ' s left).' };
    }
    const last = this.segments[this.segments.length - 1];
    const next = last.index + 1;
    const pace = this.est != null && this.play
      ? ' Each ' + this.play.toFixed(0) + ' s chunk takes about ' + this.est.toFixed(0) + ' s to generate, so the last frame holds between chunks.'
      : '';
    return { phase: 'waiting', chunk: next, text: 'Waiting for chunk ' + next + ' (' + s(now - last.end) + ' s).' + pace };
  }
}


// A URL input for an image / audio conditioning field, with a "Choose file"
// button that uploads through the fal storage API and fills in the URL.
function mediaInput(id, kind, placeholder) {
  const input = el('input', { type: 'url', id, placeholder, spellcheck: 'false', 'data-media': kind });
  const file = el('input', { type: 'file', accept: kind + '/*', hidden: true });
  const status = el('span', { class: 'hint' });
  const pick = el('button', { type: 'button', class: 'small', onclick: () => file.click() }, 'Choose file');
  file.onchange = async () => {
    const f = file.files[0];
    file.value = '';
    if (!f) return;
    status.textContent = ' uploading ' + f.name + '…';
    try { input.value = await upload(f); status.textContent = ''; } catch (e) { status.textContent = ' upload failed: ' + e.message; }
  };
  return { input, node: el('div', { class: 'urlrow' }, input, pick, file, status) };
}

// The script editor: beats `{offset, prompt, end_image_url, audio_url}`
// (messages.rs ScriptBeat). `offset` is whole seconds from the start of the
// script's first video; a beat with an end image is a keyframe at that offset.
function scriptEditor(prefix) {
  const rows = el('div', { class: 'beats', id: prefix + '-beats' });
  let allow = { images: true, audio: true };
  const empty = el('p', { class: 'hint' }, 'No beats: the prompt alone directs the stream.');
  function row(b = {}) {
    const offset = el('input', { type: 'number', min: 0, step: 1, value: b.offset ?? '', placeholder: 's', 'data-beat': 'offset', class: 'beat-offset' });
    const prompt = el('input', { value: b.prompt || '', placeholder: 'direction at this offset', 'data-beat': 'prompt', spellcheck: 'false' });
    const img = el('input', { type: 'url', value: b.end_image_url || '', placeholder: 'keyframe (end image URL)', 'data-beat': 'end_image_url', spellcheck: 'false' });
    const aud = el('input', { type: 'url', value: b.audio_url || '', placeholder: 'audio URL', 'data-beat': 'audio_url', spellcheck: 'false' });
    const rm = el('button', { type: 'button', class: 'small', title: 'Remove beat' }, '×');
    const r = el('div', { class: 'beat' }, offset, prompt, img, aud, rm);
    rm.onclick = () => { r.remove(); sync(); };
    rows.append(r);
    sync();
  }
  function sync() {
    for (const r of rows.children) {
      r.querySelector('[data-beat="end_image_url"]').hidden = !allow.images;
      r.querySelector('[data-beat="audio_url"]').hidden = !allow.audio;
    }
    empty.hidden = rows.children.length > 0;
  }
  const add = el('button', { type: 'button', class: 'small', id: prefix + '-add-beat' }, 'Add beat');
  add.onclick = () => row();
  return {
    node: el('div', {}, rows, empty, el('div', { class: 'actions', style: 'margin-top:6px' }, add)),
    // Allowed beat kinds (a causal model takes text beats only).
    allow(a) { allow = { ...allow, ...a }; sync(); },
    clear() { rows.replaceChildren(); sync(); },
    // The beats, or null for none; throws on an invalid beat.
    value() {
      const beats = [];
      for (const r of rows.children) {
        const get = (k) => r.querySelector('[data-beat="' + k + '"]');
        const off = get('offset').value.trim();
        const b = {};
        if (off === '' || !/^\d+$/.test(off)) throw new Error('every beat needs an offset in whole seconds');
        b.offset = Number(off);
        const p = get('prompt').value.trim();
        if (p) b.prompt = p;
        for (const k of ['end_image_url', 'audio_url']) {
          const v = get(k).value.trim();
          if (v && !get(k).hidden) b[k] = v;
        }
        if (!b.prompt && !b.end_image_url && !b.audio_url) throw new Error('beat at ' + b.offset + ' s has nothing to do');
        beats.push(b);
      }
      return beats.length ? beats : null;
    },
  };
}

const AUDIO_BITRATES = [96000, 128000, 192000];

// Director schema properties with their own control on this page; any other
// property the schema advertises (an enum or a bounded integer, such as the
// clip director's chunk size) is rendered generically and sent in
// `configure` under its own name.
const OWN_PROPS = new Set(['resolution', 'aspect_ratio', 'prompt', 'image_url', 'end_image_url', 'audio_url', 'audio_bitrate', 'memory', 'seed', 'script']);
// Property names that are a chunk length in seconds (labelled "N s").
const CHUNK_PROPS = /^(chunk_duration|chunk_seconds|chunk_size)$/;
const humanProp = (k) => (CHUNK_PROPS.test(k) ? 'Chunk size' : (k.charAt(0).toUpperCase() + k.slice(1)).replace(/_/g, ' '));

// Renders the director page into `root`. The form follows the app's
// director schema (`GET /fal/schema/{app}/director`: resolutions with cost
// labels, aspect ratios, `x-fv-director-mode`, `x-fv-licence`), the catalog
// entry (`director_mode`, `licence`) and, once a session is open, its
// `session_info` (scripts, end images, audio conditioning, bitrates).
export function mountDirector(root, { app, model }) {
  const appId = app + '/director';
  const video = el('video', { id: 'director-video', autoplay: true, playsinline: true, controls: true });
  const statePill = el('span', { class: 'pill', id: 'director-state' }, 'idle');
  const prompt = el('textarea', { id: 'director-prompt', rows: 4 }, 'A continuous live-action shot: a lighthouse keeper climbs the spiral stairs at dusk, lamp in hand.');
  let resLabels = {};
  const resLabel = (v) => resLabels[v] || v;
  const aspectLabel = (v) => (v === 'auto' ? (causal ? 'auto (16:9)' : 'auto (from image, else 16:9)') : v);
  const fill = (sel, values, label, def) => {
    sel.replaceChildren(...values.map((v) => el('option', { value: v }, label(v))));
    sel.value = values.includes(def) ? def : values[0];
    sel.disabled = false;
  };
  // Nothing is assumed before the schema loads: the session's own default
  // applies when it cannot be read (no `resolution` / `aspect_ratio` sent).
  const resolution = el('select', { id: 'director-resolution', disabled: true }, el('option', { value: '' }, 'loading…'));
  const aspect = el('select', { id: 'director-aspect', disabled: true }, el('option', { value: '' }, 'loading…'));
  const modeNote = el('p', { class: 'hint', id: 'director-mode-note', hidden: true });
  resolution.addEventListener('change', () => { for (const x of Object.values(extras)) if (x.setOptions) x.setOptions(resolution.value); });
  // Schema-advertised configure fields beyond the ones above: {name: {get(), causal}}.
  const extras = {};
  const extraRow = el('div', { class: 'row', id: 'director-extra', hidden: true });
  function buildExtras(props, isCausal) {
    for (const k of Object.keys(extras)) delete extras[k];
    extraRow.replaceChildren();
    for (const [k, p] of Object.entries(props || {})) {
      if (OWN_PROPS.has(k) || !p || typeof p !== 'object') continue;
      // `x-fv-causal: false` (or a chunk length on a causal model) hides it there.
      if (isCausal && (p['x-fv-causal'] === false || CHUNK_PROPS.test(k))) continue;
      const labels = (p['x-fv-labels'] && typeof p['x-fv-labels'] === 'object') ? p['x-fv-labels'] : {};
      const unit = CHUNK_PROPS.test(k) ? ' s' : '';
      let ctl;
      let get;
      if (Array.isArray(p.enum) && p.enum.length) {
        ctl = el('select', { id: 'director-' + k.replace(/_/g, '-'), 'data-configure': k });
        const note = el('p', { class: 'hint', 'data-note': k, hidden: true });
        // `x-fv-options-by-resolution`: the values served at each resolution
        // (the 10 s chunk is not served at every tier).
        const byRes = p['x-fv-options-by-resolution'] && typeof p['x-fv-options-by-resolution'] === 'object' ? p['x-fv-options-by-resolution'] : null;
        const setOptions = (res) => {
          const allowed = byRes && res && Array.isArray(byRes[res]) ? p.enum.filter((v) => byRes[res].some((x) => String(x) === String(v))) : p.enum;
          const keep = ctl.value;
          ctl.replaceChildren(...allowed.map((v) => el('option', { value: String(v) }, labels[v] || String(v) + unit)));
          ctl.value = allowed.some((v) => String(v) === keep) ? keep : String(allowed.some((v) => String(v) === String(p.default)) ? p.default : allowed[0]);
          const missing = p.enum.filter((v) => !allowed.includes(v));
          note.hidden = !missing.length;
          note.textContent = missing.length ? missing.map((v) => String(v) + unit).join(', ') + ' not served at ' + res + '.' : '';
        };
        setOptions(resolution.value);
        get = () => p.enum.find((v) => String(v) === ctl.value);
        extras[k] = { get, default: p.default, setOptions, note };
        extraRow.append(el('div', {}, el('label', {}, p.title || humanProp(k)), ctl, note, p.description ? el('p', { class: 'hint' }, p.description) : null));
        continue;
      } else if ((p.type === 'integer' || p.type === 'number') && Number.isFinite(p.minimum) && Number.isFinite(p.maximum)) {
        ctl = el('input', { type: 'number', id: 'director-' + k.replace(/_/g, '-'), 'data-configure': k, min: p.minimum, max: p.maximum, step: p.type === 'integer' ? 1 : 'any', value: p.default ?? '' });
        get = () => (ctl.value === '' ? undefined : Number(ctl.value));
      } else continue;
      extras[k] = { get, default: p.default };
      extraRow.append(field(p.title || humanProp(k), ctl, p.description));
    }
    extraRow.hidden = !extraRow.childNodes.length;
  }
  const licence = el('div', { id: 'director-licence' });
  const seed = el('input', { inputmode: 'numeric', placeholder: 'random', id: 'director-seed' });
  const memory = el('input', { type: 'number', min: 1, max: 50, value: 12, id: 'director-memory' });
  const image = mediaInput('director-image', 'image', 'optional first-frame image URL');
  const endImage = mediaInput('director-end-image', 'image', 'optional last-frame image URL (the opening chunk ends on it)');
  const audio = mediaInput('director-audio', 'audio', 'optional driving audio URL');
  const bitrate = el('select', { id: 'director-audio-bitrate' }, el('option', { value: '' }, 'server default'), AUDIO_BITRATES.map((b) => el('option', { value: String(b) }, b / 1000 + ' kbit/s')));
  const script = scriptEditor('director-script');
  const start = el('button', { class: 'primary', id: 'director-start' }, 'Start session');
  const stop = el('button', { id: 'director-stop', disabled: true }, 'Stop');
  const msg = el('div', { class: 'msg', role: 'status', id: 'director-msg' });
  const playback = el('div', { class: 'msg', role: 'status', id: 'director-playback', 'data-phase': 'idle' });
  const tracker = new PlaybackTracker();
  let ticker = null;
  function renderPlayback() {
    const v = tracker.view(Date.now());
    playback.textContent = v.text;
    playback.dataset.phase = v.phase;
    if (v.chunk == null) delete playback.dataset.chunk; else playback.dataset.chunk = String(v.chunk);
  }
  const next = el('textarea', { rows: 2, placeholder: 'Next direction, e.g. "They reach the lamp room and look out to sea."', id: 'director-next' });
  const replan = el('input', { type: 'checkbox', checked: true, id: 'director-replan' });
  const nextEnd = mediaInput('director-next-end-image', 'image', 'optional end image for this direction');
  const nextAudio = mediaInput('director-next-audio', 'audio', 'optional audio for this direction');
  const audioBehavior = el('select', { id: 'director-audio-behavior' }, el('option', { value: '' }, 'server default'), el('option', { value: 'replace' }, 'replace'), el('option', { value: 'queue' }, 'queue'));
  const nextScript = scriptEditor('director-next-script');
  const scriptMode = el('select', { id: 'director-script-mode' }, el('option', { value: 'replace' }, 'replace the script'), el('option', { value: 'append' }, 'append to the script'));
  const send = el('button', { class: 'primary', disabled: true, id: 'director-send' }, 'Send');
  const timeline = el('ul', { class: 'timeline', id: 'director-timeline' });
  const log = el('div', { class: 'logs', id: 'director-log' }, el('div', { class: 'lv' }, 'No events yet.'));
  const stats = el('dl', { id: 'director-stats' });
  const sessionFacts = el('dl', { id: 'director-session-info' });
  const unavailable = el('div', { class: 'banner', id: 'director-unavailable', hidden: true },
    'Streaming is not available on this server: the director signalling routes (', el('code', {}, '/wma/*'),
    ') are not mounted (fv-serve needs the webrtc feature and [protocols] fal_director on). Batch endpoints work.');

  // Blocks hidden for a causal model (text to video only) or a session
  // that does not take them; `data-cond` names them for the tests.
  const cond = (name, node) => el('div', { 'data-cond': name }, node);
  const imageBlock = cond('image', field('First-frame image', image.node));
  const endBlock = cond('end-image', field('Last-frame image', endImage.node));
  const audioBlock = cond('audio', el('div', {}, field('Driving audio', audio.node, 'Only where the session takes audio conditioning (checked against its session_info before configure).'), field('Audio bitrate', bitrate)));
  const nextEndBlock = cond('end-image', field('End image', nextEnd.node, 'The chunk this direction opens ends on this image.'));
  const nextAudioBlock = cond('audio', el('div', {}, field('Audio', nextAudio.node), field('Audio behaviour', audioBehavior)));
  const scriptBlock = cond('script', field('Script', script.node, 'Beats on the stream clock: a prompt, a keyframe (end image) and / or audio at whole-second offsets.'));
  const nextScriptBlock = cond('script', el('div', {}, field('Script', nextScript.node), field('Script mode', scriptMode)));

  let causal = false;
  function applyMode(isCausal, description) {
    causal = isCausal;
    root.dataset.directorMode = isCausal ? 'causal' : 'clip';
    for (const b of [imageBlock, endBlock, audioBlock, nextEndBlock, nextAudioBlock]) b.hidden = isCausal;
    script.allow({ images: !isCausal, audio: !isCausal });
    nextScript.allow({ images: !isCausal, audio: !isCausal });
    modeNote.hidden = !description;
    modeNote.textContent = description || '';
  }
  // `session_info` of the open session: what it takes at prompt time.
  function applySessionInfo(m) {
    // Why a chunk length is missing on this model or tier (`chunk_duration_note`).
    const cx = extras.chunk_duration;
    if (cx && cx.note && m.chunk_duration_note) { cx.note.hidden = false; cx.note.textContent = m.chunk_duration_note; }
    const audioOk = m.audio_conditioning === true;
    const endOk = !(m.script_max_end_images === 0);
    nextAudioBlock.hidden = causal || !audioOk;
    nextEndBlock.hidden = causal || (m.causal && m.causal.image_conditioning === false);
    nextScriptBlock.hidden = m.scripts === false;
    nextScript.allow({ images: !causal && endOk, audio: !causal && audioOk && (m.script_max_audio_beats || 0) > 0 });
    if (Array.isArray(m.script_modes) && m.script_modes.length) {
      const cur = scriptMode.value;
      scriptMode.replaceChildren(...m.script_modes.map((v) => el('option', { value: v }, v === 'append' ? 'append to the script' : v === 'replace' ? 'replace the script' : v)));
      if (m.script_modes.includes(cur)) scriptMode.value = cur;
    }
    if (Array.isArray(m.audio_behaviors)) {
      audioBehavior.replaceChildren(el('option', { value: '' }, 'server default'), ...m.audio_behaviors.map((v) => el('option', { value: v }, v)));
    }
    const facts = [['app', m.app], ['fps', m.fps], ['chunk', m.chunk_seconds != null ? m.chunk_seconds + ' s' : null],
      ['chunk options', Array.isArray(m.chunk_duration_options) ? m.chunk_duration_options.map((x) => x + ' s').join(', ') : null],
      ['resolutions', (m.resolutions || []).join(', ')], ['aspect ratios', (m.aspect_ratios || []).join(', ')],
      ['session limit', m.max_session_seconds != null ? m.max_session_seconds + ' s' : 'none'],
      ['scripts', m.scripts === false ? 'no' : 'up to ' + m.script_max_beats + ' beats, ' + m.script_max_end_images + ' keyframes'],
      ['audio conditioning', audioOk ? 'yes' : 'no']];
    if (m.causal) facts.push(['causal', m.causal.block_frames + '-frame blocks (' + m.causal.block_seconds + ' s), prompt switch: ' + m.causal.prompt_switch]);
    sessionFacts.replaceChildren(...facts.filter(([, v]) => v !== null && v !== undefined && v !== '').flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, String(v))]));
  }

  loadCatalog().then(({ apps }) => {
    const a = (apps || []).find((x) => x.id === app);
    if (!a) return;
    if (a.director_mode === 'causal' && !causal) applyMode(true, modeNote.textContent);
    if (a.licence && !licence.childNodes.length) licence.replaceChildren(licenceBanner(a.licence));
  }).catch(() => { /* the schema below still drives the form */ });
  request('GET', '/fal/schema/' + app + '/director', { auth: null }).then((form) => {
    const p = (form && form.properties) || {};
    applyMode(form['x-fv-director-mode'] === 'causal', form['x-fv-director-mode'] === 'causal' ? form.description : '');
    if (p.resolution && p.resolution['x-fv-labels'] && typeof p.resolution['x-fv-labels'] === 'object') resLabels = p.resolution['x-fv-labels'];
    if (p.resolution && Array.isArray(p.resolution.enum) && p.resolution.enum.length) fill(resolution, p.resolution.enum, resLabel, p.resolution.default);
    else fill(resolution, [''], () => 'server default', '');
    if (p.aspect_ratio && Array.isArray(p.aspect_ratio.enum) && p.aspect_ratio.enum.length) fill(aspect, p.aspect_ratio.enum, aspectLabel, p.aspect_ratio.default);
    else fill(aspect, ['auto'], aspectLabel, 'auto');
    buildExtras(p, causal);
    if (form['x-fv-licence']) licence.replaceChildren(licenceBanner(form['x-fv-licence']));
  }).catch(() => {
    // Keep the session defaults: send neither resolution nor aspect.
    fill(resolution, [''], () => 'server default', '');
    fill(aspect, ['auto'], aspectLabel, 'auto');
  });

  const prompts = new Map(); // version -> {text, status, li}
  function addTimeline(v, text, st) {
    const li = el('li', {}, el('span', { class: 'v' }, 'v' + v), el('span', { class: 't' }, text), el('span', { class: 'pill' }, st));
    prompts.set(v, { text, li });
    timeline.prepend(li);
  }
  function setPromptState(v, st, kind) {
    const p = prompts.get(v);
    if (!p) return;
    const pill = p.li.lastChild;
    pill.textContent = st; pill.className = 'pill' + (kind ? ' ' + kind : '');
  }
  function logLine(m) {
    const line = el('div', {}, el('span', { class: 'lv' }, new Date().toLocaleTimeString()), JSON.stringify(m));
    log.append(line);
    while (log.childNodes.length > 200) log.firstChild.remove();
    log.scrollTop = log.scrollHeight;
  }
  let chunks = 0;
  let meta = {};
  function renderStats(m) {
    const rows = [];
    if (meta.model) rows.push(['model', meta.model]);
    if (meta.tier) rows.push(['tier', el('span', {}, meta.tier, ' ', meta.quality === 'draft' ? draftPill() : '')]);
    if (meta.recipe) rows.push(['recipe', meta.recipe]);
    if (meta.chunk != null) rows.push(['chunk length', el('span', { id: 'director-chunk-duration' }, meta.chunk + ' s')]);
    if (m) {
      rows.push(['chunks', String(chunks)],
        ['buffer', m.buffer_depth_seconds != null ? Number(m.buffer_depth_seconds).toFixed(1) + ' s' : '—'],
        ['generation', m.generation_seconds != null ? Number(m.generation_seconds).toFixed(1) + ' s' : '—']);
      if (m.causal && m.causal.recache_ms != null) rows.push(['re-cache', m.causal.recache_ms + ' ms']);
    }
    stats.replaceChildren(...rows.flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, v)]));
  }
  let lastChunk = null;
  function onEvent(m) {
    logLine(m);
    switch (m.type) {
      case 'meta': meta = { ...m, chunk: meta.chunk }; renderStats(lastChunk); break;
      case 'session_info': applySessionInfo(m); break;
      case 'configured':
        // The chunk length the session actually runs (a tier may fall back).
        if (m.chunk_duration != null) { meta = { ...meta, chunk: m.chunk_duration }; renderStats(lastChunk); }
        setPromptState(m.prompt_version, 'applied', 'ok');
        tracker.configured(Date.now());
        if (!ticker) ticker = setInterval(renderPlayback, 250);
        renderPlayback();
        break;
      case 'prompt_pending': setPromptState(m.prompt_version, 'pending', 'warn'); break;
      case 'prompt_applied': setPromptState(m.prompt_version, 'applied', 'ok'); break;
      case 'prompt_rejected': setPromptState(m.prompt_version, 'rejected: ' + m.reason, 'bad'); break;
      case 'chunk':
        tracker.chunk(m, Date.now());
        renderPlayback();
        chunks += 1;
        lastChunk = m;
        renderStats(m);
        break;
      case 'deadline_missed': setMsg(msg, 'Chunk ' + m.chunk_index + ' late by ' + m.late_by_seconds + ' s (holding).', 'bad'); break;
      case 'error': setMsg(msg, (m.code || 'error') + ': ' + (m.error || ''), 'bad'); break;
      default: break;
    }
  }
  function onState(s, reason) {
    if (s === 'closed' || s === 'idle') {
      if (ticker) clearInterval(ticker);
      ticker = null;
      tracker.reset();
      renderPlayback();
    }
    statePill.textContent = s;
    statePill.className = 'pill' + (s === 'streaming' ? ' ok' : s === 'closed' ? '' : ' warn');
    start.disabled = s !== 'closed' && s !== 'idle';
    stop.disabled = !(s === 'connecting' || s === 'configuring' || s === 'streaming');
    send.disabled = s !== 'streaming';
    if (reason) setMsg(msg, reason);
  }

  // What the session's `session_info` says it does not take, for this configure.
  function offered(info, cfg) {
    if (cfg.audio_url && info.audio_conditioning === false) return 'This session takes no driving audio (session_info: audio_conditioning false). Remove the audio and start again.';
    if (cfg.resolution && Array.isArray(info.resolutions) && !info.resolutions.includes(cfg.resolution)) return 'This session serves ' + info.resolutions.join(', ') + ', not ' + cfg.resolution + '.';
    if (cfg.aspect_ratio && Array.isArray(info.aspect_ratios) && !info.aspect_ratios.includes(cfg.aspect_ratio)) return 'This session streams ' + info.aspect_ratios.join(', ') + ', not ' + cfg.aspect_ratio + '.';
    const beats = cfg.script || [];
    if (beats.length && info.scripts === false) return 'This session takes no script.';
    if (info.script_max_beats != null && beats.length > info.script_max_beats) return 'At most ' + info.script_max_beats + ' script beats.';
    const keys = beats.filter((b) => b.end_image_url).length + (cfg.end_image_url ? 1 : 0);
    if (info.script_max_end_images != null && keys > info.script_max_end_images) return 'At most ' + info.script_max_end_images + ' keyframes (end images) in this session.';
    if (beats.some((b) => b.audio_url) && (info.script_max_audio_beats || 0) === 0) return 'This session takes no audio beats.';
    return null;
  }

  let client = null;
  let armed = null; // a pool warning already shown: a second Start goes ahead
  start.onclick = async () => {
    await loadAuthMode();
    if (needsKey()) { setMsg(msg, 'Set an API key first.', 'bad'); return; }
    const warn = poolWarning(model);
    if (warn && armed !== warn) { armed = warn; setMsg(msg, warn + ' Click Start again to try anyway.', 'bad'); return; }
    armed = null;
    const text = prompt.value.trim();
    if (!text) { setMsg(msg, 'Enter an opening prompt.', 'bad'); return; }
    const cfg = { prompt: text, memory: Number(memory.value) || 12 };
    if (resolution.value) cfg.resolution = resolution.value;
    if (aspect.value && aspect.value !== 'auto') cfg.aspect_ratio = aspect.value;
    if (seed.value.trim()) cfg.seed = Number(seed.value.trim());
    for (const [k, x] of Object.entries(extras)) { const v = x.get(); if (v !== undefined && v !== null && v !== '') cfg[k] = v; }
    if (!causal) {
      if (image.input.value.trim()) cfg.image_url = image.input.value.trim();
      if (endImage.input.value.trim()) cfg.end_image_url = endImage.input.value.trim();
      if (audio.input.value.trim()) cfg.audio_url = audio.input.value.trim();
      if (bitrate.value) cfg.audio_bitrate = Number(bitrate.value);
    }
    try { const beats = script.value(); if (beats) cfg.script = beats; } catch (e) { setMsg(msg, 'Script: ' + e.message, 'bad'); return; }
    // With a script the opening chunk follows its beats: an end image goes in a keyframe beat.
    if (cfg.script && cfg.end_image_url) { setMsg(msg, 'With a script, put the last-frame image in a beat (a keyframe at its offset).', 'bad'); return; }
    prompts.clear(); timeline.replaceChildren(); chunks = 0; lastChunk = null; meta = {}; stats.replaceChildren(); sessionFacts.replaceChildren();
    client = new DirectorClient({ appId, onEvent, onState, onTrack: (s) => { video.srcObject = s; }, check: offered });
    addTimeline(1, text + (cfg.script ? ' (+ ' + cfg.script.length + ' beats)' : ''), 'configuring');
    setMsg(msg, '');
    try { await client.start(cfg); } catch (e) {
      client.close();
      onState('idle');
      if (UNAVAILABLE.has(e.status)) { unavailable.hidden = false; setMsg(msg, ''); } else setMsg(msg, e.message, 'bad');
    }
  };
  stop.onclick = () => client && client.stop();
  send.onclick = () => {
    if (!client) return;
    const text = next.value.trim();
    const extra = {};
    if (!nextEndBlock.hidden && nextEnd.input.value.trim()) extra.end_image_url = nextEnd.input.value.trim();
    if (!nextAudioBlock.hidden && nextAudio.input.value.trim()) {
      extra.audio_url = nextAudio.input.value.trim();
      if (audioBehavior.value) extra.audio_behavior = audioBehavior.value;
    }
    try {
      const beats = nextScriptBlock.hidden ? null : nextScript.value();
      if (beats) {
        // A script message carries only the beats (control.rs: `script` is
        // exclusive with `prompt`, `end_image_url` and `audio_url`).
        if (text || extra.end_image_url || extra.audio_url) { setMsg(msg, 'A script is sent on its own: clear the next prompt, end image and audio (put the directions in the beats).', 'bad'); return; }
        extra.script = beats; extra.script_mode = scriptMode.value;
      }
    } catch (e) { setMsg(msg, 'Script: ' + e.message, 'bad'); return; }
    if (!text && !Object.keys(extra).length) return;
    try {
      const v = client.prompt(text, { replan: replan.checked, ...extra });
      const what = [text, extra.end_image_url ? 'end image' : '', extra.audio_url ? 'audio' : '', extra.script ? extra.script.length + ' beats (' + extra.script_mode + ')' : ''].filter(Boolean).join(' + ');
      addTimeline(v, what, 'sent');
      next.value = ''; nextEnd.input.value = ''; nextAudio.input.value = ''; nextScript.clear();
    } catch (e) { setMsg(msg, e.message, 'bad'); }
  };
  next.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send.click(); });

  root.replaceChildren(
    unavailable,
    licence,
    el('div', { class: 'grid2' },
      el('section', { class: 'stage' },
        el('div', { class: 'stage-head' }, el('h2', {}, 'Session'), statePill),
        el('div', { class: 'stage-body' },
          modeNote,
          field('Opening prompt', prompt),
          el('div', { class: 'row' }, field('Resolution', resolution), field('Aspect ratio', aspect)),
          extraRow,
          imageBlock,
          el('details', { class: 'more', id: 'director-more' }, el('summary', {}, 'Additional settings'),
            el('div', { class: 'row' }, field('Seed', seed), field('Memory', memory, 'Prior segment prompts kept as context (1-50).')),
            endBlock, audioBlock, scriptBlock),
          el('div', { class: 'actions' }, start, stop, poolBadge(model)), msg)),
      el('section', { class: 'stage' },
        el('div', { class: 'stage-head' }, el('h2', {}, 'Stream')),
        el('div', { class: 'stage-body' }, el('div', { style: 'margin-top:10px' }, video), playback, stats,
          el('details', { class: 'more' }, el('summary', {}, 'Session info'), sessionFacts))),
      el('section', { class: 'stage' },
        el('div', { class: 'stage-head' }, el('h2', {}, 'Direct')),
        el('div', { class: 'stage-body' },
          field('Next prompt', next),
          el('label', { class: 'check' }, replan, 'Replan (apply at the next undispatched chunk; off appends after planned chunks)'),
          el('details', { class: 'more', id: 'director-next-more' }, el('summary', {}, 'End image, audio, script'),
            nextEndBlock, nextAudioBlock, nextScriptBlock),
          el('div', { class: 'actions' }, send),
          el('h3', { style: 'margin-top:14px' }, 'Timeline'), timeline)),
      el('section', { class: 'stage' },
        el('div', { class: 'stage-head' }, el('h2', {}, 'Events')),
        el('div', { class: 'stage-body' }, log))));
  onState('idle');

  DirectorClient.probe(appId).then((p) => {
    if (!p.available && p.reason === 'not-implemented') {
      unavailable.hidden = false;
      start.disabled = true;
      statePill.textContent = 'unavailable';
    } else if (p.reason === 'unauthorized') {
      setMsg(msg, 'The API key was refused: ' + p.message, 'bad');
    } else if (!p.available) {
      setMsg(msg, 'Could not reach the director: ' + p.message, 'bad');
    }
  });
}
