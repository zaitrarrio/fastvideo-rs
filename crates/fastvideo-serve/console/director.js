// fal director (WMA) client for the console: design §5.6, research-fal §8.
//
// Everything that talks to the director lives in this module so WP-14 can
// finish the wiring in one place. Flow:
//   1. POST /wma/ice {app_id}                      -> {ice_servers}
//   2. RTCPeerConnection, client-created data channel "control", recvonly
//      video + audio, full (non-trickle) offer after ICE gathering
//   3. POST /wma/session {app_id, sdp, type:"offer"} -> {session_id, sdp, type:"answer"}
//   4. POST /wma/session/heartbeat {session_id} every 5 s -> {alive}
//   5. on channel open: `configure` (prompt_version 1); wait for `configured`
//   6. `prompt` messages with prompt_version 2, 3, … (replan true/false)
//   7. `stop` -> `stream_exhausted`, then close
// 404 / 405 / 501 on the signalling routes means this server has no
// director yet: the page says so instead of failing.

import { el, request, setMsg, apiKey } from './common.js';

const HEARTBEAT_MS = 5000;
const UNAVAILABLE = new Set([404, 405, 501]);

function field(label, node, hint) {
  return el('div', {}, el('label', {}, label), node, hint ? el('p', { class: 'hint' }, hint) : null);
}

export class DirectorClient {
  constructor({ appId, onEvent, onState, onTrack }) {
    this.appId = appId;
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
    dc.onopen = () => { this.onState('configuring'); this.send(this.pendingConfigure); };
    dc.onmessage = (ev) => this.handle(ev.data);
    dc.onclose = () => this.close('control channel closed');

    await pc.setLocalDescription(await pc.createOffer());
    await new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') return resolve();
      const t = setTimeout(resolve, 3000);
      pc.addEventListener('icegatheringstatechange', () => {
        if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve(); }
      });
    });
    const answer = await request('POST', '/wma/session', {
      auth: 'key', body: { app_id: this.appId, sdp: pc.localDescription.sdp, type: 'offer' },
    });
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

  send(msg) {
    if (!this.dc || this.dc.readyState !== 'open') throw new Error('control channel is not open');
    this.dc.send(JSON.stringify(msg));
    this.onEvent({ type: '→ ' + msg.type, ...msg });
  }

  prompt(text, { replan = true } = {}) {
    if (!this.configured) throw new Error('wait for `configured` before sending prompts');
    this.version += 1;
    this.send({ type: 'prompt', prompt_version: this.version, prompt: text, replan });
    return this.version;
  }

  stop() { try { this.send({ type: 'stop' }); } catch { this.close('stopped'); } }

  handle(raw) {
    let m;
    try { m = JSON.parse(raw); } catch { this.onEvent({ type: 'invalid', raw }); return; }
    if (m.type === 'configured') { this.configured = true; this.onState('streaming'); }
    if (m.type === 'stream_exhausted') { this.onEvent(m); this.close('stream exhausted (' + m.reason + ')'); return; }
    if (m.type === 'error' && m.code && !['invalid_message', 'not_configured', 'immutable_settings'].includes(m.code) && m.prompt_version == null) {
      this.onEvent(m); this.close('error: ' + (m.error || m.code)); return;
    }
    this.onEvent(m);
  }

  close(reason) {
    if (this.hb) clearInterval(this.hb);
    this.hb = null;
    try { this.dc && this.dc.close(); } catch { /* ignore */ }
    try { this.pc && this.pc.close(); } catch { /* ignore */ }
    const was = this.pc;
    this.pc = null; this.dc = null; this.sessionId = null; this.configured = false;
    if (was) this.onState('closed', reason);
  }
}

// Renders the director page into `root`.
export function mountDirector(root, { app }) {
  const appId = app + '/director';
  const video = el('video', { id: 'director-video', autoplay: true, playsinline: true, controls: true });
  const statePill = el('span', { class: 'pill', id: 'director-state' }, 'idle');
  const prompt = el('textarea', { id: 'director-prompt', rows: 4 }, 'A continuous live-action shot: a lighthouse keeper climbs the spiral stairs at dusk, lamp in hand.');
  const resolution = el('select', { id: 'director-resolution' }, ['480p', '768p', '1080p'].map((v) => el('option', { value: v }, v === '1080p' ? '1080p (about 2.5x slower per chunk)' : v)));
  resolution.value = '768p';
  // `auto` sends no aspect_ratio: the session follows the image (16:9 without one).
  const aspect = el('select', { id: 'director-aspect' }, ['auto', '16:9', '9:16', '1:1'].map((v) => el('option', { value: v }, v === 'auto' ? 'auto (from image, else 16:9)' : v)));
  const seed = el('input', { inputmode: 'numeric', placeholder: 'random' });
  const memory = el('input', { type: 'number', min: 1, max: 50, value: 12 });
  const imageUrl = el('input', { type: 'url', placeholder: 'optional first-frame image URL', spellcheck: 'false' });
  const start = el('button', { class: 'primary', id: 'director-start' }, 'Start session');
  const stop = el('button', { id: 'director-stop', disabled: true }, 'Stop');
  const msg = el('div', { class: 'msg', role: 'status', id: 'director-msg' });
  const next = el('textarea', { rows: 2, placeholder: 'Next direction, e.g. "They reach the lamp room and look out to sea."', id: 'director-next' });
  const replan = el('input', { type: 'checkbox', checked: true, id: 'director-replan' });
  const send = el('button', { class: 'primary', disabled: true, id: 'director-send' }, 'Send prompt');
  const timeline = el('ul', { class: 'timeline', id: 'director-timeline' });
  const log = el('div', { class: 'logs', id: 'director-log' }, el('div', { class: 'lv' }, 'No events yet.'));
  const stats = el('dl', { id: 'director-stats' });
  const unavailable = el('div', { class: 'banner', id: 'director-unavailable', hidden: true },
    'Streaming is not available on this server yet: the director signalling routes (', el('code', {}, '/wma/*'),
    ') are not mounted. Batch endpoints work; the Director page will connect once the server ships the director.');

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
  function onEvent(m) {
    logLine(m);
    switch (m.type) {
      case 'configured': setPromptState(m.prompt_version, 'applied', 'ok'); break;
      case 'prompt_pending': setPromptState(m.prompt_version, 'pending', 'warn'); break;
      case 'prompt_applied': setPromptState(m.prompt_version, 'applied', 'ok'); break;
      case 'prompt_rejected': setPromptState(m.prompt_version, 'rejected: ' + m.reason, 'bad'); break;
      case 'chunk':
        chunks += 1;
        stats.replaceChildren(
          el('dt', {}, 'chunks'), el('dd', {}, String(chunks)),
          el('dt', {}, 'buffer'), el('dd', {}, m.buffer_depth_seconds != null ? Number(m.buffer_depth_seconds).toFixed(1) + ' s' : '—'),
          el('dt', {}, 'generation'), el('dd', {}, m.generation_seconds != null ? Number(m.generation_seconds).toFixed(1) + ' s' : '—'));
        break;
      case 'deadline_missed': setMsg(msg, 'Chunk ' + m.chunk_index + ' late by ' + m.late_by_seconds + ' s (holding).', 'bad'); break;
      case 'error': setMsg(msg, (m.code || 'error') + ': ' + (m.error || ''), 'bad'); break;
      default: break;
    }
  }
  function onState(s, reason) {
    statePill.textContent = s;
    statePill.className = 'pill' + (s === 'streaming' ? ' ok' : s === 'closed' ? '' : ' warn');
    start.disabled = s !== 'closed' && s !== 'idle';
    stop.disabled = !(s === 'connecting' || s === 'configuring' || s === 'streaming');
    send.disabled = s !== 'streaming';
    if (reason) setMsg(msg, reason);
  }

  let client = null;
  start.onclick = async () => {
    if (!apiKey()) { setMsg(msg, 'Set an API key first.', 'bad'); return; }
    const text = prompt.value.trim();
    if (!text) { setMsg(msg, 'Enter an opening prompt.', 'bad'); return; }
    const cfg = { prompt: text, resolution: resolution.value, memory: Number(memory.value) || 12 };
    if (aspect.value !== 'auto') cfg.aspect_ratio = aspect.value;
    if (seed.value.trim()) cfg.seed = Number(seed.value.trim());
    if (imageUrl.value.trim()) cfg.image_url = imageUrl.value.trim();
    prompts.clear(); timeline.replaceChildren(); chunks = 0; stats.replaceChildren();
    client = new DirectorClient({ appId, onEvent, onState, onTrack: (s) => { video.srcObject = s; } });
    addTimeline(1, text, 'configuring');
    setMsg(msg, '');
    try { await client.start(cfg); } catch (e) {
      client.close();
      onState('idle');
      if (UNAVAILABLE.has(e.status)) { unavailable.hidden = false; setMsg(msg, ''); } else setMsg(msg, e.message, 'bad');
    }
  };
  stop.onclick = () => client && client.stop();
  send.onclick = () => {
    const text = next.value.trim();
    if (!text || !client) return;
    try { const v = client.prompt(text, { replan: replan.checked }); addTimeline(v, text, 'sent'); next.value = ''; }
    catch (e) { setMsg(msg, e.message, 'bad'); }
  };
  next.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send.click(); });

  root.replaceChildren(
    unavailable,
    el('div', { class: 'grid2' },
      el('section', { class: 'stage' },
        el('div', { class: 'stage-head' }, el('h2', {}, 'Session'), statePill),
        el('div', { class: 'stage-body' },
          field('Opening prompt', prompt),
          el('div', { class: 'row' }, field('Resolution', resolution), field('Aspect ratio', aspect)),
          el('details', { class: 'more' }, el('summary', {}, 'Additional settings'),
            el('div', { class: 'row' }, field('Seed', seed), field('Memory', memory, 'Prior segment prompts kept as context (1-50).')),
            field('First-frame image URL', imageUrl)),
          el('div', { class: 'actions' }, start, stop), msg)),
      el('section', { class: 'stage' },
        el('div', { class: 'stage-head' }, el('h2', {}, 'Stream')),
        el('div', { class: 'stage-body' }, el('div', { style: 'margin-top:10px' }, video), stats)),
      el('section', { class: 'stage' },
        el('div', { class: 'stage-head' }, el('h2', {}, 'Direct')),
        el('div', { class: 'stage-body' },
          field('Next prompt', next),
          el('label', { class: 'check' }, replan, 'Replan (apply at the next undispatched chunk; off appends after planned chunks)'),
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
