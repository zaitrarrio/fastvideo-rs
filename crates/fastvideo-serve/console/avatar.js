// Script avatar page: a Reactor client (v0 JSON wire) for the avatar mode
// of fv-serve's Reactor runtime (`[reactor] mode = "avatar"`, Reactor's
// `ltx` model contract; design §5.7, research-avatar-v2v.md P0-3).
//
// Flow (reactor §3): POST /start_session (409 when one runs: join it) ->
// ice_servers -> POST connections -> RTCPeerConnection with recv-only video
// and audio and the client-created `data` and `control` channels ->
// non-trickle offer to sdp_params with the track mapping -> poll the answer.
// The first frame (a v0 ping) latches the JSON wire; resume_track opens the
// tracks. The photo (and a voice file) go through the uploads protocol:
// POST /sessions/{sid}/uploads, PUT the bytes, then reference the upload id
// in set_avatar_image. Every server message lands in `window.__avatar`
// (the headless test driver reads it).

import { $, el, topbar, setMsg } from './common.js';

const SID = '00000000-0000-0000-0000-000000000000';
const W = '/sessions/' + SID + '/transport/webrtc';
const PING_MS = 5000;

topbar('avatar');

const log = [];
window.__avatar = { log, state: null, windows: {} };

function show(line) {
  const pre = $('log');
  pre.textContent = (line + '\n' + pre.textContent).slice(0, 20000);
}

async function http(method, path, body, raw) {
  const init = { method, headers: {} };
  if (raw) { init.body = raw; init.headers['Content-Type'] = 'application/octet-stream'; }
  else if (body !== undefined) { init.body = JSON.stringify(body); init.headers['Content-Type'] = 'application/json'; }
  const r = await fetch(path, init);
  const text = await r.text();
  let v = null;
  try { v = text ? JSON.parse(text) : null; } catch { v = text; }
  if (!r.ok && r.status !== 202) {
    const e = new Error((v && v.detail) || text || ('HTTP ' + r.status));
    e.status = r.status;
    throw e;
  }
  return { status: r.status, body: v };
}

class ReactorClient {
  constructor(onMessage, onTrack, onClose) {
    this.onMessage = onMessage; this.onTrack = onTrack; this.onClose = onClose;
    this.pc = null; this.data = null; this.control = null; this.ping = null;
  }

  async connect() {
    try { await http('POST', '/start_session', {}); } catch (e) { if (e.status !== 409) throw e; }
    const ice = (await http('GET', W + '/ice_servers')).body;
    const iceServers = ((ice && ice.ice_servers) || []).map((s) => ({
      urls: s.uris, username: s.credentials && s.credentials.username, credential: s.credentials && s.credentials.password,
    }));
    const cid = (await http('POST', W + '/connections', {})).body.connection_id;
    const pc = new RTCPeerConnection({ iceServers });
    this.pc = pc;
    window.__avatar.pc = pc;
    const v = pc.addTransceiver('video', { direction: 'recvonly' });
    const a = pc.addTransceiver('audio', { direction: 'recvonly' });
    // One MediaStream for both tracks (the Reactor answer may put video and
    // audio in separate msid streams).
    const stream = new MediaStream();
    pc.ontrack = (ev) => { stream.addTrack(ev.track); this.onTrack(stream); };
    pc.onconnectionstatechange = () => {
      if (['failed', 'closed'].includes(pc.connectionState)) this.close('peer ' + pc.connectionState);
    };
    this.data = pc.createDataChannel('data');
    this.control = pc.createDataChannel('control');
    this.data.onmessage = (ev) => this.handle(ev.data);
    this.control.onmessage = (ev) => this.handle(ev.data);
    const open = (ch) => new Promise((resolve) => { ch.onopen = resolve; });
    const opened = Promise.all([open(this.data), open(this.control)]);
    await pc.setLocalDescription(await pc.createOffer());
    await new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') return resolve();
      const t = setTimeout(resolve, 3000);
      pc.addEventListener('icegatheringstatechange', () => { if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve(); } });
    });
    const mapping = [
      { mid: v.mid, name: 'main_video', kind: 'video', direction: 'recvonly' },
      { mid: a.mid, name: 'main_audio', kind: 'audio', direction: 'recvonly' },
    ];
    await http('POST', W + '/connections/' + cid + '/sdp_params', {
      sdp_offer: pc.localDescription.sdp, track_mapping: mapping, client_info: { sdk_version: 'fv-console', sdk_type: 'js' },
    });
    let answer = null;
    for (let i = 0; i < 300 && !answer; i++) {
      const r = await http('GET', W + '/connections/' + cid + '/sdp_params');
      if (r.status === 200) answer = r.body.sdp_answer;
      else await new Promise((res) => setTimeout(res, 100));
    }
    if (!answer) throw new Error('no SDP answer');
    await pc.setRemoteDescription({ type: 'answer', sdp: answer });
    await opened;
    // The first frame latches the v0 JSON wire; then open both tracks.
    this.runtime('ping', {});
    for (const name of ['main_video', 'main_audio']) {
      this.control.send(JSON.stringify({ type: 'notification', event: 'resume_track', data: { name } }));
    }
    this.ping = setInterval(() => { try { this.runtime('ping', {}); } catch { /* closed */ } }, PING_MS);
  }

  runtime(type, data) { this.data.send(JSON.stringify({ scope: 'runtime', data: { type, data } })); }

  command(type, data = {}) {
    if (!this.data || this.data.readyState !== 'open') throw new Error('not connected');
    this.data.send(JSON.stringify({ scope: 'application', data: { type, data } }));
    const entry = { t: performance.now(), dir: '→', type, data: type === 'set_script' ? { script: '(' + (data.script || '').length + ' chars)' } : data };
    log.push(entry);
    show('→ ' + type + ' ' + JSON.stringify(entry.data));
  }

  handle(raw) {
    let m;
    try { m = JSON.parse(raw); } catch { return; }
    const inner = m && m.data;
    if (!inner || !inner.type) return;
    if (m.scope === 'runtime' && inner.type === 'sessionEnded') { this.close('session ended: ' + ((inner.data && inner.data.reason) || '')); return; }
    this.onMessage(inner.type, inner.data || {});
  }

  close(reason) {
    if (this.ping) clearInterval(this.ping);
    this.ping = null;
    try { this.pc && this.pc.close(); } catch { /* ignore */ }
    const was = this.pc;
    this.pc = null; this.data = null; this.control = null;
    if (was) this.onClose(reason);
  }
}

async function upload(file) {
  const r = await http('POST', '/sessions/' + SID + '/uploads', { name: file.name, size: file.size, mime_type: file.type || 'application/octet-stream' });
  const url = r.body.presigned_url || r.body.path;
  await http('PUT', url, undefined, file);
  return { upload_id: r.body.presigned_id, name: file.name, mime_type: file.type, size: file.size };
}

let client = null;
let sentImage = null;
let sentVoice = null;

function setButtons(st) {
  const valid = new Set((st && st.valid_commands) || []);
  $('pause').disabled = !valid.has('pause');
  $('resume').disabled = !valid.has('resume');
  $('stop').disabled = !valid.has('stop');
  $('reset').disabled = !client;
  $('end').disabled = !client;
  $('start').disabled = !!(st && st.generating);
  const pill = $('take-state');
  if (!st) { pill.textContent = 'idle'; pill.className = 'pill'; return; }
  const [text, tone] = st.generating ? (st.paused ? ['paused', 'warn'] : ['generating', 'ok']) : st.finished ? ['finished', 'ok'] : ['ready', ''];
  pill.textContent = text; pill.className = 'pill' + (tone ? ' ' + tone : '');
  $('playback').textContent = st.generating || st.finished
    ? `window ${Math.min(st.window_index + 1, st.total_windows)}/${st.total_windows} · ${st.seconds_sent.toFixed(1)} of ${st.effective_seconds.toFixed(1)} s sent`
      + (st.first_frame_seconds != null ? ` · first frame after ${st.first_frame_seconds.toFixed(1)} s` : '')
      + (st.stalls ? ` · waited for generation ${st.stalls}× (${st.stalled_seconds.toFixed(1)} s)` : '')
    : '';
}

function row(i) {
  let tr = $('win-' + i);
  if (!tr) {
    tr = el('tr', { id: 'win-' + i }, ...['#' + i, '', '', '', '', '', ''].map((t) => el('td', {}, t)));
    $('windows').append(tr);
  }
  return tr.children;
}

function onMessage(type, data) {
  const entry = { t: performance.now(), dir: '←', type, data };
  log.push(entry);
  if (type !== 'state_update') show('← ' + type + ' ' + JSON.stringify(data));
  const w = window.__avatar.windows;
  switch (type) {
    case 'state_update': window.__avatar.state = data; setButtons(data); break;
    case 'generation_started': $('windows').replaceChildren(); window.__avatar.windows = {}; break;
    case 'window_built': {
      (window.__avatar.windows[data.window_index] ||= {}).built = data;
      const c = row(data.window_index);
      c[1].textContent = data.kind; c[2].textContent = data.seconds.toFixed(2);
      c[3].textContent = data.build_seconds.toFixed(2); c[4].textContent = data.rtf.toFixed(2);
      break;
    }
    case 'window_started': {
      (w[data.window_index] ||= {}).started = data;
      const c = row(data.window_index);
      c[5].textContent = data.since_start_seconds.toFixed(2) + ' s'; c[6].textContent = data.stalled_seconds.toFixed(2) + ' s';
      break;
    }
    case 'command_error': setMsg('take-msg', data.command + ': ' + data.reason, 'err'); break;
    case 'generation_failed': setMsg('take-msg', 'The take failed: ' + data.reason, 'err'); break;
    case 'generation_complete': setMsg('take-msg', 'Take complete: ' + data.seconds_sent.toFixed(1) + ' s streamed.', 'ok'); break;
    default: break;
  }
}

async function ensureClient() {
  if (client) return client;
  const pill = $('session-state');
  pill.textContent = 'connecting…'; pill.className = 'pill warn';
  const c = new ReactorClient(onMessage, (stream) => {
    const v = $('video');
    if (v.srcObject !== stream) v.srcObject = stream;
    window.__avatar.stream = stream;
  }, (reason) => {
    client = null; sentImage = null; sentVoice = null;
    pill.textContent = 'closed'; pill.className = 'pill';
    setMsg('take-msg', 'Session closed: ' + reason);
    setButtons(null);
  });
  await c.connect();
  client = c;
  pill.textContent = 'connected'; pill.className = 'pill ok';
  return c;
}

$('image').onchange = () => {
  const f = $('image').files[0];
  const img = $('image-preview');
  if (f) { img.src = URL.createObjectURL(f); img.hidden = false; } else img.hidden = true;
};
$('script').oninput = () => {
  const words = $('script').value.split(/\s+/).filter(Boolean).length;
  const wpm = Number($('wpm').value) || 140;
  $('script-facts').textContent = `${words} words · about ${(words * 60 / wpm).toFixed(1)} s at ${wpm} wpm`;
};
$('wpm').oninput = $('script').oninput;

$('start').onclick = async () => {
  setMsg('take-msg', '');
  try {
    const c = await ensureClient();
    const img = $('image').files[0];
    if (img && img !== sentImage) { c.command('set_avatar_image', { avatar_image: await upload(img) }); sentImage = img; }
    const voice = $('voice').files[0];
    if (voice && voice !== sentVoice) { c.command('set_voice_audio', { voice_audio: await upload(voice) }); sentVoice = voice; }
    c.command('set_script', { script: $('script').value });
    c.command('set_prompt', { prompt: $('prompt').value });
    c.command('set_wpm', { wpm: Number($('wpm').value) || 140 });
    c.command('set_duration_seconds', { duration_seconds: Number($('duration').value) || 0 });
    if ($('seed').value !== '') c.command('set_seed', { seed: Number($('seed').value) });
    c.command('start');
  } catch (e) {
    setMsg('take-msg', 'Could not start: ' + e.message, 'err');
  }
};
for (const cmd of ['pause', 'resume', 'stop', 'reset']) {
  $(cmd).onclick = () => { try { client && client.command(cmd); } catch (e) { setMsg('take-msg', e.message, 'err'); } };
}
$('end').onclick = async () => {
  try { await http('POST', '/stop_session', { reason: 'ended from the console' }); } catch (e) { setMsg('take-msg', e.message, 'err'); }
};
