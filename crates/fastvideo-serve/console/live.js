// Live input (design §5.11): publish the camera and microphone to a duplex
// model and watch its output, over one of two transports.
//
// Native WHIP ingest:
//   1. getUserMedia (the chosen size, the model's fps cap)
//   2. RTCPeerConnection: one send-receive video and audio transceiver
//      each (camera out, model output back on the same m-lines)
//   3. POST /fv/v1/streams/ingest?model=… (application/sdp, Bearer key)
//      -> 201 + answer + Location; DELETE <Location> stops
//   4. GET <Location> every second: the model and ingest counters
// Reactor runtime (the model must be the runtime's `[reactor] model`):
//   1. POST /start_session {context, max_seconds} (Bearer key)
//   2. POST …/connections; transceivers main_video / main_audio (recvonly)
//      and input_video / input_audio (sendonly); data channels `data` and
//      `control`; the offer with `track_mapping`
//   3. POST …/sdp_params, poll GET …/sdp_params for the answer
//   4. on `control` open (v0 JSON): publish_track input_video/input_audio,
//      resume_track main_video/main_audio; `get_state` on `data` every
//      second (-> state_update)
//   5. POST /stop_session

import { $, el, base, apiKey, request, setMsg, topbar, loadAuthMode, needsKey } from './common.js';
import { gathered } from './rtc.js';

topbar('live');

const SID = '00000000-0000-0000-0000-000000000000';
const W = '/sessions/' + SID + '/transport/webrtc';

let models = [];
let live = null;

function log(line) {
  const box = $('live-log');
  box.append(el('div', {}, new Date().toISOString().slice(11, 23) + ' ' + line));
  while (box.childElementCount > 200) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

function setState(text, kind) {
  const p = $('live-state');
  p.textContent = text;
  p.className = 'pill' + (kind ? ' ' + kind : '');
  document.body.dataset.liveState = text;
}

function authHeaders(extra = {}) {
  const h = { ...extra };
  if (apiKey() && !document.body.dataset.keyless) h.Authorization = 'Bearer ' + apiKey();
  return h;
}

async function loadModels() {
  const sel = $('model');
  try {
    const caps = await request('GET', '/fv/v1/capabilities', { auth: 'bearer' });
    models = (caps.models || []).filter((m) => m.caps && m.caps.stream && m.caps.stream.duplex);
  } catch (e) {
    setMsg('live-msg', 'Could not read the capabilities: ' + e.message, 'bad');
    models = [];
  }
  sel.replaceChildren(...models.map((m) => el('option', { value: m.caps.id }, m.caps.id)));
  if (!models.length) setMsg('live-msg', 'This server serves no duplex model (fv-serve with FV_ECHO_MODEL=1 serves the loopback echo).', 'bad');
  showFacts();
}

function current() {
  return models.find((m) => m.caps.id === $('model').value);
}

function showFacts() {
  const m = current();
  if (!m) { $('model-facts').textContent = ''; return; }
  const d = m.caps.stream.duplex;
  const v = d.input.video;
  const parts = [];
  if (v) parts.push('camera ' + v.codecs.join('/') + ' up to ' + v.max_width + '×' + v.max_height + ' at ' + v.max_fps + ' fps (model input ' + v.width + '×' + v.height + ')');
  if (d.input.audio) parts.push('microphone ' + d.input.audio.rate + ' Hz');
  parts.push('≤ ' + d.input.max_bitrate_kbps + ' kbit/s in', 'output ' + d.target_fps + ' fps' + (d.audio_out ? ' with audio' : ''), d.unit_ms + ' ms units');
  if (m.stream_limits) parts.push(m.stream_limits.default_max_s + ' s sessions (at most ' + m.stream_limits.hard_max_s + ')');
  $('model-facts').textContent = parts.join(' · ');
}

async function media() {
  const m = current();
  const v = m && m.caps.stream.duplex.input.video;
  const [w, h] = $('res').value.split('x').map(Number);
  const constraints = {
    video: $('cam').checked && v ? { width: { ideal: w }, height: { ideal: h }, frameRate: { max: v.max_fps } } : false,
    audio: $('mic').checked && m && m.caps.stream.duplex.input.audio ? { channelCount: 1, echoCancellation: true } : false,
  };
  if (!constraints.video && !constraints.audio) throw new Error('choose the camera, the microphone or both');
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  $('local').srcObject = stream;
  return stream;
}

function attachRemote(pc) {
  const out = new MediaStream();
  $('remote').srcObject = out;
  pc.ontrack = (ev) => {
    out.addTrack(ev.track);
    log('← ' + ev.track.kind + ' track');
  };
  pc.onconnectionstatechange = () => {
    log('peer ' + pc.connectionState);
    if (pc.connectionState === 'connected') setState('streaming', 'ok');
    if (pc.connectionState === 'failed') stop('the peer connection failed');
  };
}

function showStats(entries) {
  $('live-stats').replaceChildren(...entries.flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, v)]));
}

// ---- native WHIP ingest ------------------------------------------------------

async function startWhip(stream) {
  const m = current();
  const pc = new RTCPeerConnection();
  attachRemote(pc);
  const vt = stream.getVideoTracks()[0];
  const at = stream.getAudioTracks()[0];
  // Send-receive: the model's output comes back on the same m-lines.
  pc.addTransceiver(vt || 'video', { direction: vt ? 'sendrecv' : 'recvonly', streams: [stream] });
  pc.addTransceiver(at || 'audio', { direction: at ? 'sendrecv' : 'recvonly', streams: [stream] });
  await pc.setLocalDescription(await pc.createOffer());
  await gathered(pc);
  const q = new URLSearchParams({ model: m.caps.id });
  for (const k of ['scene', 'persona']) if ($(k).value.trim()) q.set(k, $(k).value.trim());
  if ($('max-seconds').value) q.set('max_seconds', $('max-seconds').value);
  const resp = await fetch(base() + '/fv/v1/streams/ingest?' + q, {
    method: 'POST', headers: authHeaders({ 'Content-Type': 'application/sdp' }), body: pc.localDescription.sdp,
  });
  const text = await resp.text();
  if (resp.status !== 201) {
    let msg = text;
    try { msg = JSON.parse(text).error.message; } catch { /* plain text */ }
    pc.close();
    throw new Error('HTTP ' + resp.status + ': ' + msg);
  }
  const location = resp.headers.get('location');
  document.body.dataset.stream = location;
  log('→ WHIP ' + location);
  await pc.setRemoteDescription({ type: 'answer', sdp: text });
  const poll = setInterval(async () => {
    try {
      const s = await request('GET', location, { auth: 'bearer' });
      const sess = s.session || {};
      const ing = s.ingest || {};
      showStats([
        ['state', s.state + (s.end_reason ? ' (' + s.end_reason + ')' : '')],
        ['model', (sess.paused ? 'paused · ' : '') + (sess.frames_out || 0) + ' frames out, ' + (sess.input_frames || 0) + ' input frames shown'],
        ['input', (ing.codec || '–') + (ing.source_size ? ' ' + ing.source_size.join('×') : '') + ' · ' + (ing.video_frames_decoded || 0) + ' decoded, '
          + (ing.dropped_waiting_keyframe || 0) + ' waiting for a keyframe, ' + (ing.dropped_behind || 0) + ' dropped (behind)'],
        ['latency', sess.input_latency_ms != null ? sess.input_latency_ms.toFixed(1) + ' ms in the model queue' : '–'],
        ['refused', ing.rejected || '–'],
        ['output', (s.output.video_codec || '–') + ' ' + s.output.width + '×' + s.output.height + ' · ' + s.output.frames_sent + ' frames sent'],
      ]);
      document.body.dataset.inputFrames = String(sess.input_frames || 0);
      if (s.state === 'closed') stop('the stream ended: ' + (s.end_reason || 'closed'));
    } catch (e) { log('status: ' + e.message); }
  }, 1000);
  return {
    pc, stream, poll,
    pause: (p) => request('POST', location + '/commands', { auth: 'bearer', body: { type: 'set_paused', data: { paused: p } } }),
    close: async () => { try { await request('DELETE', location, { auth: 'bearer' }); } catch (e) { log('DELETE: ' + e.message); } },
  };
}

// ---- Reactor runtime ---------------------------------------------------------

async function reactorCall(method, path, body) {
  const resp = await fetch(base() + path, {
    method, headers: authHeaders(body ? { 'Content-Type': 'application/json' } : {}), body: body ? JSON.stringify(body) : undefined,
  });
  const text = await resp.text();
  let v = null;
  try { v = text ? JSON.parse(text) : null; } catch { v = text; }
  return { status: resp.status, body: v };
}

async function startReactor(stream) {
  const context = {};
  for (const k of ['scene', 'persona']) if ($(k).value.trim()) context[k] = $(k).value.trim();
  const params = { context };
  if ($('max-seconds').value) params.max_seconds = Number($('max-seconds').value);
  let r = await reactorCall('POST', '/start_session', params);
  if (r.status === 404) throw new Error('this server does not mount the Reactor runtime');
  if (r.status !== 200) throw new Error('start_session: HTTP ' + r.status + ' ' + JSON.stringify(r.body));
  const tracks = (r.body.capabilities && r.body.capabilities.tracks) || [];
  if (!tracks.some((t) => t.direction === 'sendonly')) {
    await reactorCall('POST', '/stop_session', {});
    throw new Error('the Reactor runtime streams ' + r.body.model.name + ', which takes no input tracks (set [reactor] model to a duplex model)');
  }
  const ice = await reactorCall('GET', W + '/ice_servers');
  const pc = new RTCPeerConnection({ iceServers: ((ice.body && ice.body.ice_servers) || []).map((s) => ({ urls: s.uris, ...(s.credentials || {}) })) });
  attachRemote(pc);
  const data = pc.createDataChannel('data');
  const control = pc.createDataChannel('control');
  const vt = stream.getVideoTracks()[0];
  const at = stream.getAudioTracks()[0];
  const tx = {};
  for (const t of tracks) {
    if (t.direction === 'recvonly') tx[t.name] = pc.addTransceiver(t.kind, { direction: 'recvonly' });
    else if (t.kind === 'video' && vt) tx[t.name] = pc.addTransceiver(vt, { direction: 'sendonly', streams: [stream] });
    else if (t.kind === 'audio' && at) tx[t.name] = pc.addTransceiver(at, { direction: 'sendonly', streams: [stream] });
  }
  r = await reactorCall('POST', W + '/connections');
  if (r.status !== 201) throw new Error('connections: HTTP ' + r.status + ' ' + JSON.stringify(r.body));
  const cid = r.body.connection_id;
  await pc.setLocalDescription(await pc.createOffer());
  await gathered(pc);
  const mapping = Object.entries(tx).map(([name, t]) => {
    const tr = tracks.find((x) => x.name === name);
    return { mid: t.mid, name, kind: tr.kind, direction: tr.direction };
  });
  r = await reactorCall('POST', W + '/connections/' + cid + '/sdp_params', { sdp_offer: pc.localDescription.sdp, track_mapping: mapping });
  if (r.status !== 202) throw new Error('sdp_params: HTTP ' + r.status + ' ' + JSON.stringify(r.body));
  let answer = null;
  for (let i = 0; i < 100 && !answer; i++) {
    const a = await reactorCall('GET', W + '/connections/' + cid + '/sdp_params');
    if (a.status === 200) answer = a.body.sdp_answer;
    else if (a.status !== 202) throw new Error('sdp_params: HTTP ' + a.status + ' ' + JSON.stringify(a.body));
    else await new Promise((res) => setTimeout(res, 100));
  }
  if (!answer) throw new Error('no answer from the Reactor runtime');
  await pc.setRemoteDescription({ type: 'answer', sdp: answer });
  log('→ Reactor connection ' + cid);
  let rid = 0;
  const send = (ch, m) => { if (ch.readyState === 'open') ch.send(JSON.stringify(m)); };
  control.onopen = () => {
    for (const t of tracks.filter((x) => x.direction === 'sendonly' && tx[x.name])) {
      send(control, { type: 'request', method: 'publish_track', request_id: 'pub' + (++rid), data: { name: t.name } });
      log('→ publish_track ' + t.name);
    }
    for (const t of tracks.filter((x) => x.direction === 'recvonly')) send(control, { type: 'notification', event: 'resume_track', data: { name: t.name } });
  };
  control.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m.method === 'publish_track') log('← publish_track ' + (m.error ? 'refused: ' + m.error.message : 'ok'));
  };
  data.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    const msg = m.data || {};
    if (msg.type === 'state_update') {
      const s = msg.data || {};
      const ing = s.ingest || {};
      showStats([
        ['model', (s.paused ? 'paused · ' : '') + (s.frames_out || 0) + ' frames out, ' + (s.input_frames || 0) + ' input frames shown'],
        ['input', (ing.codec || '–') + (ing.source_size ? ' ' + ing.source_size.join('×') : '') + ' · ' + (ing.video_frames_decoded || 0) + ' decoded'],
        ['publishers', JSON.stringify(s.publishers || {})],
        ['latency', s.input_latency_ms != null ? s.input_latency_ms.toFixed(1) + ' ms in the model queue' : '–'],
      ]);
      document.body.dataset.inputFrames = String(s.input_frames || 0);
    } else if (msg.type === 'input_rejected') {
      log('← input refused: ' + (msg.data && msg.data.reason));
    } else if (m.data && m.data.type === 'sessionEnded') {
      stop('session ended: ' + (m.data.data && m.data.data.reason));
    }
  };
  const poll = setInterval(() => send(data, { scope: 'application', data: { type: 'get_state', data: {} } }), 1000);
  return {
    pc, stream, poll,
    pause: async (p) => send(data, { scope: 'application', data: { type: 'set_paused', data: { paused: p } } }),
    close: async () => { await reactorCall('POST', '/stop_session', {}); },
  };
}

// ---- page --------------------------------------------------------------------

async function start() {
  if (live) return;
  setMsg('live-msg', '');
  if (!current()) { setMsg('live-msg', 'Pick a duplex model.', 'bad'); return; }
  setState('starting', 'warn');
  $('start').disabled = true;
  let stream = null;
  try {
    stream = await media();
    live = $('transport').value === 'reactor' ? await startReactor(stream) : await startWhip(stream);
    live.paused = false;
    $('pause').disabled = false; $('stop').disabled = false;
    setState('connecting', 'warn');
  } catch (e) {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    setState('failed', 'bad');
    setMsg('live-msg', e.message, 'bad');
    $('start').disabled = false;
    live = null;
  }
}

async function stop(reason) {
  const l = live;
  live = null;
  if (!l) return;
  clearInterval(l.poll);
  await l.close();
  l.pc.close();
  l.stream.getTracks().forEach((t) => t.stop());
  setState('stopped');
  if (reason) setMsg('live-msg', reason);
  $('start').disabled = false; $('pause').disabled = true; $('stop').disabled = true;
}

$('start').onclick = start;
$('stop').onclick = () => stop('stopped');
$('pause').onclick = async () => {
  if (!live) return;
  live.paused = !live.paused;
  await live.pause(live.paused);
  $('pause').textContent = live.paused ? 'Resume model' : 'Pause model';
};
$('unmute').onchange = () => { $('remote').muted = !$('unmute').checked; };
$('model').onchange = showFacts;
window.addEventListener('beforeunload', () => { if (live) live.close(); });

loadAuthMode().then((mode) => {
  if (mode === 'none') document.body.dataset.keyless = '1';
  $('key-banner').hidden = !needsKey();
  loadModels();
});
