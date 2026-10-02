// WebRTC helpers shared by the console's live pages (the director, Live
// input, Live stream): full (non-trickle) offers, the Reactor runtime's
// signalling, and a WHEP player.

import { base, apiKey, keyless } from './common.js';

// Resolves once ICE gathering completes (at most `ms`): the servers here
// take complete offers, as the director's client does.
export function gathered(pc, ms = 3000) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const t = setTimeout(resolve, ms);
    pc.addEventListener('icegatheringstatechange', () => {
      if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve(); }
    });
  });
}

// `Authorization: Bearer <key>` unless the server takes no key.
export function bearer(extra = {}) {
  const h = { ...extra };
  if (apiKey() && !keyless()) h.Authorization = 'Bearer ' + apiKey();
  return h;
}

// A Reactor runtime call -> {status, body}.
export async function reactorCall(method, path, body) {
  const resp = await fetch(base() + path, {
    method, headers: bearer(body ? { 'Content-Type': 'application/json' } : {}), body: body ? JSON.stringify(body) : undefined,
  });
  const text = await resp.text();
  let v = null;
  try { v = text ? JSON.parse(text) : null; } catch { v = text; }
  return { status: resp.status, body: v };
}

// The model the Reactor runtime streams (`GET /schema` `info.title`), or
// null when the runtime is not mounted.
export async function reactorModel() {
  try {
    const r = await fetch(base() + '/schema');
    if (!r.ok) return null;
    const s = await r.json();
    return (s && s.info && s.info.title) || null;
  } catch { return null; }
}

const SID = '00000000-0000-0000-0000-000000000000';
const W = '/sessions/' + SID + '/transport/webrtc';

// Watches a Reactor session that publishes nothing (a causal or clip
// model): `start_session` (`params`: seed, max_seconds), one recvonly
// transceiver per output track, the `data` and `control` channels, the
// offer with `track_mapping`, the polled answer. `onMessage(m)` gets the
// application messages (`state_update`, `command_error`, …). Returns
// {pc, command(type, data), close()}.
export async function reactorWatch({ params = {}, onTrack, onMessage, onState, log = () => {} }) {
  let r = await reactorCall('POST', '/start_session', params);
  if (r.status === 404) throw new Error('this server does not mount the Reactor runtime');
  if (r.status !== 200) throw new Error('start_session: HTTP ' + r.status + ' ' + JSON.stringify(r.body));
  const tracks = ((r.body.capabilities && r.body.capabilities.tracks) || []).filter((t) => t.direction === 'recvonly');
  const ice = await reactorCall('GET', W + '/ice_servers');
  const pc = new RTCPeerConnection({ iceServers: ((ice.body && ice.body.ice_servers) || []).map((s) => ({ urls: s.uris, ...(s.credentials || {}) })) });
  pc.ontrack = (ev) => onTrack && onTrack(ev.track);
  pc.onconnectionstatechange = () => onState && onState(pc.connectionState);
  const data = pc.createDataChannel('data');
  const control = pc.createDataChannel('control');
  const tx = {};
  for (const t of tracks) tx[t.name] = pc.addTransceiver(t.kind, { direction: 'recvonly' });
  r = await reactorCall('POST', W + '/connections');
  if (r.status !== 201) { pc.close(); throw new Error('connections: HTTP ' + r.status + ' ' + JSON.stringify(r.body)); }
  const cid = r.body.connection_id;
  await pc.setLocalDescription(await pc.createOffer());
  await gathered(pc);
  const mapping = tracks.map((t) => ({ mid: tx[t.name].mid, name: t.name, kind: t.kind, direction: t.direction }));
  r = await reactorCall('POST', W + '/connections/' + cid + '/sdp_params', { sdp_offer: pc.localDescription.sdp, track_mapping: mapping });
  if (r.status !== 202) { pc.close(); throw new Error('sdp_params: HTTP ' + r.status + ' ' + JSON.stringify(r.body)); }
  let answer = null;
  for (let i = 0; i < 100 && !answer; i++) {
    const a = await reactorCall('GET', W + '/connections/' + cid + '/sdp_params');
    if (a.status === 200) answer = a.body.sdp_answer;
    else if (a.status !== 202) { pc.close(); throw new Error('sdp_params: HTTP ' + a.status + ' ' + JSON.stringify(a.body)); }
    else await new Promise((res) => setTimeout(res, 100));
  }
  if (!answer) { pc.close(); throw new Error('no answer from the Reactor runtime'); }
  await pc.setRemoteDescription({ type: 'answer', sdp: answer });
  log('→ Reactor connection ' + cid);
  const send = (ch, m) => { if (ch.readyState === 'open') ch.send(JSON.stringify(m)); };
  const pending = [];
  control.onopen = () => {
    for (const t of tracks) send(control, { type: 'notification', event: 'resume_track', data: { name: t.name } });
  };
  data.onopen = () => { while (pending.length) send(data, pending.shift()); };
  data.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m && m.data) onMessage && onMessage(m.data);
  };
  return {
    pc,
    // An application command (`set_prompt`, `set_paused`, `reset`, `get_state`, …).
    command(type, payload = {}) {
      const m = { scope: 'application', data: { type, data: payload } };
      if (data.readyState === 'open') send(data, m); else pending.push(m);
    },
    async close() { try { await reactorCall('POST', '/stop_session', {}); } finally { pc.close(); } },
  };
}

// Plays a WHEP endpoint (RFC 9725 WHEP: POST a recvonly offer, 201 with the
// answer and a Location; DELETE it to stop) into `onTrack`. A MediaMTX or
// Cloudflare Stream relay that the server's WHIP stream publishes to has one.
export async function whepPlay(url, { token, onTrack, onState } = {}) {
  const pc = new RTCPeerConnection();
  pc.ontrack = (ev) => onTrack && onTrack(ev.track);
  pc.onconnectionstatechange = () => onState && onState(pc.connectionState);
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });
  await pc.setLocalDescription(await pc.createOffer());
  await gathered(pc);
  const headers = { 'Content-Type': 'application/sdp' };
  if (token) headers.Authorization = 'Bearer ' + token;
  let resp;
  try {
    resp = await fetch(url, { method: 'POST', headers, body: pc.localDescription.sdp });
  } catch (e) { pc.close(); throw new Error('WHEP ' + url + ': ' + e.message); }
  const text = await resp.text();
  if (resp.status !== 201 && resp.status !== 200) { pc.close(); throw new Error('WHEP: HTTP ' + resp.status + ' ' + text.slice(0, 200)); }
  await pc.setRemoteDescription({ type: 'answer', sdp: text });
  const loc = resp.headers.get('location');
  const resource = loc ? new URL(loc, url).toString() : null;
  return {
    pc,
    async close() {
      pc.close();
      if (resource) { try { await fetch(resource, { method: 'DELETE', headers: token ? { Authorization: 'Bearer ' + token } : {} }); } catch { /* gone */ } }
    },
  };
}
