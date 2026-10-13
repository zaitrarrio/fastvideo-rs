// Stream outputs in the console (docs/serve/stream-outputs.md): the
// destination form (the Live stream page, an optional copy on the director
// page), the `output` request object it builds, and the player that matches
// what the output delivers:
//   - WHIP and Cloudflare (WebRTC): a WHEP player (rtc.js whepPlay) on the
//     playback `whep_url` (Cloudflare's `webRTCPlayback`);
//   - RTMPS / SRT: an HLS player on `hls_url`: the browser's own HLS
//     (Safari) or the vendored hls.js (hls.light.min.js 1.5.20, Apache-2.0,
//     console/vendor/), loaded on first use as a classic script;
//   - a destination of your own with no known player: the URL pasted in the
//     form, WHEP or HLS (`.m3u8`).
// Credentials stay in password inputs: never stored (no localStorage),
// never logged, and the server never echoes them.

import { el, request } from './common.js';
import { whepPlay } from './rtc.js';

export const KINDS = [
  ['cloudflare', 'Cloudflare Stream: WebRTC (WHIP in, WHEP out)'],
  ['cloudflare-rtmps', 'Cloudflare Stream: RTMPS in, HLS out'],
  ['cloudflare-srt', 'Cloudflare Stream: SRT in, HLS out'],
  ['whip', 'Your WHIP endpoint (MediaMTX, an SFU, …)'],
  ['rtmps', 'Your RTMP / RTMPS ingest'],
  ['srt', 'Your SRT listener'],
];

// The fields each kind takes.
export const FIELDS = {
  cloudflare: ['codec'],
  'cloudflare-rtmps': [],
  'cloudflare-srt': [],
  whip: ['url', 'token', 'codec', 'target'],
  rtmps: ['url', 'stream_key'],
  srt: ['url', 'stream_id', 'passphrase'],
};

// An `output` object fit for a log line: credentials replaced by "set" and
// the URL's userinfo / query dropped.
export const SECRET_FIELDS = ['token', 'stream_key', 'passphrase', 'stream_id'];
export function redactOutput(o) {
  if (!o || typeof o !== 'object') return o;
  const r = { ...o };
  for (const k of SECRET_FIELDS) if (r[k]) r[k] = 'set';
  if (r.url) {
    try { const u = new URL(r.url); u.username = ''; u.password = ''; u.search = ''; r.url = u.toString(); } catch { r.url = 'set'; }
  }
  return r;
}

// GET /fv/v1/outputs: {default, kinds, …}, or null (an older server).
export async function loadOutputs() {
  try { return await request('GET', '/fv/v1/outputs', { auth: 'bearer' }); } catch { return null; }
}

function row(label, node, hint) {
  return el('div', { class: 'out-field' }, el('label', { for: node.id }, label), node, hint ? el('p', { class: 'hint' }, hint) : null);
}

// The destination form. `optional`: a "no output" choice first (the
// director's copy). Element ids are `<prefix>-kind`, `-url`, `-token`,
// `-key`, `-stream-id`, `-passphrase`, `-codec`, `-target`, `-play`.
export function outputForm(prefix, { optional = false } = {}) {
  const id = (s) => prefix + '-' + s;
  const kind = el('select', { id: id('kind') },
    optional ? el('option', { value: '' }, 'No copy') : null,
    KINDS.map(([v, label]) => el('option', { value: v }, label)));
  const url = el('input', { id: id('url'), type: 'url', spellcheck: 'false', autocomplete: 'off' });
  const token = el('input', { id: id('token'), type: 'password', spellcheck: 'false', autocomplete: 'new-password', placeholder: 'optional' });
  const key = el('input', { id: id('key'), type: 'password', spellcheck: 'false', autocomplete: 'new-password', placeholder: 'or the last path segment of the URL' });
  const streamId = el('input', { id: id('stream-id'), spellcheck: 'false', autocomplete: 'off', placeholder: 'optional (MediaMTX: publish:<path>)' });
  const passphrase = el('input', { id: id('passphrase'), type: 'password', spellcheck: 'false', autocomplete: 'new-password', placeholder: 'optional, 10-79 characters' });
  const codec = el('select', { id: id('codec') }, ['h264', 'vp8', 'vp9'].map((c) => el('option', { value: c }, c === 'h264' ? 'H.264 (default)' : c.toUpperCase())));
  const target = el('select', { id: id('target') }, el('option', { value: '' }, 'guess from the URL'), el('option', { value: 'mediamtx' }, 'mediamtx'), el('option', { value: 'cloudflare' }, 'cloudflare'));
  const play = el('input', { id: id('play'), type: 'url', spellcheck: 'false', autocomplete: 'off', placeholder: 'https://…/whep or https://…/index.m3u8' });
  const note = el('p', { class: 'hint', id: id('note') });
  const rows = {
    url: row('Destination URL', url),
    token: row('WHIP token (Bearer)', token),
    stream_key: row('Stream key', key),
    stream_id: row('SRT stream id', streamId),
    passphrase: row('SRT passphrase', passphrase),
    codec: row('Video codec', codec, 'Cloudflare and most WHIP relays take H.264, VP8 and VP9.'),
    target: row('WHIP endpoint kind', target),
    play: row('Playback URL (optional)', play, 'Your relay\'s WHEP URL, or an HLS playlist (.m3u8): this page plays it.'),
  };
  const node = el('div', { class: 'outputs', id: id('fields') }, row('Destination', kind), note, Object.values(rows));
  let info = null;
  const placeholders = { whip: 'https://relay.example/live/whip', rtmps: 'rtmps://live.example.com:443/live', srt: 'srt://live.example.com:778' };
  function sync() {
    const k = kind.value;
    const take = FIELDS[k] || [];
    for (const [name, r] of Object.entries(rows)) r.hidden = name === 'play' ? !k || k.startsWith('cloudflare') : !take.includes(name);
    url.placeholder = placeholders[k] || '';
    note.textContent = !k ? '' : k.startsWith('cloudflare')
      ? 'The server creates a Cloudflare Stream live input for this session (its account) and deletes it when the session ends; this page plays it back.'
      : 'Credentials go to the server in this request only: never stored in this browser, never shown back.';
  }
  kind.addEventListener('change', sync);
  sync();
  return {
    node,
    kind: () => kind.value,
    // Marks the kinds the server cannot publish (no Cloudflare account).
    setInfo(i) {
      info = i;
      const kinds = (i && i.kinds) || null;
      for (const o of kind.options) {
        if (!o.value) continue;
        o.disabled = !!kinds && !kinds.includes(o.value);
        if (o.disabled && o.value.startsWith('cloudflare')) o.textContent = o.textContent.replace(/ \(not configured on this server\)$/, '') + ' (not configured on this server)';
      }
      if (kind.selectedOptions[0] && kind.selectedOptions[0].disabled) {
        const first = [...kind.options].find((o) => !o.disabled);
        if (first) kind.value = first.value;
      }
      sync();
    },
    info: () => info,
    // The request's `output` object (null: no copy); throws on a missing URL.
    read() {
      const k = kind.value;
      if (!k) return null;
      const take = FIELDS[k] || [];
      const out = { kind: k };
      const v = (x) => x.value.trim();
      if (take.includes('url')) {
        if (!v(url)) throw new Error('enter the destination URL');
        out.url = v(url);
      }
      if (take.includes('token') && v(token)) out.token = v(token);
      if (take.includes('stream_key') && v(key)) out.stream_key = v(key);
      if (take.includes('stream_id') && v(streamId)) out.stream_id = v(streamId);
      if (take.includes('passphrase') && v(passphrase)) out.passphrase = v(passphrase);
      if (take.includes('codec') && codec.value !== 'h264') out.codec = codec.value;
      if (take.includes('target') && target.value) out.target = target.value;
      return out;
    },
    pasted: () => (rows.play.hidden ? '' : play.value.trim()),
  };
}

// The player for an output: {kind: 'whep' | 'hls' | null, url, why}.
// `playback` is the stream object's (or `output_ready`'s) playback block;
// `pasted` a URL from the form for destinations of your own.
export function playerFor(playback, pasted) {
  const p = playback || {};
  if (p.whep_url && (p.format === 'whep' || !p.hls_url)) return { kind: 'whep', url: p.whep_url, why: 'WebRTC (WHEP) playback of the output' };
  if (p.hls_url) return { kind: 'hls', url: p.hls_url, why: 'HLS playback of the output' };
  if (pasted) {
    const hls = /\.m3u8(\?|#|$)/i.test(pasted);
    return { kind: hls ? 'hls' : 'whep', url: pasted, why: (hls ? 'HLS' : 'WHEP') + ' URL from the form' };
  }
  return { kind: null, url: null, why: 'no playback URL: this destination has no known player (paste its WHEP or HLS URL)' };
}

let hlsLoading = null;
// The vendored hls.js, loaded once as a classic script (it defines `Hls`).
function loadHls() {
  if (window.Hls) return Promise.resolve(window.Hls);
  if (!hlsLoading) {
    hlsLoading = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = '/console/assets/hls.light.min.js';
      s.onload = () => (window.Hls ? resolve(window.Hls) : reject(new Error('hls.js did not load')));
      s.onerror = () => reject(new Error('hls.js did not load'));
      document.head.append(s);
    });
  }
  return hlsLoading;
}

// Plays an HLS playlist into `video`: native HLS when the browser has it,
// else hls.js (no worker: the console's CSP allows scripts from this origin
// only). A live input's playlist appears a few seconds after the broadcast
// starts: 404s are retried for `retryMs`. Returns {engine, close()}.
export async function hlsPlay(url, video, { log = () => {}, retryMs = 120000 } = {}) {
  if (video.canPlayType('application/vnd.apple.mpegurl')) {
    video.src = url;
    log('HLS: native playback');
    return { engine: 'native', close() { video.removeAttribute('src'); video.load(); } };
  }
  const Hls = await loadHls();
  if (!Hls.isSupported()) throw new Error('this browser has no Media Source Extensions: HLS cannot play here');
  const hls = new Hls({ enableWorker: false, lowLatencyMode: true, liveDurationInfinity: true });
  const t0 = Date.now();
  hls.on(Hls.Events.ERROR, (_e, d) => {
    if (!d.fatal) return;
    const notYet = d.details === Hls.ErrorDetails.MANIFEST_LOAD_ERROR || d.details === Hls.ErrorDetails.MANIFEST_LOAD_TIMEOUT;
    if (notYet && Date.now() - t0 < retryMs) {
      log('HLS: playlist not there yet, retrying');
      setTimeout(() => hls.loadSource(url), 3000);
    } else if (d.type === Hls.ErrorTypes.MEDIA_ERROR) {
      hls.recoverMediaError();
    } else {
      log('HLS error: ' + d.details);
    }
  });
  hls.on(Hls.Events.MANIFEST_PARSED, () => { log('HLS: playing'); video.play().catch(() => {}); });
  hls.loadSource(url);
  hls.attachMedia(video);
  return { engine: 'hls.js', close() { hls.destroy(); } };
}

// Plays an output into `video` per `playerFor`; returns {kind, url, close()}.
export async function playOutput(video, playback, pasted, { log = () => {} } = {}) {
  const choice = playerFor(playback, pasted);
  video.dataset.player = choice.kind || 'none';
  if (!choice.kind) return { ...choice, close() {} };
  if (choice.kind === 'hls') {
    const p = await hlsPlay(choice.url, video, { log });
    video.dataset.hlsEngine = p.engine;
    return { ...choice, close: p.close };
  }
  const out = new MediaStream();
  video.srcObject = out;
  const p = await whepPlay(choice.url, { onTrack: (t) => out.addTrack(t), onState: (s) => log('WHEP ' + s) });
  return { ...choice, close: () => p.close() };
}
