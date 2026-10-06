// Shared helpers for the fv-serve console pages (no build step, no deps).

export const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, String(v));
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

// localStorage / sessionStorage that never throw (private windows).
function wrap(get) {
  return {
    get(k) { try { return get().getItem(k) || ''; } catch { return ''; } },
    set(k, v) { try { if (v) get().setItem(k, v); else get().removeItem(k); } catch { /* ignore */ } },
    json(k, fallback) { try { return JSON.parse(get().getItem(k)) ?? fallback; } catch { return fallback; } },
    setJson(k, v) { try { get().setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
  };
}
export const store = wrap(() => localStorage);
export const session = wrap(() => sessionStorage);

export const K = { base: 'fv.base', key: 'fv.key', admin: 'fv.admin', history: 'fv.history', theme: 'fv.theme' };

export function base() {
  const b = store.get(K.base).trim().replace(/\/+$/, '');
  return b || location.origin;
}
export const apiKey = () => store.get(K.key).trim();

// The server's auth mode, `auth.mode` of `GET /fv/v1/capabilities` probed
// with no credentials. `none` (FV_AUTH_MODE=none) means the API takes no key:
// the console then drops its key prompts and sends no `Authorization`
// (except the admin token on the admin page, which admin routes always
// need). A 401, a server without the native API or an older server that does
// not report the mode all count as `keys`.
let authModePromise = null;
let openServer = false;
export function loadAuthMode() {
  if (!authModePromise) {
    authModePromise = request('GET', '/fv/v1/capabilities', { auth: null })
      .then((c) => (c && c.auth && typeof c.auth.mode === 'string' ? c.auth.mode : 'keys'), () => 'keys')
      .then((m) => { openServer = m === 'none'; return m; });
  }
  return authModePromise;
}
// Probe again (the server URL changed).
export function resetAuthMode() { authModePromise = null; openServer = false; capsPromise = null; }
// True once the server reported `auth.mode = none`.
export const keyless = () => openServer;
// A request would be refused for lack of a key.
export const needsKey = () => !openServer && !apiKey();

export function errorText(status, body) {
  if (body && body.error && body.error.message) return body.error.message;
  if (body && body.detail !== undefined) {
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d) => (d.loc ? d.loc.filter((x) => x !== 'body').join('.') + ': ' : '') + (d.msg || JSON.stringify(d))).join('; ');
    }
    return JSON.stringify(body.detail);
  }
  if (body && body.base_resp && body.base_resp.status_msg) return body.base_resp.status_msg;
  if (typeof body === 'string' && body) return body.slice(0, 300);
  return 'HTTP ' + status;
}

export class HttpError extends Error {
  constructor(status, body) {
    super(errorText(status, body));
    this.status = status;
    this.body = body;
  }
}

// fetch JSON. `auth`: 'key' (fal `Key`), 'bearer', 'admin' (bearer with the
// admin token) or null. Throws HttpError on non-2xx. `full: true` resolves
// `{body, headers, status}` instead of the body (for the `x-fv-*` headers).
export async function request(method, path, { auth = 'key', token, body, root, full = false } = {}) {
  const headers = { Accept: 'application/json' };
  const secret = token ?? (auth === 'admin' ? session.get(K.admin) : apiKey());
  // An open server (auth mode `none`) gets no API key; the admin token still goes to admin routes.
  if (auth && secret && (auth === 'admin' || !openServer)) headers.Authorization = (auth === 'key' ? 'Key ' : 'Bearer ') + secret;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const url = /^https?:/.test(path) ? path : (root || base()) + path;
  let resp;
  try {
    resp = await fetch(url, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch (e) {
    throw new HttpError(0, 'Could not reach ' + (root || base()) + ' (' + e.message + ')');
  }
  const text = await resp.text();
  let parsed = text;
  try { parsed = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!resp.ok) throw new HttpError(resp.status, parsed);
  return full ? { body: parsed, headers: resp.headers, status: resp.status } : parsed;
}

// Statuses a session start answers while the engine is still busy: 409 (a
// streaming session holds the executor; the engine sends `Retry-After`),
// 429 (busy), 503 (loading). A session just stopped on another page frees
// its lease a moment later, so starts retry for a while instead of failing.
export const BUSY = new Set([409, 429, 503]);
export const ADMIT_WAIT_MS = 20000;

// Runs `attempt()` until it does not report busy (`isBusy(result or error)`)
// or ADMIT_WAIT_MS passes, calling `onWait(seconds waited)` between tries.
export async function admit(attempt, { isBusy, onWait = () => {}, every = 1000 } = {}) {
  const t0 = Date.now();
  for (;;) {
    let r; let err = null;
    try { r = await attempt(); } catch (e) { err = e; }
    const busy = isBusy(err || r, !!err);
    if (!busy || Date.now() - t0 > ADMIT_WAIT_MS) { if (err) throw err; return r; }
    onWait(Math.round((Date.now() - t0) / 1000));
    await new Promise((res) => setTimeout(res, every));
  }
}

// The model metadata headers of a fal result or a director session
// (`x-fv-tier`, `x-fv-quality`, `x-fv-recipe`, `x-fv-model`): {tier, quality, recipe, model}.
export function metaHeaders(h) {
  const get = (k) => (h && h.get(k)) || null;
  return { tier: get('x-fv-tier'), quality: get('x-fv-quality'), recipe: get('x-fv-recipe'), model: get('x-fv-model') };
}

// A "draft quality" pill: the tier did not pass the quality gate.
export const draftPill = () => el('span', { class: 'pill warn', 'data-quality': 'draft', title: 'This tier does not pass the quality gate (x-fv-quality: draft): a fast preview, not a final render.' }, 'draft quality');

// A licence label (the LongLive-1.3B weights are non-commercial).
export function licenceBanner(text) {
  if (!text) return null;
  const nc = /non-commercial/i.test(text);
  return el('div', { class: 'banner licence', 'data-licence': nc ? 'non-commercial' : 'other', role: 'note' },
    el('b', {}, nc ? 'Non-commercial licence. ' : 'Licence. '), text);
}

let capsPromise = null;
// `GET /fv/v1/capabilities` (Bearer key): served models with caps, recipe,
// tier and stream limits, tier bindings, aliases, the auth mode and (newer
// servers) the mounted protocols. Null when the native API is not mounted
// or the key is refused.
export function loadCapabilities() {
  if (!capsPromise) capsPromise = request('GET', '/fv/v1/capabilities', { auth: 'bearer' }).catch(() => null);
  return capsPromise;
}

// The protocols this server mounts (`protocols` of the capabilities), or
// null when the server does not say (older servers, the gateway).
export const mountedProtocols = (caps) => (caps && caps.protocols && typeof caps.protocols === 'object' ? caps.protocols : null);

// `{attention, vae, steps, profile}` of a capabilities model's recipe, as text.
export function recipeText(r) {
  if (!r || typeof r !== 'object') return '';
  const parts = [];
  if (r.attention) parts.push('attention ' + r.attention);
  if (r.vae) parts.push('vae ' + r.vae);
  if (r.steps != null) parts.push(r.steps + ' steps');
  if (r.profile) parts.push('profile ' + r.profile);
  return parts.join(' · ');
}

// A capabilities model's stream kind: 'causal', 'duplex', 'clip' or null.
export function streamKind(m) {
  const s = m && m.caps && m.caps.stream;
  if (!s || typeof s !== 'object') return null;
  // `{causal: {...}}`, `{duplex: {...}}` or `{clip: {...}}` (StreamCaps).
  for (const k of ['causal', 'duplex', 'clip']) if (s[k]) return k;
  return null;
}

// Uploads a browser file through the fal storage API -> its URL.
export async function upload(file) {
  await loadAuthMode();
  if (needsKey()) throw new Error('set an API key first');
  const type = file.type || 'application/octet-stream';
  const name = (file.name || 'upload').replace(/[^A-Za-z0-9._-]+/g, '_').slice(-100) || 'upload';
  const init = await request('POST', '/storage/upload/initiate?storage_type=fal-cdn-v3', {
    auth: 'key', body: { content_type: type, file_name: name },
  });
  const put = await fetch(init.upload_url, { method: 'PUT', headers: { 'Content-Type': type }, body: file });
  if (!put.ok) throw new Error('upload failed: HTTP ' + put.status);
  return init.file_url;
}

export function setMsg(node, text, kind) {
  if (typeof node === 'string') node = $(node);
  if (!node) return;
  node.textContent = text || '';
  node.className = 'msg' + (kind ? ' ' + kind : '');
}

export function ago(iso) {
  if (!iso) return 'never';
  const t = typeof iso === 'number' ? iso : Date.parse(iso);
  if (!Number.isFinite(t)) return String(iso);
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 5) return 'just now';
  if (s < 60) return s + ' s ago';
  if (s < 3600) return Math.round(s / 60) + ' min ago';
  if (s < 86400) return Math.round(s / 3600) + ' h ago';
  return new Date(t).toLocaleDateString();
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch { /* fall through */ }
  try {
    const ta = el('textarea', { style: 'position:fixed;opacity:0' }, text);
    document.body.append(ta); ta.select();
    const ok = document.execCommand('copy'); ta.remove(); return ok;
  } catch { return false; }
}

let catalogPromise = null;
// `GET /fal/schema`: the fal apps this server mounts.
export function loadCatalog() {
  if (!catalogPromise) {
    catalogPromise = request('GET', '/fal/schema', { auth: null }).catch((e) => { catalogPromise = null; throw e; });
  }
  return catalogPromise;
}

// The live director page of an app (its tag says when it is a causal rollout).
const directorTask = (app) => ({ sub: 'director', title: 'Director', tag: app && app.director_mode === 'causal' ? 'live · causal' : 'live' });

// An app's pages from its catalog entry: the batch endpoints (sub-paths may
// have several segments, e.g. `v2.2-5b/text-to-video/fast-wan`), then the
// live director when the server says the app has one. A catalog entry
// without `endpoints` (a server older than the served catalog) lists no
// batch endpoints: the model page then asks the server for the endpoint's
// schema directly instead of assuming the H3 set.
export function appTasks(app) {
  if (!app) return [];
  // Per-endpoint tier tags on the family apps (their endpoints differ in tier).
  const list = Array.isArray(app.endpoints)
    ? app.endpoints.map((e) => ({ sub: e.sub, title: e.title || e.sub, tag: app.tier ? null : e.tier || null }))
    : [];
  if (app.director === true) list.push(directorTask(app));
  return list;
}

export const modelHref = (app, sub) => '/console/models/' + app + '/' + sub;

function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}

// ---- server status (GET /fv/v1/status) --------------------------------------
// Public and credential-free: pool and worker states, queue depth, running
// jobs. Polled every POLL_MS while the tab is visible, backing off on errors.

const STATE_TEXT = {
  ready: 'ready', busy: 'busy', loading: 'loading', scaled_to_zero: 'scaled to zero',
  draining: 'draining', unhealthy: 'unhealthy', down: 'down', failed: 'failed',
};
// Dot colour: green ready, amber loading / busy / draining, grey scaled to zero or unknown, red down.
export function stateTone(s) {
  return { ready: 'ok', busy: 'warn', loading: 'warn', draining: 'warn', unhealthy: 'bad', down: 'bad', failed: 'bad' }[s] || 'idle';
}
export const stateText = (s) => STATE_TEXT[s] || s || 'unknown';
export const dot = (state, title) => el('span', { class: 'dot ' + stateTone(state), title, 'data-state': state || 'unknown', 'aria-hidden': 'true' });

const POLL_MS = 7000;
const POLL_MAX_MS = 60000;
let lastStatus = null;
const statusSubs = new Set();
let pollTimer = null;
let pollDelay = POLL_MS;
let pollBusy = false;
let polling = false;

// `cb(status)` on every poll (`status.error` set when the poll failed); now if known.
export function onStatus(cb) {
  statusSubs.add(cb);
  if (lastStatus) cb(lastStatus);
  return () => statusSubs.delete(cb);
}

// The model behind `name` (a fal app's `model`, an alias or an id) and its pools.
export function modelStatus(st, name) {
  if (!st || !st.models || !name) return null;
  const id = (st.names && st.names[name]) || name;
  const m = st.models[id];
  if (!m) return null;
  return { id, state: m.state, pools: (st.pools || []).filter((p) => (m.pools || []).includes(p.id)) };
}

// A warning before submitting to `name`'s model, or null when it can take work.
export function poolWarning(name) {
  const m = modelStatus(lastStatus, name);
  if (!m) return null;
  const where = m.pools.map((p) => p.id).join(', ') || m.id;
  switch (m.state) {
    case 'ready': case 'busy': return null;
    case 'loading': return 'The pool serving ' + m.id + ' (' + where + ') is still loading; the job will wait until it is ready.';
    case 'scaled_to_zero': return 'The pool serving ' + m.id + ' (' + where + ') has no running worker; a cold start can take minutes.';
    case 'draining': return 'The pool serving ' + m.id + ' (' + where + ') is draining and takes no new work.';
    default: return 'The pool serving ' + m.id + ' (' + where + ') is ' + stateText(m.state) + '; the request will probably fail.';
  }
}

function publish(st) {
  lastStatus = st;
  for (const cb of statusSubs) { try { cb(st); } catch { /* a subscriber's bug is not the poller's */ } }
}

async function pollStatus() {
  pollTimer = null;
  if (document.hidden || pollBusy) return;
  pollBusy = true;
  let stop = false;
  try {
    publish(await request('GET', '/fv/v1/status', { auth: null }));
    pollDelay = POLL_MS;
  } catch (e) {
    // 404: a server without the status view; stop asking.
    stop = e.status === 404;
    publish({ error: e.message, http_status: e.status, pools: [], models: {} });
    pollDelay = Math.min(pollDelay * 2, POLL_MAX_MS);
  } finally {
    pollBusy = false;
  }
  if (!stop && !document.hidden) pollTimer = setTimeout(pollStatus, pollDelay);
}

export function startStatusPolling() {
  if (polling) return;
  polling = true;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { clearTimeout(pollTimer); pollTimer = null; } else if (!pollTimer) pollStatus();
  });
  pollStatus();
}

function ageText(s) {
  if (s === null || s === undefined) return 'never seen';
  if (s < 1) return 'just now';
  if (s < 90) return Math.round(s) + ' s ago';
  return Math.round(s / 60) + ' min ago';
}

function poolLine(p) {
  const w = p.workers || [];
  const counts = p.worker_counts ? Object.entries(p.worker_counts).filter(([, n]) => n).map(([k, n]) => n + ' ' + k).join(', ') : '';
  return p.id + ' (' + p.kind + '): ' + stateText(p.state)
    + (w.length ? ', ' + w.length + ' worker' + (w.length > 1 ? 's' : '') : counts ? ', workers: ' + counts : '')
    + ', ' + (p.queued || 0) + ' queued, ' + (p.running || 0) + ' running'
    + (p.versions && p.versions.length ? ', build ' + versionText(p) : '');
}

// `abc1234 (stable)`, or every build with its worker count when the pool is mixed.
function versionText(p) {
  const v = p.versions || [];
  const one = (x) => x.sha + (x.channel ? ' (' + x.channel + ')' : '');
  return v.length === 1 ? one(v[0]) : v.map((x) => one(x) + ' ×' + x.workers).join(', ');
}

function renderStatus(strip, panel, st) {
  if (st.error) {
    strip.replaceChildren(dot(null), el('span', { class: 'status-label' }, st.http_status === 404 ? 'no status' : 'status unavailable'));
    strip.title = 'Server status: ' + st.error;
    panel.replaceChildren(el('p', { class: 'hint' }, 'Could not read ' + base() + '/fv/v1/status: ' + st.error));
    strip.hidden = st.http_status === 404;
    return;
  }
  strip.hidden = false;
  const pools = st.pools || [];
  strip.replaceChildren(...pools.map((p) => el('span', { class: 'status-pool', 'data-pool': p.id, 'data-state': p.state },
    dot(p.state), el('span', { class: 'status-label' }, p.id))));
  if (!pools.length) strip.append(dot('down'), el('span', { class: 'status-label' }, 'no pools'));
  strip.title = pools.map(poolLine).join('\n') || 'No pools';
  panel.replaceChildren(el('div', { class: 'status-scroll' }, el('table', { class: 'status-table' },
    el('thead', {}, el('tr', {}, ['Pool', 'State', 'Workers', 'Queued', 'Running', 'Last seen', 'Build', 'Models'].map((h) => el('th', {}, h)))),
    el('tbody', {}, pools.map((p) => el('tr', { 'data-pool': p.id },
      el('td', {}, dot(p.state), ' ', p.id, el('small', {}, ' ' + p.kind)),
      el('td', {}, stateText(p.state), p.loading ? ' (' + p.loading.done + '/' + p.loading.total + ')' : ''),
      el('td', {}, (p.workers || []).length
        ? (p.workers || []).map((w) => el('div', { 'data-worker': w.label, 'data-state': w.state }, dot(w.state), ' ', w.label + ' ' + stateText(w.state)
          + ' · ' + ageText(w.last_seen_s) + (w.running || w.queued ? ' · ' + w.running + ' running, ' + w.queued + ' queued' : '')))
        : p.worker_counts ? Object.entries(p.worker_counts).map(([k, n]) => n + ' ' + k).join(', ') || 'none' : 'none'),
      el('td', {}, String(p.queued || 0)),
      el('td', {}, String(p.running || 0)),
      el('td', {}, ageText(p.last_seen_s)),
      el('td', { class: 'mono', 'data-versions': (p.versions || []).map((x) => x.sha).join(' ') },
        (p.versions || []).length ? versionText(p) : '–',
        p.mixed_versions ? el('span', { class: 'pill warn', title: 'the workers run different builds' }, ' mixed') : ''),
      el('td', {}, (p.models || []).join(', '))))))));
}

// A "pool: state" badge for `name`'s model (the model and director pages).
export function poolBadge(name) {
  const node = el('span', { class: 'pool-badge', id: 'pool-state', hidden: true });
  onStatus((st) => {
    const m = modelStatus(st, name);
    if (!m) { node.hidden = true; return; }
    node.hidden = false;
    node.dataset.state = m.state;
    node.replaceChildren(dot(m.state), ' ', (m.pools.map((p) => p.id).join(', ') || m.id) + ': ' + stateText(m.state));
    node.title = m.pools.map(poolLine).join('\n');
  });
  return node;
}

// The top bar: brand, nav, status strip, connection pill, theme toggle.
export function topbar(active) {
  applyTheme(store.get(K.theme));
  const link = (href, text, id) => el('a', { href, 'aria-current': active === id ? 'page' : undefined }, text);
  const pill = el('span', { id: 'conn', class: 'pill' });
  connPill(pill);
  loadAuthMode().then(() => refreshConnPill());
  const theme = el('button', { class: 'small', type: 'button', title: 'Switch light / dark / system theme' });
  const label = () => { theme.textContent = { light: 'Light', dark: 'Dark' }[store.get(K.theme)] || 'Auto'; };
  theme.onclick = () => {
    const next = { '': 'light', light: 'dark', dark: '' }[store.get(K.theme) || ''];
    store.set(K.theme, next); applyTheme(next); label();
  };
  label();
  const strip = el('button', { id: 'status-strip', class: 'status-strip', type: 'button', 'aria-expanded': 'false', 'aria-controls': 'status-panel', title: 'Server status' },
    dot(null), el('span', { class: 'status-label' }, 'status…'));
  const panel = el('div', { id: 'status-panel', class: 'status-panel', hidden: true });
  const bar = el('header', { class: 'topbar' },
    el('div', { class: 'topbar-in' },
      el('a', { class: 'brand', href: '/console' }, 'fv-serve', el('small', {}, 'console')),
      el('nav', { class: 'topnav', 'aria-label': 'Console' },
        link('/console', 'Models', 'home'), link('/console/stream', 'Live stream', 'stream'), link('/console/live', 'Live input', 'live'),
        link('/console/native', 'Native API', 'native'), link('/console/avatar', 'Avatar', 'avatar'),
        link('/console/admin', 'API keys', 'admin'), link('/console/deployments', 'Deployments', 'deployments')),
      el('span', { class: 'spacer' }), strip, pill, theme),
    panel);
  strip.onclick = () => {
    panel.hidden = !panel.hidden;
    strip.setAttribute('aria-expanded', String(!panel.hidden));
  };
  onStatus((st) => renderStatus(strip, panel, st));
  startStatusPolling();
  document.body.prepend(bar);
  return bar;
}

function connPill(p) {
  if (openServer) { p.textContent = 'no key needed'; p.className = 'pill ok'; return; }
  p.textContent = apiKey() ? 'key set' : 'no API key';
  p.className = 'pill' + (apiKey() ? ' ok' : '');
}

export function refreshConnPill() {
  const p = $('conn');
  if (p) connPill(p);
}
