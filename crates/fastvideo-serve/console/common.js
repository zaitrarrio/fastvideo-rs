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
// admin token) or null. Throws HttpError on non-2xx.
export async function request(method, path, { auth = 'key', token, body, root } = {}) {
  const headers = { Accept: 'application/json' };
  const secret = token ?? (auth === 'admin' ? session.get(K.admin) : apiKey());
  if (auth && secret) headers.Authorization = (auth === 'key' ? 'Key ' : 'Bearer ') + secret;
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
  return parsed;
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

export const TASKS = [
  { sub: 'text-to-video', title: 'Text to Video' },
  { sub: 'image-to-video', title: 'Image to Video' },
  { sub: 'reference-to-video', title: 'Reference to Video' },
  { sub: 'director', title: 'Director', tag: 'live' },
];

export const modelHref = (app, sub) => '/console/models/' + app + '/' + sub;

function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}

// The top bar: brand, nav, connection pill, theme toggle.
export function topbar(active) {
  applyTheme(store.get(K.theme));
  const link = (href, text, id) => el('a', { href, 'aria-current': active === id ? 'page' : undefined }, text);
  const pill = el('span', { id: 'conn', class: 'pill' }, apiKey() ? 'key set' : 'no API key');
  if (apiKey()) pill.classList.add('ok');
  const theme = el('button', { class: 'small', type: 'button', title: 'Switch light / dark / system theme' });
  const label = () => { theme.textContent = { light: 'Light', dark: 'Dark' }[store.get(K.theme)] || 'Auto'; };
  theme.onclick = () => {
    const next = { '': 'light', light: 'dark', dark: '' }[store.get(K.theme) || ''];
    store.set(K.theme, next); applyTheme(next); label();
  };
  label();
  const bar = el('header', { class: 'topbar' },
    el('div', { class: 'topbar-in' },
      el('a', { class: 'brand', href: '/console' }, 'fv-serve', el('small', {}, 'console')),
      el('nav', { class: 'topnav', 'aria-label': 'Console' },
        link('/console', 'Models', 'home'), link('/console/admin', 'API keys', 'admin')),
      el('span', { class: 'spacer' }), pill, theme));
  document.body.prepend(bar);
  return bar;
}

export function refreshConnPill() {
  const p = $('conn');
  if (!p) return;
  p.textContent = apiKey() ? 'key set' : 'no API key';
  p.className = 'pill' + (apiKey() ? ' ok' : '');
}
