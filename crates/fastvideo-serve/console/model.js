// A model page: /console/models/{owner}/{alias}/{task}. Like a fal.ai model
// page: Playground (schema form + result) and API (snippets) tabs, with
// navigation between the app's endpoints and the owner's other apps.

import {
  $, el, store, K, base, apiKey, request, setMsg, ago, copyText, loadCatalog, TASKS, modelHref, topbar,
} from './common.js';
import { buildForm } from './form.js';
import { snippets } from './snippets.js';

topbar('home');

const parts = location.pathname.replace(/\/+$/, '').split('/').slice(3).map(decodeURIComponent);
const [owner, alias, task] = parts;
const app = owner + '/' + alias;
const endpointId = app + '/' + task;
const HISTORY_MAX = 50;

document.title = endpointId + ' · fv-serve console';
$('title').textContent = endpointId;
$('crumbs').append(' / ', owner, ' / ', alias);
$('key-banner').hidden = !!apiKey();

function fail(msg) {
  $('page-error').textContent = msg;
  $('page-error').hidden = false;
}

async function header() {
  let apps = [];
  try { ({ apps } = await loadCatalog()); } catch (e) { fail('Could not load the model catalog: ' + e.message); }
  const known = apps.find((a) => a.id === app);
  if (!known) fail('`' + app + '` is not mounted on this server. Mounted: ' + (apps.map((a) => a.id).join(', ') || 'none') + '.');
  const siblings = apps.filter((a) => a.id.split('/')[0] === owner);
  if (siblings.length > 1) {
    $('variants').replaceChildren(el('span', { class: 'label' }, 'Variant'),
      el('div', { class: 'tabs' }, siblings.map((a) => el('a', {
        href: modelHref(a.id, task), 'aria-current': a.id === app ? 'page' : undefined, 'data-variant': a.id,
      }, a.id.split('/')[1]))));
  }
  $('tasks').replaceChildren(...TASKS.map((t) => el('a', {
    href: modelHref(app, t.sub), 'aria-current': t.sub === task ? 'page' : undefined, 'data-task': t.sub,
  }, t.title, t.tag ? el('span', { class: 'tag' }, t.tag) : null)));
  return known;
}

// ---- uploads ---------------------------------------------------------------
async function upload(file) {
  if (!apiKey()) throw new Error('set an API key first');
  const type = file.type || 'application/octet-stream';
  const name = (file.name || 'upload').replace(/[^A-Za-z0-9._-]+/g, '_').slice(-100) || 'upload';
  const init = await request('POST', '/storage/upload/initiate?storage_type=fal-cdn-v3', {
    auth: 'key', body: { content_type: type, file_name: name },
  });
  const put = await fetch(init.upload_url, { method: 'PUT', headers: { 'Content-Type': type }, body: file });
  if (!put.ok) throw new Error('upload failed: HTTP ' + put.status);
  return init.file_url;
}

// ---- history -----------------------------------------------------------------
const history = () => store.json(K.history, []);
function saveHistory(entry) {
  const list = history().filter((h) => h.request_id !== entry.request_id);
  list.unshift(entry);
  store.setJson(K.history, list.slice(0, HISTORY_MAX));
  renderHistory();
}
function patchHistory(id, patch) {
  const list = history();
  const h = list.find((x) => x.request_id === id);
  if (!h) return;
  Object.assign(h, patch);
  store.setJson(K.history, list);
  renderHistory();
}
function renderHistory() {
  const list = history().filter((h) => h.app === app);
  const body = $('history').tBodies[0];
  body.replaceChildren(...list.map((h) => el('tr', {
    class: 'clickable' + (current && current.id === h.request_id ? ' selected' : ''), 'data-request': h.request_id,
    onclick: () => loadFromHistory(h),
  },
  el('td', {}, el('span', { class: 'pill s-' + (h.status || '') }, pretty(h.status))),
  el('td', { class: 'mono' }, h.sub),
  el('td', { class: 'clip', title: (h.input && h.input.prompt) || '' }, (h.input && h.input.prompt) || ''),
  el('td', { class: 'mono' }, h.request_id.slice(0, 13) + '…'),
  el('td', { title: new Date(h.created_at).toISOString() }, ago(h.created_at)))));
  $('history').hidden = list.length === 0;
  $('history-empty').hidden = list.length !== 0;
}
$('clear-history').onclick = () => { store.setJson(K.history, history().filter((h) => h.app !== app)); renderHistory(); };

// ---- result pane ----------------------------------------------------------------
const pretty = (s) => ({ IN_QUEUE: 'in queue', IN_PROGRESS: 'in progress', COMPLETED: 'completed', FAILED: 'failed', SUBMITTING: 'submitting', CANCELLED: 'cancelled' }[s] || (s || '').toLowerCase());
let current = null; // {id, sub, t0, tStart, tEnd, timer}

function status(s) {
  const p = $('result-status');
  p.hidden = !s;
  p.textContent = pretty(s);
  p.className = 'pill s-' + s;
  $('cancel').hidden = !(s === 'IN_QUEUE' || s === 'IN_PROGRESS');
}

document.querySelectorAll('[data-rtab]').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('[data-rtab]').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
  document.querySelectorAll('[data-rpanel]').forEach((p) => { p.hidden = p.dataset.rpanel !== b.dataset.rtab; });
}));

function renderLogs(logs) {
  const box = $('logs');
  if (!logs || !logs.length) { box.replaceChildren(el('div', { class: 'lv' }, 'No logs yet.')); return; }
  box.replaceChildren(...logs.map((l) => el('div', { class: l.level || '' },
    el('span', { class: 'lv' }, (l.timestamp || '').replace(/^.*T/, '').replace('Z', '') + ' ' + (l.level || '')), l.message)));
  box.scrollTop = box.scrollHeight;
}

function fmtS(ms) { return ms == null ? '—' : (ms / 1000).toFixed(2) + ' s'; }

function renderFacts(out, c) {
  const facts = [['request_id', c.id], ['endpoint', app + '/' + c.sub]];
  if (out && out.video) {
    if (out.video.file_name) facts.push(['file', out.video.file_name]);
    if (out.video.file_size) facts.push(['size', (out.video.file_size / 1e6).toFixed(2) + ' MB']);
  }
  if (out && out.seed != null) facts.push(['seed', String(out.seed)]);
  if (out && out.timings && out.timings.inference != null) facts.push(['inference', Number(out.timings.inference).toFixed(2) + ' s']);
  if (c.tStart) facts.push(['queue wait', fmtS(c.tStart - c.t0)]);
  if (c.tEnd) facts.push(['total', fmtS(c.tEnd - c.t0)]);
  $('facts').replaceChildren(...facts.flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, v)]));
}

function showVideo(url) {
  const v = $('video');
  if (url) {
    v.src = url; v.hidden = false; $('video-empty').hidden = true;
    $('download').href = url; $('open-video').href = url; $('result-links').hidden = false;
  } else {
    v.removeAttribute('src'); v.hidden = true; $('video-empty').hidden = false; $('result-links').hidden = true;
  }
}

function resetResult(text) {
  showVideo(null);
  $('video-empty').textContent = text || 'Run a request to see the video here.';
  $('output-json').textContent = '{}';
  $('facts').replaceChildren();
  renderLogs([]);
  setMsg('result-msg', '');
}

async function fetchResult(c) {
  try {
    const out = await request('GET', '/' + app + '/requests/' + encodeURIComponent(c.id));
    $('output-json').textContent = JSON.stringify(out, null, 2);
    const url = out && out.video && out.video.url;
    showVideo(url);
    renderFacts(out, c);
    status('COMPLETED');
    setMsg('result-msg', url ? '' : 'Completed without a video URL.', url ? '' : 'bad');
    patchHistory(c.id, { status: 'COMPLETED', video_url: url && !url.startsWith('data:') ? url : null, elapsed_ms: c.tEnd ? c.tEnd - c.t0 : null });
  } catch (e) {
    $('output-json').textContent = JSON.stringify(e.body ?? e.message, null, 2);
    showVideo(null);
    $('video-empty').textContent = 'No video.';
    status('FAILED');
    setMsg('result-msg', e.message, 'bad');
    renderFacts(null, c);
    patchHistory(c.id, { status: 'FAILED', error: e.message });
  }
}

async function poll(c) {
  if (current !== c) return;
  try {
    const st = await request('GET', '/' + app + '/requests/' + encodeURIComponent(c.id) + '/status?logs=1');
    if (current !== c) return;
    status(st.status);
    if (st.logs) renderLogs(st.logs);
    if (st.status === 'IN_QUEUE') {
      $('video-empty').textContent = 'In queue' + (st.queue_position ? ' (position ' + st.queue_position + ')' : '') + '…';
    } else if (st.status === 'IN_PROGRESS') {
      c.tStart = c.tStart || Date.now();
      $('video-empty').textContent = 'Generating… ' + fmtS(Date.now() - c.t0);
    }
    patchHistory(c.id, { status: st.status });
    if (st.status === 'COMPLETED') {
      c.tStart = c.tStart || Date.now();
      c.tEnd = Date.now();
      await fetchResult(c);
      $('run').disabled = false;
      return;
    }
  } catch (e) {
    if (e.status === 404) { status(''); setMsg('result-msg', 'This request is no longer known to the server (expired or restarted).', 'bad'); $('run').disabled = false; return; }
    setMsg('result-msg', e.message + ' (retrying)', 'bad');
  }
  c.timer = setTimeout(() => poll(c), 700);
}

function track(id, sub, t0) {
  if (current && current.timer) clearTimeout(current.timer);
  current = { id, sub, t0: t0 || Date.now() };
  renderHistory();
  poll(current);
}

$('cancel').onclick = async () => {
  if (!current) return;
  try {
    await request('PUT', '/' + app + '/requests/' + encodeURIComponent(current.id) + '/cancel');
    setMsg('result-msg', 'Cancellation requested.');
  } catch (e) { setMsg('result-msg', e.message, 'bad'); }
};

function loadFromHistory(h) {
  if (h.sub !== task) { location.href = modelHref(app, h.sub) + '#request=' + encodeURIComponent(h.request_id); return; }
  if (form && h.input) form.setValues(h.input);
  resetResult('Loading ' + h.request_id + '…');
  track(h.request_id, h.sub, h.created_at);
}

// ---- tabs, snippets -----------------------------------------------------------
let form = null;
let lang = 'curl';
function updateSnippet() {
  if (!form) return;
  const s = snippets({ base: base(), app, sub: task, input: form.values() });
  $('snippet').textContent = s[lang];
}
document.querySelectorAll('[data-lang]').forEach((b) => b.addEventListener('click', () => {
  lang = b.dataset.lang;
  document.querySelectorAll('[data-lang]').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
  updateSnippet();
}));
$('copy-snippet').onclick = async () => { const ok = await copyText($('snippet').textContent); $('copy-snippet').textContent = ok ? 'Copied' : 'Copy failed'; setTimeout(() => { $('copy-snippet').textContent = 'Copy'; }, 1500); };
function view(which) {
  $('tab-playground').setAttribute('aria-selected', String(which === 'playground'));
  $('tab-api').setAttribute('aria-selected', String(which === 'api'));
  $('playground').hidden = which !== 'playground';
  $('api').hidden = which !== 'api';
  if (which === 'api') updateSnippet();
}
$('tab-playground').onclick = () => view('playground');
$('tab-api').onclick = () => view('api');

// ---- run ----------------------------------------------------------------------
async function run() {
  if (!form) return;
  if (!apiKey()) { setMsg('run-msg', 'Set an API key first (Models page or API keys page).', 'bad'); return; }
  if (form.busy()) { setMsg('run-msg', 'Wait for the uploads to finish.', 'bad'); return; }
  const input = form.values();
  if (!input.prompt || !String(input.prompt).trim()) { setMsg('run-msg', 'Enter a prompt.', 'bad'); return; }
  $('run').disabled = true;
  setMsg('run-msg', '');
  resetResult('Submitting…');
  status('SUBMITTING');
  const t0 = Date.now();
  try {
    const sub = await request('POST', '/' + endpointId, { auth: 'key', body: input });
    setMsg('run-msg', 'Queued as ' + sub.request_id + '.', 'ok');
    saveHistory({ request_id: sub.request_id, app, sub: task, input, created_at: t0, status: 'IN_QUEUE' });
    track(sub.request_id, task, t0);
  } catch (e) {
    status('FAILED');
    $('video-empty').textContent = 'Not submitted.';
    setMsg('run-msg', e.message, 'bad');
    $('output-json').textContent = JSON.stringify(e.body ?? e.message, null, 2);
    $('run').disabled = false;
  }
}
$('run').onclick = run;
$('form').addEventListener('submit', (e) => e.preventDefault());
$('reset').onclick = () => form && form.reset();

// ---- boot -----------------------------------------------------------------------
async function boot() {
  const known = await header();
  if (task === 'director') {
    $('director').hidden = false;
    const { mountDirector } = await import('./director.js');
    mountDirector($('director'), { app, available: !!known });
    return;
  }
  if (!TASKS.some((t) => t.sub === task)) { fail('Unknown endpoint `' + task + '`.'); return; }
  if (!known) return;
  $('batch').hidden = false;
  $('endpoint-id').textContent = endpointId;
  let schema;
  try {
    schema = await request('GET', '/fal/schema/' + endpointId, { auth: null });
  } catch (e) { fail('Could not load the input schema: ' + e.message); return; }
  $('schema-json').textContent = JSON.stringify(schema, null, 2);
  form = buildForm(schema, $('form'), { upload, onChange: updateSnippet });
  $('form').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run(); });
  renderHistory();
  const m = location.hash.match(/request=([^&]+)/);
  const h = m && history().find((x) => x.request_id === decodeURIComponent(m[1]));
  if (h) loadFromHistory(h);
  updateSnippet();
}
boot();
