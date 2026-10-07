// A model page: /console/models/{owner}/{alias}/{task}. Like a fal.ai model
// page: Playground (schema form + result) and API (snippets) tabs, with
// navigation between the app's endpoints and the owner's other apps.

import {
  $, el, store, K, base, request, setMsg, ago, copyText, loadCatalog, appTasks, modelHref, topbar,
  loadAuthMode, needsKey, poolBadge, poolWarning, upload, metaHeaders, draftPill, licenceBanner,
  loadCapabilities, mountedProtocols, BASE_PATH,
} from './common.js';
import { buildForm } from './form.js';
import { ClientTrace, TRACE_KEY, analyze } from './trace.js';
import { snippets, protocolSnippets, snippetProtocols } from './snippets.js';
import { reactorModel } from './rtc.js';

topbar('home');

const parts = location.pathname.slice(BASE_PATH.length).replace(/\/+$/, '').split('/').slice(3).map(decodeURIComponent);
// The sub-path may have several segments (`text-to-video/fast`).
const [owner, alias] = parts;
const task = parts.slice(2).join('/');
const app = owner + '/' + alias;
const endpointId = app + '/' + task;
const HISTORY_MAX = 50;

document.title = endpointId + ' · fv-serve console';
$('title').textContent = endpointId;
$('crumbs').append(' / ', owner, ' / ', alias);
// The banner shows once the server is known to need a key (not for auth mode `none`).
loadAuthMode().then(() => { $('key-banner').hidden = !needsKey(); });

function fail(msg) {
  $('page-error').textContent = msg;
  $('page-error').hidden = false;
}

async function header() {
  let apps = [];
  try { ({ apps } = await loadCatalog()); } catch (e) { fail('Could not load the model catalog: ' + e.message); }
  const known = apps.find((a) => a.id === app);
  if (!known) fail('`' + app + '` is not mounted on this server. Mounted: ' + (apps.map((a) => a.id).join(', ') || 'none') + '.');
  const tasks = appTasks(known);
  // Variants: the owner's other apps that have this endpoint.
  const siblings = apps.filter((a) => a.id.split('/')[0] === owner && appTasks(a).some((t) => t.sub === task));
  if (siblings.length > 1) {
    $('variants').replaceChildren(el('span', { class: 'label' }, 'Variant'),
      el('div', { class: 'tabs' }, siblings.map((a) => el('a', {
        href: modelHref(a.id, task), 'aria-current': a.id === app ? 'page' : undefined, 'data-variant': a.id,
      }, a.id.split('/')[1]))));
  }
  $('tasks').replaceChildren(...tasks.map((t) => el('a', {
    href: modelHref(app, t.sub), 'aria-current': t.sub === task ? 'page' : undefined, 'data-task': t.sub,
  }, t.title, t.tag ? el('span', { class: 'tag' }, t.tag) : null)));
  return { known, tasks };
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
  el('td', {}, el('span', { class: 'pill s-' + (h.status || '') }, pretty(h.status)), h.quality === 'draft' ? [' ', draftPill()] : null),
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

// `meta`: the result's `x-fv-tier`, `x-fv-quality` and `x-fv-recipe` headers.
function renderFacts(out, c, meta = {}) {
  const facts = [['request_id', c.id], ['endpoint', app + '/' + c.sub]];
  if (meta.tier) facts.push(['tier', el('span', {}, meta.tier, meta.quality === 'draft' ? [' ', draftPill()] : '')]);
  if (meta.quality) facts.push(['quality', meta.quality]);
  if (meta.recipe) facts.push(['recipe', meta.recipe]);
  if (out && out.video) {
    if (out.video.file_name) facts.push(['file', out.video.file_name]);
    if (out.video.file_size) facts.push(['size', (out.video.file_size / 1e6).toFixed(2) + ' MB']);
  }
  if (out && out.seed != null) facts.push(['seed', String(out.seed)]);
  if (out && out.timings && out.timings.inference != null) facts.push(['inference', Number(out.timings.inference).toFixed(2) + ' s']);
  if (c.tStart) facts.push(['queue wait', fmtS(c.tStart - c.t0)]);
  if (c.tEnd) facts.push(['total', fmtS(c.tEnd - c.t0)]);
  $('facts').replaceChildren(...facts.flatMap(([k, v]) => [el('dt', {}, k), el('dd', { 'data-fact': k }, v)]));
  // A draft-tier result is flagged above the video, not only in the facts.
  $('result-quality').hidden = meta.quality !== 'draft';
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
  $('result-quality').hidden = true;
  renderLogs([]);
  setMsg('result-msg', '');
}

async function fetchResult(c) {
  try {
    const r = await request('GET', '/' + app + '/requests/' + encodeURIComponent(c.id), { full: true, trace: c.trace, traceName: 'result' });
    const out = r.body;
    const meta = metaHeaders(r.headers);
    $('output-json').textContent = JSON.stringify(out, null, 2);
    const url = out && out.video && out.video.url;
    if (c.trace && url) watchVideo(c, url);
    showVideo(url);
    renderFacts(out, c, meta);
    status('COMPLETED');
    setMsg('result-msg', url ? '' : 'Completed without a video URL.', url ? '' : 'bad');
    patchHistory(c.id, {
      status: 'COMPLETED', video_url: url && !url.startsWith('data:') ? url : null, elapsed_ms: c.tEnd ? c.tEnd - c.t0 : null,
      tier: meta.tier, quality: meta.quality, recipe: meta.recipe,
    });
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
    const st = await request('GET', '/' + app + '/requests/' + encodeURIComponent(c.id) + '/status?logs=1', { trace: c.trace, traceName: 'poll' });
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

// ---- tracing (docs/serve/tracing.md) -------------------------------------------
// Marks only while the request runs; the beacon, the pod's events and the
// waterfall come after the video is playable, in an idle callback.
const traceOn = () => { try { return localStorage.getItem(TRACE_KEY) === '1' || /[?&]trace=1/.test(location.search); } catch { return false; } };
$('trace-on').checked = traceOn();
$('trace-on').onchange = () => { try { localStorage.setItem(TRACE_KEY, $('trace-on').checked ? '1' : '0'); } catch { /* private mode */ } };

function watchVideo(c, url) {
  const v = $('video');
  const tr = c.trace;
  const once = (ev, name) => v.addEventListener(ev, () => {
    if (tr.marks[name]) return;
    tr.point(name);
    if (name === 'canplay') {
      tr.resource(url);
      tr.spanMs('e2e', tr.marks.click, tr.marks.canplay);
      const later = globalThis.requestIdleCallback || ((f) => setTimeout(f, 50));
      later(() => shipTrace(c));
    }
  }, { once: true });
  tr.point('video_src');
  once('loadstart', 'video_loadstart');
  once('loadedmetadata', 'video_metadata');
  once('canplay', 'canplay');
}

async function shipTrace(c) {
  const tr = c.trace;
  const path = '/fv/v1/traces/' + tr.id;
  const body = JSON.stringify({ events: tr.events });
  try {
    if (!(navigator.sendBeacon && navigator.sendBeacon(base() + path + '/events', body))) {
      await fetch(base() + path + '/events', { method: 'POST', body, keepalive: true });
    }
  } catch { /* best effort */ }
  let server = [];
  try {
    const d = await request('GET', path, { auth: null });
    server = (d && d.events) || [];
  } catch { /* tracing off on the server */ }
  const own = new Set(tr.events.map((e) => e.name + e.t_wall_ns));
  renderTrace(analyze([...tr.events, ...server.filter((e) => !(e.host === 'client' && own.has(e.name + e.t_wall_ns)))]), tr.id);
}

function renderTrace(res, id) {
  $('trace-tab').hidden = false;
  const total = res.totalMs || 1;
  const pct = (ms) => Math.max(0, Math.min(100, (ms / total) * 100));
  const rows = res.rows.map((r) => el('div', { class: 'trace-row', 'data-comp': r.comp, title: r.host + (r.uncertMs ? ' (±' + r.uncertMs.toFixed(1) + ' ms)' : '') },
    el('span', { class: 'mono t' }, r.startMs.toFixed(1)),
    el('span', { class: 'mono d' }, r.durMs ? r.durMs.toFixed(1) : '·'),
    el('span', { class: 'n' }, r.comp + '.' + r.name + (r.arg != null ? ' [' + r.arg + ']' : '') + (r.clock === 'gpu' ? ' (gpu)' : '')),
    el('span', { class: 'bar' }, el('i', { style: 'left:' + pct(r.startMs) + '%;width:' + Math.max(0.3, pct(r.durMs)) + '%' }))));
  const ph = res.phases.map((p) => el('div', { class: 'trace-row' }, el('span', { class: 'mono d' }, p.ms.toFixed(1)), el('span', { class: 'n' }, p.from + ' → ' + p.to)));
  const offs = Object.entries(res.offsets).map(([h, o]) => h + ' ' + (o.offset / 1e6).toFixed(1) + ' ms ± ' + (o.uncert / 1e6).toFixed(1));
  $('trace-view').replaceChildren(
    el('p', { class: 'hint mono' }, 'trace ' + id + ' · total ' + total.toFixed(1) + ' ms · unaccounted ' + res.unaccountedMs.toFixed(1) + ' ms'),
    el('p', { class: 'hint' }, 'clocks vs ' + res.reference + ': ' + offs.join('; ')),
    el('h3', {}, 'Critical path (ms)'), ...ph,
    el('h3', {}, 'Waterfall (start ms from the click, duration ms)'), ...rows);
}

function track(id, sub, t0, trace) {
  if (current && current.timer) clearTimeout(current.timer);
  current = { id, sub, t0: t0 || Date.now(), trace: trace || null };
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
let proto = 'fal';
// Context for the other protocols' snippets: the served model behind the
// endpoint and its capabilities entry (set at boot).
let snippetCtx = null;
function updateSnippet() {
  if (!form) return;
  const input = form.values();
  const s = proto === 'fal' || !snippetCtx
    ? snippets({ base: base(), app, sub: task, input })
    : protocolSnippets(proto, { ...snippetCtx, base: base(), input });
  $('snippet').textContent = s[lang] || s.curl;
  $('snippet-note').textContent = s.note || (proto === 'fal' ? 'Filled with the inputs currently set in the Playground. Auth is fal\'s Authorization: Key <key>.' : '');
}
function renderProtocolTabs(list) {
  $('protocols').replaceChildren(...list.map((p) => el('button', {
    type: 'button', 'data-proto': p.id, 'aria-selected': String(p.id === proto), title: p.title,
    onclick: () => {
      proto = p.id;
      $('protocols').querySelectorAll('[data-proto]').forEach((x) => x.setAttribute('aria-selected', String(x.dataset.proto === proto)));
      updateSnippet();
    },
  }, p.label)));
  $('protocols').hidden = list.length < 2;
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
// The served model behind this endpoint (the catalog's `model`), for the pool state.
let modelName = null;
// A pool warning already shown: a second Run with the same warning submits.
let armed = null;
async function run() {
  if (!form) return;
  await loadAuthMode();
  if (needsKey()) { setMsg('run-msg', 'Set an API key first (Models page or API keys page).', 'bad'); return; }
  const warn = poolWarning(modelName);
  if (warn && armed !== warn) {
    armed = warn;
    setMsg('run-msg', warn + ' Click Run again to submit anyway.', 'bad');
    return;
  }
  armed = null;
  if (form.busy()) { setMsg('run-msg', 'Wait for the uploads to finish.', 'bad'); return; }
  const invalid = form.validate();
  if (invalid) { setMsg('run-msg', invalid, 'bad'); return; }
  const input = form.values();
  if (!input.prompt || !String(input.prompt).trim()) { setMsg('run-msg', 'Enter a prompt.', 'bad'); return; }
  $('run').disabled = true;
  setMsg('run-msg', '');
  resetResult('Submitting…');
  status('SUBMITTING');
  const t0 = Date.now();
  const trace = clickTrace;
  clickTrace = null;
  try {
    const sub = await request('POST', '/' + endpointId, { auth: 'key', body: input, trace, traceName: 'submit' });
    setMsg('run-msg', 'Queued as ' + sub.request_id + '.', 'ok');
    saveHistory({ request_id: sub.request_id, app, sub: task, input, created_at: t0, status: 'IN_QUEUE' });
    track(sub.request_id, task, t0, trace);
  } catch (e) {
    status('FAILED');
    $('video-empty').textContent = 'Not submitted.';
    setMsg('run-msg', e.message, 'bad');
    $('output-json').textContent = JSON.stringify(e.body ?? e.message, null, 2);
    $('run').disabled = false;
  }
}
// The click is the trace's origin: marked before anything else runs.
let clickTrace = null;
$('run').onclick = () => {
  clickTrace = $('trace-on').checked ? new ClientTrace() : null;
  if (clickTrace) clickTrace.point('click');
  run();
};
$('form').addEventListener('submit', (e) => e.preventDefault());
$('reset').onclick = () => form && form.reset();

// ---- boot -----------------------------------------------------------------------
async function boot() {
  const { known, tasks } = await header();
  const ep = known && Array.isArray(known.endpoints) ? known.endpoints.find((e) => e.sub === task) : null;
  modelName = (ep && ep.model) || (known && known.model) || null;
  if (task === 'director') {
    $('director').hidden = false;
    const { mountDirector } = await import('./director.js');
    mountDirector($('director'), { app, available: !!known, model: modelName });
    return;
  }
  $('run').before(poolBadge(modelName));
  // A catalog without `endpoints` (an older server) does not list them: the schema request decides.
  if (known && Array.isArray(known.endpoints) && !tasks.some((t) => t.sub === task)) { fail('Unknown endpoint `' + task + '`.'); return; }
  if (!known) return;
  if (known.licence) $('page-error').after(licenceBanner(known.licence));
  $('batch').hidden = false;
  $('endpoint-id').textContent = endpointId;
  let schema;
  try {
    schema = await request('GET', '/fal/schema/' + endpointId, { auth: null });
  } catch (e) { fail('Could not load the input schema: ' + e.message); return; }
  $('schema-json').textContent = JSON.stringify(schema, null, 2);
  form = buildForm(schema, $('form'), { upload, onChange: updateSnippet });
  // The other APIs this server mounts that can run this endpoint's model.
  loadCapabilities().then((caps) => {
    // The model name may be an id, a served name, an alias or a tier alias (`h3-max`).
    const tierHit = caps ? (caps.tiers || []).find((t) => t && t.alias === modelName) || null : null;
    const id = (caps && caps.aliases && caps.aliases[modelName]) || (tierHit && tierHit.model) || modelName;
    const entry = caps && (caps.models || []).find((m) => m.caps && (m.caps.id === id || (m.caps.served_names || []).includes(id)));
    snippetCtx = { model: modelName, entry, tierHit, endpoint: ep, app: known, sub: task };
    const protos = mountedProtocols(caps);
    // The Reactor runtime streams one model (`GET /schema`): offered for that one only.
    const reactor = !protos || protos.reactor ? reactorModel() : Promise.resolve(null);
    reactor.then((r) => {
      snippetCtx.reactorModel = r;
      renderProtocolTabs(snippetProtocols(protos, snippetCtx));
    });
  });
  $('form').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run(); });
  renderHistory();
  const m = location.hash.match(/request=([^&]+)/);
  const h = m && history().find((x) => x.request_id === decodeURIComponent(m[1]));
  if (h) loadFromHistory(h);
  updateSnippet();
}
boot();
