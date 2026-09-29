// Deployments page (docs/serve/releases.md): channels, live builds, the
// deployment registry and release history from the gateway's admin API;
// Promote / Rollback dispatch the release workflow after a dry-run plan.
import { $, el, session, K, request, setMsg, ago, topbar, dot, stateText } from './common.js';

topbar('deployments');

const token = () => session.get(K.admin);
$('admintoken').value = token();
let dispatchReady = false;

function pill(id, text, kind) {
  const p = $(id);
  p.textContent = text; p.className = 'pill' + (kind ? ' ' + kind : ''); p.hidden = false;
}
const short = (s) => (s || '?').slice(0, 7);
const dshort = (d) => (d ? String(d).split('@').pop().replace('sha256:', '').slice(0, 12) : '–');
const when = (ms) => (ms ? ago(Number(ms)) : '–');
function driftCell(d, follows) {
  if (d === false) return el('span', { class: 'pill ok', 'data-drift': 'ok' }, 'ok' + (follows ? ' (' + follows + ')' : ''));
  if (d === null || d === undefined) return el('span', { class: 'pill', 'data-drift': 'unknown', title: 'no release recorded for this image or channel' }, '?');
  return el('span', { class: 'pill warn', 'data-drift': 'drift', title: (follows || 'the channel') + ' is ' + d }, 'behind: ' + (follows || '') + ' is ' + d);
}
function show(table, empty, rows, emptyText) {
  $(table).tBodies[0].replaceChildren(...rows);
  $(table).hidden = rows.length === 0;
  $(empty).hidden = rows.length !== 0;
  if (emptyText) $(empty).textContent = emptyText;
}

async function load() {
  if (!token()) {
    pill('admin-state', 'not set');
    for (const [t, e] of [['heads', 'heads-empty'], ['live', 'live-empty'], ['deps', 'deps-empty'], ['history', 'history-empty']]) show(t, e, [], 'Enter the admin token.');
    return;
  }
  try {
    const [dep, rel] = await Promise.all([
      request('GET', '/fv/v1/admin/deployments', { auth: 'admin' }),
      request('GET', '/fv/v1/admin/releases?limit=50', { auth: 'admin' }),
    ]);
    pill('admin-state', 'accepted', 'ok');
    setMsg('page-msg', '');
    render(dep, rel);
  } catch (e) {
    if (e.status === 401) pill('admin-state', 'refused', 'bad');
    else if (e.status === 404) pill('admin-state', 'not a gateway', 'bad');
    else pill('admin-state', 'error', 'bad');
    setMsg('page-msg', e.status === 404 ? 'This server is not a gateway: releases and deployments live on the gateway.' : e.message, 'bad');
  }
}

function render(dep, rel) {
  dispatchReady = !!(dep.dispatch && dep.dispatch.configured);
  pill('dispatch-state', dispatchReady ? 'workflow ready' : 'dispatch not configured', dispatchReady ? 'ok' : 'warn');
  $('dispatch-state').title = dispatchReady ? dep.dispatch.repo + ' · ' + dep.dispatch.workflow
    : 'Set FV_GITHUB_TOKEN on the gateway to dispatch; dry runs still work.';
  pill('template-channel', 'templates follow ' + dep.template_channel);

  show('heads', 'heads-empty', (rel.heads || []).map((h) => el('tr', { 'data-channel': h.channel },
    el('td', {}, el('strong', {}, h.channel)),
    el('td', {}, '#' + h.id),
    el('td', { class: 'mono', title: h.git_sha }, short(h.git_sha)),
    el('td', {}, h.action),
    el('td', { title: new Date(Number(h.promoted_at)).toISOString() }, when(h.promoted_at)),
    el('td', {}, h.promoted_by || ''),
    el('td', { class: 'mono' }, Object.keys(h.digests || {}).length + ' images'),
    el('td', {}, el('button', { class: 'small', 'data-rollback': h.channel, onclick: () => rollback(h.channel) }, 'Rollback…')))),
  'No releases recorded yet: promote a build to start the history.');

  const g = dep.gateway || {};
  const liveRows = [el('tr', { 'data-pool': 'gateway' },
    el('td', {}, 'gateway'), el('td', {}, 'this server'), el('td', {}, dot('ready'), ' serving'),
    el('td', { class: 'mono', title: g.git_sha || '' }, g.sha || '?'), el('td', {}, g.channel || '–'),
    el('td', { class: 'mono', title: g.image_digest || '' }, (g.variant || 'debug') + ' ' + dshort(g.image_digest)),
    el('td', {}, driftCell(g.drift, g.follows)))];
  let mixed = false;
  for (const p of dep.pools || []) {
    if (p.mixed_versions) mixed = true;
    const ws = p.workers || [];
    if (!ws.length) {
      liveRows.push(el('tr', { 'data-pool': p.id }, el('td', {}, p.id + (p.mixed_versions ? ' ⚠' : '')), el('td', { colspan: 6, class: 'hint' },
        p.kind === 'runpod-serverless' ? 'serverless: follows its Runpod template' : 'no worker')));
    }
    for (const w of ws) {
      const b = w.build || {};
      liveRows.push(el('tr', { 'data-pool': p.id, 'data-worker': w.id || w.url },
        el('td', {}, p.id, p.mixed_versions ? el('span', { class: 'pill warn', title: 'workers run different builds' }, ' mixed') : ''),
        el('td', { class: 'mono', title: w.url }, w.id || w.url),
        el('td', {}, dot(w.state), ' ', stateText(w.state)),
        el('td', { class: 'mono', title: b.git_sha || '' }, b.sha || '?'),
        el('td', {}, b.channel || '–'),
        el('td', { class: 'mono', title: b.image_digest || '' }, (b.variant || 'debug') + ' ' + dshort(b.image_digest)),
        el('td', {}, driftCell(b.drift, b.follows))));
    }
  }
  show('live', 'live-empty', liveRows);
  $('mixed').hidden = !mixed;

  show('deps', 'deps-empty', (dep.deployments || []).map((d) => el('tr', { 'data-deployment': d.id },
    el('td', {}, d.kind),
    el('td', { class: 'mono' }, d.runpod_id),
    el('td', {}, d.name || ''),
    el('td', {}, [d.pool, d.variant].filter(Boolean).join(' / ') || '–'),
    el('td', { class: 'mono', title: d.digest || '' }, short(d.git_sha)),
    el('td', {}, el('span', { class: 'pill' + (d.status === 'ready' ? ' ok' : d.status === 'failed' ? ' bad' : '') }, d.status)),
    el('td', { title: d.created_at ? new Date(Number(d.created_at)).toISOString() : '' }, when(d.created_at)),
    el('td', {}, d.created_by || ''),
    el('td', {}, driftCell(d.drift, d.follows)))), 'No live deployments recorded.');

  show('history', 'history-empty', (rel.history || []).map((h) => el('tr', { 'data-release': h.id },
    el('td', {}, String(h.id)),
    el('td', {}, h.channel),
    el('td', {}, h.action + (h.source_release ? ' of #' + h.source_release : '')),
    el('td', { class: 'mono', title: h.git_sha }, short(h.git_sha), h.rolled_back_at ? el('small', {}, ' rolled back') : ''),
    el('td', {}, when(h.promoted_at)),
    el('td', {}, h.promoted_by || ''),
    el('td', {}, h.notes || ''))), 'No releases recorded yet.');
}

function planText(r) {
  const lines = [r.inputs.action + ' → ' + r.inputs.channel];
  if (r.current && r.current.id) lines.push('current: #' + r.current.id + ' ' + short(r.current.git_sha) + ' (' + r.current.action + ')');
  if (r.target) lines.push('target:  #' + r.target.id + ' ' + short(r.target.git_sha) + ' (' + r.target.action + ')');
  if (r.inputs.target) lines.push('build:   ' + r.inputs.target);
  lines.push('templates: ' + (r.templates ? 'updated (the channel they follow)' : 'unchanged'));
  lines.push('workflow: ' + r.workflow + ' @ ' + r.ref + (r.dispatch && r.dispatch.configured ? '' : ' (dispatch not configured)'));
  return lines.join('\n');
}

async function run(path, body, confirmText) {
  const plan = await request('POST', path, { auth: 'admin', body: { ...body, dry_run: true } });
  $('plan-out').textContent = planText(plan); $('plan-out').hidden = false;
  if (!confirmText) return plan;
  if (!plan.dispatch || !plan.dispatch.configured) throw new Error('Dispatch is not configured on this gateway (FV_GITHUB_TOKEN); run the release workflow or scripts/serve/release.sh instead.');
  if (!confirm(confirmText + '\n\n' + planText(plan))) return null;
  const done = await request('POST', path, { auth: 'admin', body: { ...body, dry_run: false } });
  return done;
}

function promoteBody() {
  return { target: $('target').value.trim(), channel: $('channel').value, notes: $('notes').value.trim() };
}

$('plan').onclick = async () => {
  if (!token()) return setMsg('promote-msg', 'Enter the admin token first.', 'bad');
  if (!promoteBody().target) return setMsg('promote-msg', 'Name a build: a git sha, a digest or a tag.', 'bad');
  try { await run('/fv/v1/admin/releases/promote', promoteBody(), null); setMsg('promote-msg', 'Plan only: nothing was changed.', 'ok'); }
  catch (e) { setMsg('promote-msg', e.message, 'bad'); }
};
$('promote').onclick = async () => {
  if (!token()) return setMsg('promote-msg', 'Enter the admin token first.', 'bad');
  const b = promoteBody();
  if (!b.target) return setMsg('promote-msg', 'Name a build: a git sha, a digest or a tag.', 'bad');
  try {
    const r = await run('/fv/v1/admin/releases/promote', b, 'Promote ' + b.target + ' to ' + b.channel + '?');
    if (r) { setMsg('promote-msg', 'Dispatched: follow it at ' + r.dispatch.runs_url, 'ok'); setTimeout(load, 5000); }
  } catch (e) { setMsg('promote-msg', e.message, 'bad'); }
};

async function rollback(channel) {
  try {
    const r = await run('/fv/v1/admin/releases/rollback', { channel }, 'Roll ' + channel + ' back?');
    if (r) { setMsg('promote-msg', 'Rollback dispatched: follow it at ' + r.dispatch.runs_url, 'ok'); setTimeout(load, 5000); }
    else setMsg('promote-msg', 'Rollback cancelled; the plan is shown above.', '');
  } catch (e) { setMsg('promote-msg', e.message, 'bad'); }
}

$('admin-save').onclick = () => { session.set(K.admin, $('admintoken').value.trim()); load(); };
$('admintoken').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('admin-save').click(); });
$('admin-clear').onclick = () => { session.set(K.admin, ''); $('admintoken').value = ''; load(); };
$('refresh').onclick = load;
load();
