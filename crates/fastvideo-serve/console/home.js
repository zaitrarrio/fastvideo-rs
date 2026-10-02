import {
  $, el, store, K, base, apiKey, request, setMsg, loadCatalog, appTasks, modelHref, topbar, refreshConnPill,
  loadAuthMode, resetAuthMode, keyless, recipeText, streamKind, mountedProtocols,
} from './common.js';

topbar('home');

$('base').value = store.get(K.base);
$('apikey').value = apiKey();

async function check() {
  const pill = $('conn-state');
  const facts = $('server-facts');
  facts.hidden = true;
  pill.textContent = 'checking…'; pill.className = 'pill warn';
  await loadAuthMode();
  // Auth mode `none`: no key field, no key check.
  for (const id of ['key-fields', 'toggle-key', 'forget']) $(id).hidden = keyless();
  refreshConnPill();
  if (!keyless() && !apiKey()) {
    pill.textContent = 'no key'; pill.className = 'pill';
    setMsg('connect-msg', 'Enter an API key (or mint one on the API keys page).');
    return;
  }
  pill.textContent = 'checking…'; pill.className = 'pill warn';
  try {
    // Native capabilities need `Authorization: Bearer`; minted keys work for every API.
    const caps = await request('GET', '/fv/v1/capabilities', { auth: 'bearer' });
    const models = Array.isArray(caps.models) ? caps.models : [];
    pill.textContent = 'connected'; pill.className = 'pill ok';
    setMsg('connect-msg', keyless() ? base() + ' needs no API key (auth mode none).' : 'Key accepted by ' + base() + '.', 'ok');
    const protos = mountedProtocols(caps);
    facts.replaceChildren(
      el('dt', {}, 'server'), el('dd', {}, base()),
      el('dt', {}, 'models'), el('dd', {}, String(models.length)),
      ...(protos ? [el('dt', {}, 'APIs'), el('dd', { id: 'server-apis' }, Object.entries(protos).filter(([, v]) => v).map(([k]) => k).join(', ') || '—')] : []),
    );
    facts.hidden = false;
    renderServed(caps);  } catch (e) {
    if (e.status === 401) {
      pill.textContent = 'key refused'; pill.className = 'pill bad';
    } else if (e.status === 404) {
      pill.textContent = 'reachable'; pill.className = 'pill';
      setMsg('connect-msg', 'The native API is not mounted here; the fal pages still work.', '');
      return;
    } else {
      pill.textContent = 'error'; pill.className = 'pill bad';
    }
    setMsg('connect-msg', e.message, 'bad');
  }
}

$('connect').onclick = () => {
  store.set(K.base, $('base').value.trim().replace(/\/+$/, ''));
  if (!keyless()) store.set(K.key, $('apikey').value.trim());
  resetAuthMode();
  refreshConnPill();
  check();
  renderModels();
};
$('forget').onclick = () => { store.set(K.key, ''); $('apikey').value = ''; refreshConnPill(); check(); };
$('toggle-key').onclick = () => {
  const f = $('apikey'); const show = f.type === 'password';
  f.type = show ? 'text' : 'password'; $('toggle-key').textContent = show ? 'Hide key' : 'Show key';
};

// The served models (`/fv/v1/capabilities`): id, tier, the recipe it runs
// (attention, VAE, steps, profile), tasks, and the live pages for stream
// models (causal: Live stream, with its session-length rule; duplex: Live input).
async function renderServed(caps) {
  const box = $('served');
  const models = (caps && Array.isArray(caps.models)) ? caps.models : [];
  let licences = {};
  try {
    const { apps } = await loadCatalog();
    for (const a of apps || []) if (a.licence && a.model) licences[a.model] = a.licence;
  } catch { licences = {}; }
  const aliasesOf = (id) => Object.entries((caps && caps.aliases) || {}).filter(([, v]) => v === id).map(([k]) => k);
  const rows = models.map((m) => {
    const c = m.caps || {};
    const kind = streamKind(m);
    const l = m.stream_limits;
    const live = kind === 'causal'
      ? el('span', {}, el('a', { href: '/console/stream?model=' + encodeURIComponent(c.id), 'data-live': c.id }, 'Live stream'),
        l ? el('small', { class: 'hint' }, ' ' + l.default_max_s + ' s default, ' + l.hard_max_s + ' s max' + (l.reset_restarts_clock ? '; reset restarts the clock' : '')) : '')
      : kind === 'duplex' ? el('a', { href: '/console/live' }, 'Live input') : kind === 'clip' ? 'clip stream' : '–';
    const lic = licences[c.id] || (c.served_names || []).map((n) => licences[n]).find(Boolean) || m.licence;
    return el('tr', { 'data-model': c.id },
      el('td', { class: 'mono' }, c.id, aliasesOf(c.id).length ? el('div', { class: 'hint' }, 'aka ' + aliasesOf(c.id).join(', ')) : ''),
      el('td', {}, c.tier || '–', lic ? el('div', {}, el('span', { class: 'pill warn', title: lic, 'data-licence': 'non-commercial' }, /non-commercial/i.test(lic) ? 'non-commercial' : 'licence')) : ''),
      el('td', { 'data-recipe': '' }, el('div', { class: 'mono' }, c.recipe || ''), el('div', { class: 'hint' }, recipeText(m.recipe))),
      el('td', {}, (c.tasks || []).join(', ')),
      el('td', {}, live),
      el('td', {}, el('a', { href: '/console/native?model=' + encodeURIComponent(c.id) }, 'Native API')));
  });
  box.replaceChildren(...rows);
  $('served-table').hidden = !rows.length;
  const tiers = (caps && Array.isArray(caps.tiers)) ? caps.tiers : [];
  $('served-tiers').textContent = tiers.length
    ? 'Tier bindings: ' + tiers.map((t) => (t.alias || t.name || [t.family, t.tier].filter(Boolean).join('/')) + ' → ' + (t.model || t.id || '?')).join(', ') + '.'
    : '';
  setMsg('served-msg', rows.length ? '' : 'Connect with a key to list the served models (GET /fv/v1/capabilities).');
}

async function renderModels() {
  const box = $('models');
  try {
    const { apps } = await loadCatalog();
    box.replaceChildren(...apps.map((app) => el('div', { class: 'card', 'data-app': app.id },
      el('h3', {}, app.id, app.tier ? el('span', { class: 'tag' }, app.tier) : null,
        app.licence ? el('span', { class: 'tag warn', title: app.licence, 'data-licence': 'non-commercial' }, /non-commercial/i.test(app.licence) ? 'non-commercial' : 'licence') : null),
      app.model ? el('p', { class: 'hint mono' }, 'model ' + app.model) : null,
      !Array.isArray(app.endpoints) ? el('p', { class: 'hint' }, 'This server does not list the endpoints; open one by its path.')
        : !app.endpoints.length && app.director !== true ? el('p', { class: 'hint', 'data-unserved': '' }, 'Not served here: no model behind ' + (app.model || 'this app') + ' is loaded.') : null,
      el('ul', {}, appTasks(app).map((t) => el('li', {},
        el('a', { href: modelHref(app.id, t.sub), 'data-endpoint': app.id + '/' + t.sub }, t.title),
        t.tag ? el('span', { class: 'tag' }, t.tag) : null))))));
    setMsg('models-msg', apps.length ? '' : 'No fal apps are mounted on this server.');
  } catch (e) {
    box.replaceChildren();
    setMsg('models-msg', 'Could not list models: ' + e.message + ' (is the fal API enabled?)', 'bad');
  }
}

check();
renderModels();
