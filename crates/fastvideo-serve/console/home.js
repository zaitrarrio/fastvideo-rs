import { $, el, store, K, base, apiKey, request, setMsg, loadCatalog, appTasks, modelHref, topbar, refreshConnPill } from './common.js';

topbar('home');

$('base').value = store.get(K.base);
$('apikey').value = apiKey();

async function check() {
  const pill = $('conn-state');
  const facts = $('server-facts');
  facts.hidden = true;
  if (!apiKey()) {
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
    setMsg('connect-msg', 'Key accepted by ' + base() + '.', 'ok');
    facts.replaceChildren(
      el('dt', {}, 'server'), el('dd', {}, base()),
      el('dt', {}, 'models'), el('dd', {}, models.map((m) => String((m.caps && m.caps.id) || m.id || '')).filter(Boolean).join(', ') || '—'),
    );
    facts.hidden = false;
  } catch (e) {
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
  store.set(K.key, $('apikey').value.trim());
  refreshConnPill();
  check();
  renderModels();
};
$('forget').onclick = () => { store.set(K.key, ''); $('apikey').value = ''; refreshConnPill(); check(); };
$('toggle-key').onclick = () => {
  const f = $('apikey'); const show = f.type === 'password';
  f.type = show ? 'text' : 'password'; $('toggle-key').textContent = show ? 'Hide key' : 'Show key';
};

async function renderModels() {
  const box = $('models');
  try {
    const { apps } = await loadCatalog();
    box.replaceChildren(...apps.map((app) => el('div', { class: 'card' },
      el('h3', {}, app.id, app.tier ? el('span', { class: 'tag' }, app.tier) : null),
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
