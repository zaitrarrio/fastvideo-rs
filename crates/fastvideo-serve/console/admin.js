import { $, el, store, session, K, request, setMsg, ago, copyText, topbar, refreshConnPill } from './common.js';

topbar('admin');

const token = () => session.get(K.admin);
$('admintoken').value = token();

function adminState(text, kind) {
  const p = $('admin-state');
  p.textContent = text; p.className = 'pill' + (kind ? ' ' + kind : '');
}

async function listKeys() {
  const table = $('keys');
  if (!token()) {
    table.hidden = true; $('keys-empty').hidden = false;
    $('keys-empty').textContent = 'Enter the admin token to list keys.';
    adminState('not set');
    return;
  }
  try {
    const { keys, backend } = await request('GET', '/fv/v1/admin/keys', { auth: 'admin' });
    adminState('accepted', 'ok');
    $('backend').textContent = 'store: ' + backend; $('backend').hidden = false;
    table.tBodies[0].replaceChildren(...keys.map((k) => {
      const revoke = k.revoked ? null : el('button', {
        class: 'link', 'data-revoke': k.id,
        onclick: async () => {
          if (!confirm('Revoke "' + k.name + '"? Clients using it stop working immediately.')) return;
          try { await request('DELETE', '/fv/v1/admin/keys/' + encodeURIComponent(k.id), { auth: 'admin' }); await listKeys(); }
          catch (e) { setMsg('admin-msg', e.message, 'bad'); }
        },
      }, 'revoke');
      return el('tr', { 'data-key-id': k.id },
        el('td', {}, k.name),
        el('td', { class: 'mono' }, k.id),
        el('td', { class: 'mono' }, k.prefix),
        el('td', { title: k.created_at || '' }, ago(k.created_at)),
        el('td', { title: k.last_used_at || '' }, ago(k.last_used_at)),
        el('td', {}, el('span', { class: 'pill ' + (k.revoked ? 'bad' : 'ok') }, k.revoked ? 'revoked' : 'active')),
        el('td', {}, revoke));
    }));
    table.hidden = keys.length === 0;
    $('keys-empty').hidden = keys.length !== 0;
    $('keys-empty').textContent = 'No keys yet. Create one above.';
    setMsg('admin-msg', '');
  } catch (e) {
    table.hidden = true; $('keys-empty').hidden = false;
    $('keys-empty').textContent = 'Keys are hidden until the admin token is accepted.';
    if (e.status === 401) adminState('refused', 'bad'); else adminState('error', 'bad');
    setMsg('admin-msg', e.message, 'bad');
  }
}

$('admin-save').onclick = () => { session.set(K.admin, $('admintoken').value.trim()); listKeys(); };
$('admintoken').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('admin-save').click(); });
$('admin-clear').onclick = () => { session.set(K.admin, ''); $('admintoken').value = ''; listKeys(); };
$('refresh').onclick = listKeys;

$('mint').onclick = async () => {
  const name = $('keyname').value.trim();
  if (!name) return setMsg('mint-msg', 'Give the key a name so you can tell it apart later.', 'bad');
  if (!token()) return setMsg('mint-msg', 'Enter the admin token first.', 'bad');
  try {
    const created = await request('POST', '/fv/v1/admin/keys', { auth: 'admin', body: { name } });
    $('minted-key').textContent = created.api_key;
    $('minted').hidden = false;
    $('keyname').value = '';
    setMsg('mint-msg', 'Created "' + created.key.name + '" (' + created.key.id + ').', 'ok');
    await listKeys();
  } catch (e) { setMsg('mint-msg', e.message, 'bad'); }
};
$('keyname').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('mint').click(); });
$('use-minted').onclick = () => {
  store.set(K.key, $('minted-key').textContent);
  refreshConnPill();
  setMsg('mint-msg', 'Saved as this browser\'s API key. Open a model from the Models page.', 'ok');
};
$('copy-minted').onclick = async () => {
  const ok = await copyText($('minted-key').textContent);
  setMsg('mint-msg', ok ? 'Copied.' : 'Clipboard unavailable here; select the key and copy it.', ok ? 'ok' : 'bad');
};
$('hide-minted').onclick = () => { $('minted-key').textContent = ''; $('minted').hidden = true; };

listKeys();
