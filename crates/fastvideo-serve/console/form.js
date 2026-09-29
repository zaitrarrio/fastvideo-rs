// A form built from an endpoint's input JSON Schema (`GET /fal/schema/...`).
//
// Primary fields come first; properties marked `x-fv-advanced` go under
// "Additional settings". `x-fv-media` string / array properties become drop
// zones that upload through `upload(file) -> url` and also take pasted URLs.

import { el } from './common.js';

const nullable = (p) => Array.isArray(p.anyOf) && p.anyOf.some((a) => a.type === 'null');
const baseType = (p) => p.type || (Array.isArray(p.anyOf) ? (p.anyOf.find((a) => a.type !== 'null') || {}).type : undefined);
const inner = (p) => (Array.isArray(p.anyOf) ? p.anyOf.find((a) => a.type !== 'null') || {} : p);

function human(name) {
  const s = name.replace(/_urls?$/, (m) => (m === '_urls' && !name.includes('audio') ? 's' : '')).replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const ACCEPT = { image: 'image/*', video: 'video/*', audio: 'audio/*' };

function previewNode(kind, src) {
  if (kind === 'image') return el('img', { src, alt: '' });
  if (kind === 'video') return el('video', { src, muted: true, playsinline: true, preload: 'metadata' });
  if (kind === 'audio') return el('div', { class: 'cap' }, '♪ audio');
  return null;
}

function mediaField(name, prop, { upload, onChange, multiple }) {
  const kind = prop['x-fv-media'];
  const max = multiple ? prop.maxItems || 12 : 1;
  const items = []; // {url, preview, name, busy}
  const list = el('div', { class: 'media-items' });
  const file = el('input', { type: 'file', accept: ACCEPT[kind] || '', multiple: multiple || undefined, hidden: true, 'data-field-file': name });
  const status = el('div', { class: 'hint' });
  const what = multiple ? kind + ' files' : { image: 'an image', audio: 'an audio clip', video: 'a video' }[kind] || 'a file';
  const drop = el('div', { class: 'drop', 'data-drop': name },
    'Drop ' + what + ' here or ',
    el('button', { type: 'button', class: 'small', onclick: () => file.click() }, 'Choose file'));
  const urlInput = el('input', { type: 'url', placeholder: 'or paste a URL (https:// or data:)', spellcheck: 'false', 'data-field-url': name });
  const addUrl = el('button', { type: 'button', class: 'small' }, multiple ? 'Add' : 'Use');

  function render() {
    list.replaceChildren(...items.map((it, i) => {
      const node = el('div', { class: 'media-item' + (it.busy ? ' busy' : ''), title: it.url || it.name || '' },
        previewNode(kind, it.preview || it.url),
        el('div', { class: 'cap' }, (multiple ? human(kind) + ' ' + (i + 1) + ' · ' : '') + (it.name || (it.url || '').split('/').pop().split('?')[0].slice(0, 40))),
        el('button', { type: 'button', class: 'rm', title: 'Remove', onclick: () => { items.splice(i, 1); render(); onChange(); } }, '×'));
      return node;
    }));
    drop.hidden = items.length >= max;
    urlInput.parentElement && (urlInput.parentElement.hidden = items.length >= max);
  }

  async function addFiles(files) {
    for (const f of [...files].slice(0, max - items.length)) {
      if (!multiple) items.length = 0;
      const it = { name: f.name, preview: URL.createObjectURL(f), busy: true, url: '' };
      items.push(it); render();
      status.textContent = 'Uploading ' + f.name + '…';
      try {
        it.url = await upload(f);
        it.busy = false;
        status.textContent = '';
      } catch (e) {
        items.splice(items.indexOf(it), 1);
        status.textContent = 'Upload failed: ' + e.message;
      }
      render(); onChange();
    }
  }
  file.onchange = () => { addFiles(file.files); file.value = ''; };
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); addFiles(e.dataTransfer.files); });
  addUrl.onclick = () => {
    const u = urlInput.value.trim();
    if (!u) return;
    if (!multiple) items.length = 0;
    if (items.length < max) items.push({ url: u });
    urlInput.value = ''; render(); onChange();
  };
  urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); addUrl.click(); } });

  const node = el('div', { class: 'field', 'data-field': name }, list, drop, file, el('div', { class: 'urlrow' }, urlInput, addUrl), status);
  render();
  return {
    node,
    get() {
      const urls = items.filter((i) => !i.busy && i.url).map((i) => i.url);
      return multiple ? urls : urls[0] ?? null;
    },
    set(v) {
      items.length = 0;
      for (const u of (multiple ? v || [] : v ? [v] : [])) items.push({ url: u });
      render();
    },
    busy: () => items.some((i) => i.busy),
  };
}

function scalarField(name, prop, { onChange }) {
  const t = baseType(prop);
  const p = inner(prop);
  const id = 'f-' + name;
  if (t === 'boolean') {
    const box = el('input', { type: 'checkbox', id, 'data-input': name });
    box.checked = !!prop.default;
    box.onchange = onChange;
    return { node: el('label', { class: 'check', for: id }, box, human(name)), get: () => box.checked, set: (v) => { box.checked = !!v; }, inline: true };
  }
  if (Array.isArray(p.enum)) {
    // Values keep their JSON type (integer `fps`, `duration: 6 | "auto"`).
    const opts = nullable(prop) ? [null, ...p.enum] : p.enum;
    const sel = el('select', { id, 'data-input': name }, opts.map((v) => el('option', { value: String(v) }, v === null ? '—' : String(v))));
    const set = (v) => { const i = opts.findIndex((o) => String(o) === String(v)); sel.selectedIndex = i < 0 ? 0 : i; };
    set(prop.default !== undefined ? prop.default : opts[0]);
    sel.onchange = onChange;
    return { node: sel, get: () => opts[sel.selectedIndex], set };
  }
  if (t === 'integer' || t === 'number') {
    const lo = p.minimum; const hi = p.maximum;
    if (!nullable(prop) && Number.isFinite(lo) && Number.isFinite(hi) && hi - lo <= 30) {
      const out = el('span', { class: 'mono' }, String(prop.default ?? lo));
      const r = el('input', { type: 'range', id, min: lo, max: hi, step: t === 'integer' ? 1 : 0.1, value: prop.default ?? lo, 'data-input': name });
      r.oninput = () => { out.textContent = r.value; onChange(); };
      return {
        node: el('div', {}, r), get: () => Number(r.value),
        set: (v) => { r.value = v; out.textContent = String(r.value); }, out,
        // A lower maximum for the current choice elsewhere in the form
        // (`x-fv-max-by-resolution`); the value is clamped to it.
        setMax: (m) => {
          r.max = String(m);
          r.dataset.max = String(m);
          if (Number(r.value) > m) r.value = String(m);
          out.textContent = r.value;
        },
      };
    }
    const inp = el('input', { id, inputmode: 'numeric', 'data-input': name, placeholder: nullable(prop) ? 'random' : '' });
    if (prop.default !== null && prop.default !== undefined) inp.value = prop.default;
    inp.oninput = onChange;
    return {
      node: inp,
      get: () => { const s = inp.value.trim(); return s === '' ? null : Number(s); },
      set: (v) => { inp.value = v ?? ''; },
    };
  }
  if (prop['x-fv-multiline']) {
    const ta = el('textarea', { id, 'data-input': name, maxlength: p.maxLength, rows: 6 });
    ta.oninput = onChange;
    return { node: ta, get: () => ta.value, set: (v) => { ta.value = v ?? ''; } };
  }
  const listId = id + '-list';
  const inp = el('input', { id, 'data-input': name, spellcheck: 'false', list: prop.examples ? listId : undefined });
  inp.value = prop.default ?? '';
  inp.oninput = onChange;
  const dl = prop.examples ? el('datalist', { id: listId }, prop.examples.map((v) => el('option', { value: v }))) : null;
  return { node: el('div', {}, inp, dl), get: () => (inp.value === '' ? null : inp.value), set: (v) => { inp.value = v ?? ''; } };
}

// Builds the form into `root`. Returns {values(), setValues(v), reset(), busy()}.
export function buildForm(schema, root, { upload, onChange: notify = () => {} }) {
  const props = schema.properties || {};
  // Limits that depend on the chosen resolution (the H3 1080P tier's
  // shorter duration cap, `x-fv-max-by-resolution`).
  const applyLimits = () => {
    const res = fields.resolution ? fields.resolution.get() : null;
    for (const f of Object.values(fields)) {
      const by = f.prop['x-fv-max-by-resolution'];
      if (!by || !f.setMax) continue;
      const base = inner(f.prop).maximum;
      const cap = res !== null && Object.prototype.hasOwnProperty.call(by, res) ? Math.min(by[res], base) : base;
      f.setMax(cap);
    }
  };
  const onChange = () => { applyLimits(); notify(); };
  const order = schema['x-fal-order-properties'] || Object.keys(props);
  const required = new Set(schema.required || []);
  const fields = {};
  const primary = el('div', { class: 'primary-fields' });
  const advancedBody = el('div', {});
  const advanced = el('details', { class: 'more', 'data-advanced': true }, el('summary', {}, 'Additional settings'), advancedBody);

  for (const name of order) {
    const prop = props[name];
    if (!prop) continue;
    const media = prop['x-fv-media'];
    const f = media
      ? mediaField(name, prop, { upload, onChange, multiple: baseType(prop) === 'array' })
      : scalarField(name, prop, { onChange });
    fields[name] = { ...f, prop };
    const label = f.inline ? null : el('label', { for: 'f-' + name },
      human(name), required.has(name) ? el('span', { class: 'req', title: 'required' }, '*') : null,
      f.out ? el('span', {}, ' · ', f.out, name === 'duration' ? ' s' : '') : null);
    const help = prop.description ? el('p', { class: 'hint' }, prop.description) : null;
    const block = el('div', { class: 'field-block', 'data-block': name }, label, f.node, media || prop['x-fv-multiline'] ? help : null);
    if (!media && !prop['x-fv-multiline'] && !f.inline) block.title = prop.description || '';
    const target = prop['x-fv-advanced'] ? advancedBody : primary;
    // Short scalar fields (selects, sliders) share a row.
    const short = !media && !prop['x-fv-multiline'] && !f.inline && !prop['x-fv-advanced'];
    if (short) {
      if (!primary.lastChild || !primary.lastChild.classList || !primary.lastChild.classList.contains('row')) primary.append(el('div', { class: 'row' }));
      primary.lastChild.append(block);
    } else {
      target.append(block);
    }
  }
  root.replaceChildren(primary, advancedBody.childNodes.length ? advanced : '');
  applyLimits();

  function values() {
    const out = {};
    for (const name of order) {
      const f = fields[name];
      if (!f) continue;
      const v = f.get();
      const d = f.prop.default;
      const isDefault = JSON.stringify(v) === JSON.stringify(d === undefined ? null : d) || (Array.isArray(v) && v.length === 0 && (d === undefined || (Array.isArray(d) && d.length === 0)));
      if (required.has(name) || (!isDefault && v !== null && v !== '')) out[name] = v;
    }
    return out;
  }
  function setValues(v) {
    // Lift the dependent limits first so a restored value is not clamped
    // by the previous resolution's cap.
    for (const f of Object.values(fields)) if (f.setMax) f.setMax(inner(f.prop).maximum);
    for (const [name, f] of Object.entries(fields)) {
      if (v && Object.prototype.hasOwnProperty.call(v, name)) f.set(v[name]);
      else f.set(f.prop.default ?? (baseType(f.prop) === 'array' ? [] : null));
    }
    onChange();
  }
  return {
    values,
    setValues,
    reset: () => setValues({}),
    busy: () => Object.values(fields).some((f) => f.busy && f.busy()),
    fields,
  };
}
