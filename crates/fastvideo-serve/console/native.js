// Native API (advanced): `POST /fv/v1/jobs` with the fields only the native
// API takes, on any served model, tier alias or alias from
// `GET /fv/v1/capabilities`, including tiers no fal endpoint serves (such
// as `ltx-draft`). The form follows NativeBody
// (crates/fastvideo-serve/src/native.rs) and the model's caps: its tasks,
// canvas tiers, fps, frame range, reference limits and the sampling knobs
// it honours (`knobs`: negative, seed, steps, guidance, reference_strength).
//
// `flow_shift` and `guidance_scale_2` are not native fields: OpenAI
// `/v1/videos` takes them (crates/fastvideo-openai-videos/src/videos.rs),
// so they appear with the OpenAI API chosen, on models whose knobs honour
// them. No API here takes an `audio_out` or `callback` field: audio follows
// the model (`caps.audio`), and only MiniMax `/v2/video_generation` has a
// `callback_url` (see the model page's API tab).

import {
  $, el, request, setMsg, topbar, loadAuthMode, needsKey, loadCapabilities, mountedProtocols, upload, draftPill, recipeText,
  base, copyText,
} from './common.js';
import { bearer } from './rtc.js';

topbar('native');

const TASKS = {
  t2v: 'Text to video', i2v: 'Image to video', keyframes: 'Keyframes (first / last frame)', ref2v: 'Reference to video',
  a2v: 'Audio to video', retake: 'Retake (regenerate a window)', extend: 'Extend a video',
};
const OPENAI_TASKS = new Set(['t2v', 'i2v']);
const RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4'];
const EDIT = new Set(['retake', 'extend']);

let caps = null;
let protocols = null;
let choices = []; // [{name, label, entry}]
let fields = {};
let job = null; // {id, api, timer}

const choice = () => choices.find((c) => c.name === $('model').value);
const api = () => $('api').value;

// ---- the field set -----------------------------------------------------------
// Each field: {name (native), label, kind, show(ctx), openai (the OpenAI name,
// or null when only native takes it), hint, required(ctx)}.
function spec(ctx) {
  const k = ctx.knobs;
  const c = ctx.caps;
  const t = ctx.task;
  const native = ctx.api === 'native';
  const frames = c.frames || {};
  const fps = (c.fps && c.fps.default) || 24;
  const secs = (n) => Math.round((n / fps) * 100) / 100;
  return [
    { name: 'prompt', label: 'Prompt', kind: 'textarea', show: true, openai: 'prompt',
      required: !(t === 'a2v' || EDIT.has(t)), hint: t === 'a2v' ? 'Optional with a first-frame image.' : EDIT.has(t) ? 'What happens in the new section (optional).' : '' },
    { name: 'negative_prompt', label: 'Negative prompt', kind: 'text', show: k.negative, openai: 'negative_prompt' },
    { name: 'video_url', label: 'Source video', kind: 'media', media: 'video', show: native && EDIT.has(t), required: true, hint: 'MP4 / MOV / MKV / WebM, 8 to 60 fps, at most 60 s.' },
    { name: 'image_url', label: t === 'a2v' ? 'First frame (optional)' : 'First-frame image', kind: 'media', media: 'image', show: ['i2v', 'keyframes', 'a2v'].includes(t), openai: 'input_reference', required: t === 'i2v' },
    { name: 'last_image_url', label: 'Last-frame image', kind: 'media', media: 'image', show: native && t === 'keyframes', required: true },
    { name: 'reference_urls', label: 'Reference images', kind: 'media', media: 'image', multiple: (c.refs && c.refs.images) || 1, show: native && t === 'ref2v', required: true,
      hint: 'Up to ' + ((c.refs && c.refs.images) || 1) + ' (this model\'s reference limit).' },
    { name: 'audio_url', label: t === 'retake' ? 'Audio for the window (optional)' : 'Driving audio', kind: 'media', media: 'audio', show: native && (t === 'a2v' || t === 'retake'), required: t === 'a2v',
      hint: t === 'a2v' ? 'Sets the length (2 to 20 s).' : 'With replace_video, the window gets this audio and the picture follows it.' },
    { name: 'start_s', label: 'Window start (s)', kind: 'number', step: 0.1, show: native && t === 'retake', required: true },
    { name: 'end_s', label: 'Window end (s)', kind: 'number', step: 0.1, show: native && t === 'retake', required: true, hint: '2 to 20 s after the start.' },
    { name: 'retake_mode', label: 'Retake mode', kind: 'select', options: ['', 'replace_audio_and_video', 'replace_video', 'replace_audio'], show: native && t === 'retake' },
    { name: 'extend_s', label: 'Extend by (s)', kind: 'number', step: 0.1, min: 2, max: 20, show: native && t === 'extend', required: true },
    { name: 'extend_at', label: 'Extend at', kind: 'select', options: ['', 'end', 'start'], show: native && t === 'extend' },
    { name: 'context_s', label: 'Context (s)', kind: 'number', step: 0.1, min: 1, max: 20, show: native && t === 'extend', hint: 'Seconds of the source the model continues from (default: as many as fit).' },
    { name: 'aspect_ratio', label: 'Aspect ratio', kind: 'select', options: ['', ...RATIOS], show: !EDIT.has(t), openai: 'aspect_ratio',
      hint: t === 't2v' ? 'With a short edge; empty: the model default canvas.' : 'Empty: follow the image.' },
    { name: 'short_edge', label: 'Short edge', kind: 'select', options: ['', ...((c.canvas && c.canvas.short_edges) || []).slice().sort((a, b) => a - b).map(String)], show: !EDIT.has(t), openai: 'short_edge', int: true },
    { name: 'size', label: 'Size (WxH)', kind: 'text', show: native, advanced: true,
      hint: EDIT.has(t) ? 'Optional: at most the source\'s size (default: the source).' : 'Instead of aspect ratio + short edge, e.g. 1280x704.' },
    { name: 'seconds', label: 'Seconds', kind: 'number', step: 0.5, min: frames.min ? secs(frames.min) : undefined, max: frames.max ? secs(frames.max) : undefined, show: !EDIT.has(t), openai: 'seconds',
      hint: frames.min ? secs(frames.min) + ' to ' + secs(frames.max) + ' s at ' + fps + ' fps (snapped up to the frame grid).' : '' },
    { name: 'num_frames', label: 'Frames', kind: 'number', step: 1, min: frames.min, max: frames.max, show: !EDIT.has(t), openai: 'num_frames', int: true, advanced: true,
      hint: frames.step ? 'Instead of seconds: ' + frames.min + ' to ' + frames.max + ', on the ' + frames.step + 'n+' + (frames.offset || 0) + ' grid.' : '' },
    { name: 'fps', label: 'FPS', kind: 'select', options: ['', ...((c.fps && c.fps.allowed) || []).map(String)], show: !EDIT.has(t), openai: 'fps', int: true, advanced: true },
    { name: 'seed', label: 'Seed', kind: 'number', step: 1, show: k.seed, openai: 'seed', int: true, advanced: true },
    { name: 'steps', label: 'Steps', kind: 'number', step: 1, min: 1, show: k.steps, openai: 'num_inference_steps', int: true, advanced: true,
      hint: ctx.entry && ctx.entry.recipe && ctx.entry.recipe.steps ? 'The recipe runs ' + ctx.entry.recipe.steps + '.' : '' },
    { name: 'guidance', label: 'Guidance', kind: 'number', step: 0.1, show: k.guidance, openai: 'guidance_scale', advanced: true },
    { name: 'flow_shift', label: 'Flow shift', kind: 'number', step: 0.1, show: !native && k.flow_shift, openai: 'flow_shift', advanced: true, hint: 'OpenAI /v1/videos only (not a native field).' },
    { name: 'guidance_scale_2', label: 'Guidance 2', kind: 'number', step: 0.1, show: !native && k.guidance_2, openai: 'guidance_scale_2', advanced: true, hint: 'OpenAI /v1/videos only (the second expert\'s guidance).' },
    { name: 'reference_strength', label: 'Reference strength', kind: 'number', step: 0.05, min: 0, max: 1, show: native && t === 'ref2v' && k.reference_strength, advanced: true },
    { name: 'reference_lora_strength', label: 'Reference LoRA strength', kind: 'number', step: 0.05, min: 0, max: 2, show: native && t === 'ref2v' && k.reference_strength, advanced: true },
  ].filter((f) => f.show && (native || f.openai));
}

// ---- media: a URL, or a file uploaded (fal storage) or inlined (data URI) ----

function fileUrl(file) {
  if (!protocols || protocols.fal !== false) return upload(file).catch(() => dataUri(file));
  return dataUri(file);
}
function dataUri(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(new Error('could not read ' + file.name));
    r.readAsDataURL(file);
  });
}

function mediaControl(f) {
  const max = f.multiple || 1;
  const box = el('div', { class: 'field', 'data-field': f.name });
  const rows = el('div', {});
  const status = el('span', { class: 'hint' });
  const add = (v = '') => {
    const input = el('input', { type: 'url', value: v, spellcheck: 'false', placeholder: 'https:// or data: URL', 'data-input': f.name });
    input.oninput = changed;
    const file = el('input', { type: 'file', accept: f.media + '/*', hidden: true });
    const pick = el('button', { type: 'button', class: 'small', onclick: () => file.click() }, 'Choose file');
    file.onchange = async () => {
      const x = file.files[0]; file.value = '';
      if (!x) return;
      status.textContent = 'uploading ' + x.name + '…';
      try { input.value = await fileUrl(x); status.textContent = ''; changed(); } catch (e) { status.textContent = 'upload failed: ' + e.message; }
    };
    rows.append(el('div', { class: 'urlrow' }, input, pick, file));
    more.hidden = rows.children.length >= max;
  };
  const more = el('button', { type: 'button', class: 'small', onclick: () => add() }, 'Add another');
  add();
  box.append(rows, max > 1 ? more : '', status);
  return {
    node: box,
    get: () => {
      const vs = [...rows.querySelectorAll('input[type="url"]')].map((i) => i.value.trim()).filter(Boolean);
      return max > 1 ? (vs.length ? vs : null) : vs[0] || null;
    },
  };
}

function control(f) {
  const id = 'n-' + f.name;
  if (f.kind === 'media') return mediaControl(f);
  if (f.kind === 'textarea') {
    const ta = el('textarea', { id, rows: 4, 'data-input': f.name });
    ta.oninput = changed;
    return { node: ta, get: () => ta.value.trim() || null };
  }
  if (f.kind === 'select') {
    const sel = el('select', { id, 'data-input': f.name }, f.options.map((o) => el('option', { value: o }, o === '' ? '—' : o)));
    sel.onchange = changed;
    return { node: sel, get: () => (sel.value === '' ? null : f.int ? Number(sel.value) : sel.value) };
  }
  const inp = el('input', { id, 'data-input': f.name, type: f.kind === 'number' ? 'number' : 'text', step: f.step, min: f.min, max: f.max, spellcheck: 'false' });
  inp.oninput = changed;
  return { node: inp, get: () => { const s = inp.value.trim(); return s === '' ? null : f.kind === 'number' ? Number(s) : s; } };
}

function ctx() {
  const c = choice();
  const entry = c && c.entry;
  return { api: api(), task: $('task').value, entry, caps: (entry && entry.caps) || {}, knobs: (entry && entry.caps && entry.caps.knobs) || {} };
}

function buildForm() {
  const x = ctx();
  const list = spec(x);
  fields = {};
  const primary = el('div', {});
  const advancedBody = el('div', { class: 'row' });
  for (const f of list) {
    const ctl = control(f);
    fields[f.name] = { ...f, ...ctl };
    const block = el('div', { class: 'field-block', 'data-block': f.name },
      el('label', { for: 'n-' + f.name }, f.label, f.required ? el('span', { class: 'req', title: 'required' }, '*') : null),
      ctl.node, f.hint ? el('p', { class: 'hint' }, f.hint) : null);
    (f.advanced ? advancedBody : primary).append(block);
  }
  $('form').replaceChildren(primary, advancedBody.childNodes.length ? el('details', { class: 'more', open: true, 'data-advanced': true }, el('summary', {}, 'Sampling and timing'), advancedBody) : '');
  changed();
}

// The request body for the chosen API (only the fields set).
function body() {
  const x = ctx();
  const c = choice();
  const b = { model: c ? c.name : '' };
  const native = x.api === 'native';
  if (!native && x.task === 't2v') delete b.task;
  for (const f of Object.values(fields)) {
    const v = f.get();
    if (v === null || v === undefined || v === '') continue;
    if (native) b[f.name] = v;
    else b[f.openai] = f.name === 'seconds' ? String(v) : v;
  }
  if (native && b.prompt === undefined) b.prompt = '';
  return b;
}

// Null when the request can go, else why not.
function validate(b) {
  for (const f of Object.values(fields)) {
    const v = f.get();
    if (f.required && (v === null || v === undefined || v === '' || (Array.isArray(v) && !v.length))) return f.label + ' is required.';
  }
  const has = (k) => b[k] !== undefined;
  if (has('seconds') && has('num_frames')) return 'Give seconds or frames, not both.';
  if (has('aspect_ratio') && !has('short_edge')) return 'An aspect ratio needs a short edge.';
  if (has('size') && (has('aspect_ratio') || has('short_edge'))) return 'Give a size, or an aspect ratio with a short edge, not both.';
  if (has('size') && !/^\d+\s*[xX*]\s*\d+$/.test(String(b.size))) return 'Size is WxH, e.g. 1280x704.';
  if (ctx().task === 't2v' && has('short_edge') && !has('aspect_ratio')) return 'Text to video needs an aspect ratio with the short edge.';
  if (has('end_s') && has('start_s') && !(b.end_s > b.start_s)) return 'The window end must be after its start.';
  return null;
}

function curl(b, path) {
  return [
    'curl -s -X POST "' + base() + path + '" \\',
    '  -H "Authorization: Bearer $FV_KEY" -H "Content-Type: application/json" \\',
    "  -d '" + JSON.stringify(b, (k, v) => (typeof v === 'string' && v.startsWith('data:') && v.length > 120 ? v.slice(0, 60) + '…' : v), 2).replace(/'/g, "'\\''") + "'",
  ].join('\n');
}

function changed() {
  const b = body();
  const path = api() === 'native' ? '/fv/v1/jobs' : '/v1/videos';
  $('endpoint').textContent = 'POST ' + path;
  $('body-json').textContent = JSON.stringify(b, null, 2);
  $('curl').textContent = curl(b, path);
}

// ---- model, task, API --------------------------------------------------------

function showModel() {
  const c = choice();
  const entry = c && c.entry;
  const cp = (entry && entry.caps) || {};
  const tasks = (cp.tasks || []).filter((t) => api() === 'native' || OPENAI_TASKS.has(t));
  const prev = $('task').value;
  $('task').replaceChildren(...tasks.map((t) => el('option', { value: t }, TASKS[t] || t)));
  if (tasks.includes(prev)) $('task').value = prev;
  const k = cp.knobs || {};
  const facts = [];
  if (c && c.name !== cp.id) facts.push(c.name + ' → ' + cp.id);
  if (cp.tier) facts.push('tier ' + cp.tier);
  if (cp.recipe) facts.push('recipe ' + cp.recipe + (recipeText(entry.recipe) ? ' (' + recipeText(entry.recipe) + ')' : ''));
  facts.push('honours: ' + (Object.entries(k).filter(([, v]) => v).map(([n]) => n).join(', ') || 'no sampling knobs'));
  $('model-facts').replaceChildren(facts.join(' · '), cp.tier === 'draft' ? [' ', draftPill()] : '');
  buildForm();
}

async function load() {
  await loadAuthMode();
  $('key-banner').hidden = !needsKey();
  caps = await loadCapabilities();
  if (!caps) { $('page-error').textContent = 'Could not read GET /fv/v1/capabilities (is the API key set and the native API mounted?).'; $('page-error').hidden = false; return; }
  protocols = mountedProtocols(caps);
  const models = caps.models || [];
  const byId = (id) => models.find((m) => m.caps && m.caps.id === id);
  // Served models, then the tier aliases (ltx-draft, h3-turbo, …) and other aliases.
  choices = models.map((m) => ({ name: m.caps.id, label: m.caps.id + (m.caps.tier ? ' (' + m.caps.tier + ')' : ''), entry: m }));
  for (const t of caps.tiers || []) {
    if (t.alias && byId(t.model) && !choices.some((c) => c.name === t.alias)) choices.push({ name: t.alias, label: t.alias + ' → ' + t.model + ' (tier ' + t.tier + ')', entry: byId(t.model) });
  }
  for (const [a, id] of Object.entries(caps.aliases || {})) {
    if (byId(id) && !choices.some((c) => c.name === a)) choices.push({ name: a, label: a + ' → ' + id, entry: byId(id) });
  }
  // A causal or duplex model takes no batch jobs.
  choices = choices.filter((c) => !(c.entry.caps.stream && (c.entry.caps.stream.causal || c.entry.caps.stream.duplex)));
  $('model').replaceChildren(...choices.map((c) => el('option', { value: c.name }, c.label)));
  const want = new URLSearchParams(location.search).get('model');
  if (want && choices.some((c) => c.name === want)) $('model').value = want;
  const apis = [['native', 'Native /fv/v1/jobs']];
  if (!protocols || protocols.openai_videos) apis.push(['openai', 'OpenAI /v1/videos (flow_shift, guidance_scale_2)']);
  $('api').replaceChildren(...apis.map(([v, l]) => el('option', { value: v }, l)));
  showModel();
}

// ---- submit and follow ---------------------------------------------------------

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'completed']);

function status(s) {
  const p = $('job-status');
  p.hidden = !s;
  p.textContent = s || '';
  p.className = 'pill s-' + ({ succeeded: 'COMPLETED', completed: 'COMPLETED', failed: 'FAILED', queued: 'IN_QUEUE', in_progress: 'IN_PROGRESS', running: 'IN_PROGRESS' }[s] || '');
  $('cancel').hidden = !(s && !TERMINAL.has(s));
}

function showVideo(url) {
  const v = $('video');
  if (url) { v.src = url; v.hidden = false; $('video-empty').hidden = true; } else { v.removeAttribute('src'); v.hidden = true; $('video-empty').hidden = false; }
}

function renderJob(j) {
  $('job-json').textContent = JSON.stringify(j, null, 2);
  const m = j.metrics || {};
  const rows = [
    ['id', j.id], ['status', j.status], ['model', j.model + (j.resolved_model && j.resolved_model !== j.model ? ' → ' + j.resolved_model : '')],
    ['tier', j.tier ? el('span', {}, j.tier, j.tier === 'draft' ? [' ', draftPill()] : '') : null], ['recipe', j.recipe],
    ['task', j.task], ['canvas', j.width && j.height ? j.width + '×' + j.height : null],
    ['frames', j.num_frames ? j.num_frames + ' at ' + j.fps + ' fps' : null], ['seed', j.seed != null ? String(j.seed) : null],
    ['queue', m.queue_s != null ? Number(m.queue_s).toFixed(2) + ' s' : null], ['run', m.run_s != null ? Number(m.run_s).toFixed(2) + ' s' : null],
    ['inference', m.inference_s != null ? Number(m.inference_s).toFixed(2) + ' s' : null],
    ['notes', (j.notes || []).join('; ')], ['error', j.error ? (j.error.message || JSON.stringify(j.error)) : null],
  ];
  $('facts').replaceChildren(...rows.filter(([, v]) => v !== null && v !== undefined && v !== '').flatMap(([k, v]) => [el('dt', {}, k), el('dd', { 'data-fact': k }, v)]));
  $('result-quality').hidden = j.tier !== 'draft';
}

async function poll() {
  const cur = job;
  if (!cur) return;
  try {
    const j = await request('GET', cur.api === 'native' ? '/fv/v1/jobs/' + cur.id : '/v1/videos/' + cur.id, { auth: 'bearer' });
    if (job !== cur) return;
    status(j.status);
    renderJob(j);
    if (TERMINAL.has(j.status)) {
      $('run').disabled = false;
      if (j.status === 'succeeded' && j.output && j.output.url) showVideo(j.output.url);
      else if (j.status === 'completed' && cur.api === 'openai') {
        // The content route needs the key: fetch it and play the blob.
        const r = await fetch(base() + '/v1/videos/' + cur.id + '/content', { headers: bearer() });
        if (r.ok) showVideo(URL.createObjectURL(await r.blob()));
      } else if (j.status === 'failed') setMsg('result-msg', (j.error && (j.error.message || j.error.code)) || 'failed', 'bad');
      document.body.dataset.job = j.status;
      return;
    }
    $('video-empty').textContent = (j.status === 'queued' ? 'In queue' : 'Generating') + (j.progress != null ? ' (' + Math.round(j.progress * 100) + ' %)' : '') + '…';
  } catch (e) { setMsg('result-msg', e.message + ' (retrying)', 'bad'); }
  cur.timer = setTimeout(poll, 700);
}

async function run() {
  await loadAuthMode();
  if (needsKey()) { setMsg('run-msg', 'Set an API key first (Models page or API keys page).', 'bad'); return; }
  const b = body();
  const bad = validate(b);
  if (bad) { setMsg('run-msg', bad, 'bad'); return; }
  const a = api();
  $('run').disabled = true;
  setMsg('run-msg', ''); setMsg('result-msg', '');
  showVideo(null); $('video-empty').textContent = 'Submitting…'; $('facts').replaceChildren(); $('result-quality').hidden = true;
  delete document.body.dataset.job;
  if (job && job.timer) clearTimeout(job.timer);
  try {
    const j = await request('POST', a === 'native' ? '/fv/v1/jobs' : '/v1/videos', { auth: 'bearer', body: b });
    job = { id: j.id, api: a };
    setMsg('run-msg', 'Submitted ' + j.id + '.', 'ok');
    status(j.status);
    renderJob(j);
    poll();
  } catch (e) {
    $('run').disabled = false;
    status('');
    $('video-empty').textContent = 'Not submitted.';
    setMsg('run-msg', e.message, 'bad');
  }
}

$('run').onclick = run;
$('reset').onclick = buildForm;
$('form').addEventListener('submit', (e) => e.preventDefault());
$('cancel').onclick = async () => {
  if (!job) return;
  try {
    await request('DELETE', job.api === 'native' ? '/fv/v1/jobs/' + job.id : '/v1/videos/' + job.id, { auth: 'bearer' });
    setMsg('result-msg', 'Cancellation requested.');
  } catch (e) { setMsg('result-msg', e.message, 'bad'); }
};
$('model').onchange = showModel;
$('task').onchange = buildForm;
$('api').onchange = showModel;
$('curl').addEventListener('dblclick', () => copyText($('curl').textContent));

load();
