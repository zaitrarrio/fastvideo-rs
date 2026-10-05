// API snippets for one endpoint and input, pointed at this server: the fal
// queue (cURL, Python fal-client, JS @fal-ai/client), and the same request
// on every other API the server mounts that can run the endpoint's model
// (native /fv/v1/jobs, OpenAI /v1/videos, MiniMax /v2/video_generation, the
// LTX API /v2/{endpoint}, the Reactor runtime). The bodies use only fields
// each API's parser accepts (crates/fastvideo-*/src); fal inputs with no
// counterpart are listed in a comment instead of being invented.

function toPython(v, ind = 0) {
  const pad = '    '.repeat(ind + 1); const end = '    '.repeat(ind);
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return v.length ? '[\n' + v.map((x) => pad + toPython(x, ind + 1)).join(',\n') + ',\n' + end + ']' : '[]';
  const ks = Object.keys(v);
  return ks.length ? '{\n' + ks.map((k) => pad + JSON.stringify(k) + ': ' + toPython(v[k], ind + 1)).join(',\n') + ',\n' + end + '}' : '{}';
}

const shq = (s) => "'" + s.replace(/'/g, "'\\''") + "'";

// Long data URIs make snippets unreadable; abbreviate them.
function tidy(input) {
  return JSON.parse(JSON.stringify(input, (k, v) => (typeof v === 'string' && v.startsWith('data:') && v.length > 120 ? v.slice(0, 60) + '…' : v)));
}

export function snippets({ base, app, sub, input }) {
  const endpoint = app + '/' + sub;
  const host = base.replace(/^https?:\/\//, '');
  const body = tidy(input);
  const json = JSON.stringify(body, null, 2);
  const curl = [
    '# A key minted on /console/admin (or one listed in FV_API_KEYS).',
    'export FAL_KEY="fv_…"',
    'BASE=' + shq(base),
    '',
    '# 1. Submit to the queue',
    'curl -s -X POST "$BASE/' + endpoint + '" \\',
    '  -H "Authorization: Key $FAL_KEY" \\',
    '  -H "Content-Type: application/json" \\',
    '  -d ' + shq(json),
    '# → {"request_id": "…", "status_url": "…", "response_url": "…", "cancel_url": "…"}',
    '',
    '# 2. Poll the status (with logs)',
    'curl -s "$BASE/' + app + '/requests/$REQUEST_ID/status?logs=1" -H "Authorization: Key $FAL_KEY"',
    '',
    '# 3. Fetch the result',
    'curl -s "$BASE/' + app + '/requests/$REQUEST_ID" -H "Authorization: Key $FAL_KEY"',
    '',
    '# Or run it synchronously on one connection:',
    'curl -s -X POST "$BASE/run/' + endpoint + '" -H "Authorization: Key $FAL_KEY" \\',
    '  -H "Content-Type: application/json" -d ' + shq(JSON.stringify(body)),
  ].join('\n');

  const python = [
    '# pip install fal-client',
    'import os',
    '',
    '# fal_client reads these at import time and only speaks https:',
    '# put fv-serve behind TLS (or a TLS proxy) for Python.',
    'os.environ["FAL_RUN_HOST"] = ' + JSON.stringify(host + '/run'),
    'os.environ["FAL_QUEUE_RUN_HOST"] = ' + JSON.stringify(host),
    'os.environ.setdefault("FAL_KEY", "fv_…")  # minted on /console/admin',
    '',
    'import fal_client',
    '',
    '',
    'def on_queue_update(update):',
    '    if isinstance(update, fal_client.InProgress):',
    '        for log in update.logs:',
    '            print(log["message"])',
    '',
    '',
    'result = fal_client.subscribe(',
    '    ' + JSON.stringify(endpoint) + ',',
    '    arguments=' + toPython(body, 1) + ',',
    '    with_logs=True,',
    '    on_queue_update=on_queue_update,',
    ')',
    'print(result["video"]["url"])',
    '',
    '# Python uploads go to fal\'s own storage and cannot be redirected: pass',
    '# https URLs or data URIs for image/audio inputs.',
  ].join('\n');

  const js = [
    '// npm install @fal-ai/client',
    'import { fal } from "@fal-ai/client";',
    '',
    'const BASE = ' + JSON.stringify(base) + ';',
    '',
    'fal.config({',
    '  credentials: process.env.FAL_KEY, // minted on /console/admin',
    '  // Send fal\'s hosts to this server (queue, sync run, storage uploads).',
    '  requestMiddleware: async (req) => ({',
    '    ...req,',
    '    url: req.url',
    '      .replace(/^https:\\/\\/queue\\.fal\\.run\\//, BASE + "/")',
    '      .replace(/^https:\\/\\/fal\\.run\\//, BASE + "/run/")',
    '      .replace(/^https:\\/\\/rest\\.fal\\.ai\\//, BASE + "/"),',
    '  }),',
    '});',
    '',
    'const result = await fal.subscribe(' + JSON.stringify(endpoint) + ', {',
    '  input: ' + JSON.stringify(body, null, 2).replace(/\n/g, '\n  ') + ',',
    '  logs: true,',
    '  onQueueUpdate: (update) => {',
    '    if (update.status === "IN_PROGRESS") update.logs.map((l) => l.message).forEach(console.log);',
    '  },',
    '});',
    'console.log(result.data.video.url, result.requestId);',
    '',
    '// Browser Files/Blobs in `input` are uploaded through',
    '// POST /storage/upload/initiate and replaced by their URL.',
  ].join('\n');

  return { curl, python, js };
}

// ---- the other protocols -------------------------------------------------------

const TASK = {
  'text-to-video': 't2v', 'image-to-video': 'i2v', 'reference-to-video': 'ref2v', 'audio-to-video': 'a2v', 'retake-video': 'retake', 'extend-video': 'extend',
};
// The task of a fal endpoint sub-path (`v2.2-5b/text-to-video/fast-wan` -> t2v).
export function taskOf(sub, input = {}) {
  for (const [k, v] of Object.entries(TASK)) if (sub.includes(k)) return v === 't2v' && input.image_url ? 'i2v' : v;
  if (sub.includes('ingredient')) return 'ref2v';
  return 't2v';
}

const num = (v) => (typeof v === 'number' ? v : typeof v === 'string' && /^\d+(\.\d+)?$/.test(v) ? Number(v) : null);

// The native `/fv/v1/jobs` body (crates/fastvideo-serve/src/native.rs NativeBody).
export function nativeBody(model, input, sub) {
  const b = { model, prompt: input.prompt || '' };
  const left = [];
  const map = {
    negative_prompt: 'negative_prompt', seed: 'seed', image_url: 'image_url', end_image_url: 'last_image_url',
    audio_url: 'audio_url', video_url: 'video_url', num_inference_steps: 'steps', guidance_scale: 'guidance',
  };
  for (const [k, v] of Object.entries(input)) {
    if (k === 'prompt' || v === null || v === undefined || v === '') continue;
    if (map[k]) b[map[k]] = v;
    else if (k === 'duration' && num(v) !== null) b.seconds = num(v);
    else if (k === 'reference_image_urls' && Array.isArray(v) && v.length) b.reference_urls = v;
    else if (['sync_mode', 'enable_safety_checker', 'resolution', 'aspect_ratio'].includes(k)) continue;
    else left.push(k);
  }
  if (taskOf(sub, input) === 't2v' && input.aspect_ratio && /^\d+:\d+$/.test(input.aspect_ratio) && typeof input.resolution === 'string') {
    const se = num(input.resolution.replace(/p$/i, ''));
    if (se) { b.aspect_ratio = input.aspect_ratio; b.short_edge = se; } else left.push('resolution');
  } else {
    if (input.resolution) left.push('resolution');
    if (input.aspect_ratio) left.push('aspect_ratio');
  }
  return { body: b, left };
}

// The OpenAI `/v1/videos` body (crates/fastvideo-openai-videos/src/videos.rs).
function openaiBody(model, input) {
  const b = { model, prompt: input.prompt || '' };
  const left = [];
  for (const [k, v] of Object.entries(input)) {
    if (k === 'prompt' || v === null || v === undefined || v === '') continue;
    if (['seed', 'negative_prompt', 'num_inference_steps', 'guidance_scale', 'flow_shift', 'video_url'].includes(k)) b[k] = v;
    else if (k === 'duration' && num(v) !== null) b.seconds = String(num(v));
    else if (k === 'image_url') b.input_reference = v;
    else if (k === 'sync_mode' || k === 'enable_safety_checker') continue;
    else left.push(k);
  }
  return { body: b, left };
}

const MINIMAX = { max: 'MiniMax-H3-Max', turbo: 'MiniMax-H3-Turbo', draft: 'MiniMax-H3-Draft' };
// The MiniMax `/v2/video_generation` body (crates/fastvideo-minimax/src/create.rs).
function minimaxBody(tier, input) {
  const content = [{ type: 'text', text: input.prompt || '' }];
  if (input.image_url) content.push({ type: 'image_url', image_url: { url: input.image_url }, role: 'first_frame' });
  if (input.end_image_url) content.push({ type: 'image_url', image_url: { url: input.end_image_url }, role: 'last_frame' });
  for (const u of input.reference_image_urls || []) content.push({ type: 'image_url', image_url: { url: u }, role: 'reference_image' });
  for (const u of input.reference_video_urls || []) content.push({ type: 'video_url', video_url: { url: u } });
  for (const u of input.reference_audio_urls || []) content.push({ type: 'audio_url', audio_url: { url: u } });
  const b = { model: MINIMAX[tier] || 'MiniMax-H3', content };
  if (input.resolution) b.resolution = String(input.resolution).toUpperCase();
  if (num(input.duration) !== null) b.duration = num(input.duration);
  b.ratio = input.image_url || input.end_image_url ? 'adaptive' : input.aspect_ratio && input.aspect_ratio !== 'auto' ? input.aspect_ratio : '16:9';
  const left = Object.keys(input).filter((k) => input[k] !== null && input[k] !== '' && !['prompt', 'image_url', 'end_image_url', 'reference_image_urls', 'reference_video_urls', 'reference_audio_urls', 'resolution', 'duration', 'aspect_ratio', 'sync_mode', 'enable_safety_checker'].includes(k));
  return { body: b, left };
}

const LTX_MODEL = { max: 'ltx-2-5-pro', turbo: 'ltx-2-5-fast' };
// The LTX API body (crates/fastvideo-ltxapi/src/request.rs).
function ltxBody(tier, input, task) {
  const b = { prompt: input.prompt || '', model: LTX_MODEL[tier] || 'ltx-2-5-fast' };
  const d = num(input.duration);
  if (task !== 'a2v') { b.duration = d && [6, 8, 10].includes(d) ? d : 6; b.resolution = '1920x1080'; }
  if (input.image_url) b.image_uri = input.image_url;
  if (input.audio_url) b.audio_uri = input.audio_url;
  const left = Object.keys(input).filter((k) => input[k] !== null && input[k] !== '' && !['prompt', 'image_url', 'audio_url', 'duration', 'resolution', 'sync_mode', 'enable_safety_checker'].includes(k));
  return { body: b, left };
}

// cURL / Python requests / JS fetch for a submit-then-poll JSON API.
function httpSnippets({ base, title, auth, submit, body, poll, idField, result, left, note }) {
  const json = JSON.stringify(tidy(body), null, 2);
  const hdr = auth === 'key' ? 'Key $FV_KEY' : 'Bearer $FV_KEY';
  const pyHdr = auth === 'key' ? '"Key " + KEY' : '"Bearer " + KEY';
  const leftNote = left && left.length ? ['', 'Not sent (no counterpart on this API): ' + left.join(', ') + '.'] : [];
  const curl = [
    '# ' + title,
    'export FV_KEY="fv_…"   # minted on /console/admin',
    'BASE=' + shq(base),
    '',
    '# 1. Submit',
    'curl -s -X POST "$BASE' + submit + '" \\',
    '  -H "Authorization: ' + hdr + '" \\',
    '  -H "Content-Type: application/json" \\',
    '  -d ' + shq(json),
    '# → {"' + idField + '": "…", …}',
    '',
    '# 2. Poll until it is done',
    'curl -s "$BASE' + poll('$ID') + '" -H "Authorization: ' + hdr + '"',
    ...(result ? ['', '# ' + result] : []),
    ...leftNote.map((l) => (l ? '# ' + l : l)),
  ].join('\n');
  const python = [
    '# pip install requests',
    'import time, requests',
    '',
    'BASE = ' + JSON.stringify(base),
    'KEY = "fv_…"  # minted on /console/admin',
    'H = {"Authorization": ' + pyHdr + '}',
    '',
    'job = requests.post(BASE + ' + JSON.stringify(submit) + ', headers=H, json=' + toPython(tidy(body)) + ').json()',
    'job_id = job[' + JSON.stringify(idField) + ']',
    'while True:',
    '    st = requests.get(BASE + ' + JSON.stringify(poll('{}')).replace('{}', '" + job_id + "').replace(/ \+ ""$/, '') + ', headers=H).json()',
    '    print(st)',
    '    if str(st.get("status", "")).lower() in ("completed", "succeeded", "success", "failed", "fail", "cancelled"):',
    '        break',
    '    time.sleep(2)',
    ...leftNote.map((l) => (l ? '# ' + l : l)),
  ].join('\n');
  const js = [
    'const BASE = ' + JSON.stringify(base) + ';',
    'const H = { Authorization: ' + (auth === 'key' ? '"Key "' : '"Bearer "') + ' + process.env.FV_KEY, "Content-Type": "application/json" };',
    '',
    'const job = await (await fetch(BASE + ' + JSON.stringify(submit) + ', {',
    '  method: "POST", headers: H,',
    '  body: JSON.stringify(' + json.replace(/\n/g, '\n  ') + '),',
    '})).json();',
    'const id = job[' + JSON.stringify(idField) + '];',
    'for (;;) {',
    '  const st = await (await fetch(BASE + ' + JSON.stringify(poll('${id}')).replace(/^"/, '`').replace(/"$/, '`') + ', { headers: H })).json();',
    '  console.log(st);',
    '  if (/^(completed|succeeded|success|failed|fail|cancelled)$/i.test(String(st.status))) break;',
    '  await new Promise((r) => setTimeout(r, 2000));',
    '}',
    ...leftNote.map((l) => (l ? '// ' + l : l)),
  ].join('\n');
  return { curl, python, js, note };
}

// The APIs (beyond fal) that can run this endpoint, among those mounted
// (`mounted` null: the server does not say, so offer the generic ones).
export function snippetProtocols(mounted, ctx) {
  const on = (k) => (mounted ? mounted[k] === true : k === 'native' || k === 'openai_videos');
  const caps = ctx.entry && ctx.entry.caps;
  const family = caps && caps.family;
  const tier = (caps && caps.tier) || (ctx.tierHit && ctx.tierHit.tier) || null;
  const task = taskOf(ctx.sub || '', {});
  const list = [{ id: 'fal', label: 'fal', title: 'fal queue (this endpoint)' }];
  if (on('native')) list.push({ id: 'native', label: 'Native', title: 'POST /fv/v1/jobs' });
  if (on('openai_videos') && ['t2v', 'i2v'].includes(task)) list.push({ id: 'openai_videos', label: 'OpenAI', title: 'POST /v1/videos' });
  if (on('minimax') && /h3/i.test(String(family || ctx.model || '')) && ['t2v', 'i2v', 'ref2v'].includes(task) && tier) list.push({ id: 'minimax', label: 'MiniMax', title: 'POST /v2/video_generation' });
  if (on('ltx') && /ltx/i.test(String(family || ctx.model || '')) && LTX_MODEL[tier] && ['t2v', 'i2v', 'a2v'].includes(task)) list.push({ id: 'ltx', label: 'LTX API', title: 'POST /v2/{endpoint}' });
  const streamsThis = caps && ctx.reactorModel && (caps.id === ctx.reactorModel || (caps.served_names || []).includes(ctx.reactorModel));
  if (on('reactor') && caps && caps.stream && streamsThis) list.push({ id: 'reactor', label: 'Reactor', title: 'Reactor runtime (it streams this model)' });
  return list;
}

// Snippets for `proto` with ctx {base, model, entry, tierHit, sub, input}.
export function protocolSnippets(proto, ctx) {
  const { base, input } = ctx;
  const caps = ctx.entry && ctx.entry.caps;
  const tier = (caps && caps.tier) || (ctx.tierHit && ctx.tierHit.tier) || null;
  const task = taskOf(ctx.sub || '', input);
  const model = ctx.model || (caps && caps.id) || '';
  if (proto === 'native') {
    const { body, left } = nativeBody(model, input, ctx.sub || '');
    return httpSnippets({
      base, title: 'Native API: POST /fv/v1/jobs (Authorization: Bearer)', auth: 'bearer', submit: '/fv/v1/jobs', body,
      poll: (id) => '/fv/v1/jobs/' + id, idField: 'id', left,
      result: 'Done: GET $BASE/fv/v1/jobs/$ID/content redirects to the MP4 (or read the job\'s output URL).',
      note: 'The same request on the native API (Bearer key). The Native API page adds the fields only it takes.',
    });
  }
  if (proto === 'openai_videos') {
    const { body, left } = openaiBody(model, input);
    return httpSnippets({
      base, title: 'OpenAI Videos API: POST /v1/videos (Authorization: Bearer)', auth: 'bearer', submit: '/v1/videos', body,
      poll: (id) => '/v1/videos/' + id, idField: 'id', left,
      result: 'Done: GET $BASE/v1/videos/$ID/content is the MP4 (openai.videos.download_content).',
      note: 'The openai SDK works too: OpenAI(base_url=BASE + "/v1", api_key=KEY).videos.create(...).',
    });
  }
  if (proto === 'minimax') {
    const { body, left } = minimaxBody(tier, input);
    return httpSnippets({
      base, title: 'MiniMax API: POST /v2/video_generation (Authorization: Bearer)', auth: 'bearer', submit: '/v2/video_generation', body,
      poll: (id) => '/v2/query/video_generation/' + id, idField: 'task_id', left,
      result: 'The task\'s file URL is in the query reply once status is "succeeded". callback_url (optional) receives status posts.',
      note: 'MiniMax names the tier in `model` (' + (MINIMAX[tier] || 'MiniMax-H3') + ').',
    });
  }
  if (proto === 'ltx') {
    const ep = task === 'a2v' ? 'audio-to-video' : input.image_url ? 'image-to-video' : 'text-to-video';
    const { body, left } = ltxBody(tier, input, task);
    return httpSnippets({
      base, title: 'LTX API: POST /v2/' + ep + ' (Authorization: Bearer)', auth: 'bearer', submit: '/v2/' + ep, body,
      poll: (id) => '/v2/' + ep + '/' + id, idField: 'id', left,
      result: 'Completed: the status reply\'s result.video_url (no key needed). /v1/' + ep + ' answers synchronously with the MP4.',
      note: 'LTX names the tier in `model` (' + body.model + '); durations and resolutions follow the LTX support matrix.',
    });
  }
  if (proto === 'reactor') {
    const curl = [
      '# Reactor runtime: one live WebRTC session of the model the server\'s [reactor] model names.',
      'export FV_KEY="fv_…"',
      'BASE=' + shq(base),
      'curl -s "$BASE/schema"                       # the model and its command set',
      'curl -s -X POST "$BASE/start_session" -H "Authorization: Bearer $FV_KEY" \\',
      '  -H "Content-Type: application/json" -d \'{}\'',
      '# then connect with the Reactor SDK (below) or the console\'s Live stream page.',
      'curl -s -X POST "$BASE/stop_session" -H "Authorization: Bearer $FV_KEY" -d \'{}\'',
    ].join('\n');
    const python = [
      '# pip install reactor-sdk',
      'import asyncio',
      'from reactor_sdk import Reactor',
      '',
      'async def main():',
      '    r = Reactor("fastvideo", local=True, api_url=' + JSON.stringify(base) + ')',
      '    await r.connect()',
      '    print(await r.send_command("get_state", {}))',
      '    # causal models: set_prompt / set_paused / set_seed / reset;',
      '    # clip models: enqueue / play / set_autoplay (see GET /schema)',
      '    await r.send_command("set_prompt", {"prompt": ' + JSON.stringify(input.prompt || 'a forest road') + '})',
      '    await asyncio.sleep(10)',
      '    await r.disconnect()',
      '',
      'asyncio.run(main())',
    ].join('\n');
    return { curl, python, js: '// The browser flow is the console\'s Live stream page (live.js / stream.js):\n// POST /start_session, then the WebRTC connection under /sessions/…/transport/webrtc.', note: 'The Reactor runtime streams one model (the server\'s [reactor] model).' };
  }
  return snippets({ base, app: ctx.app && ctx.app.id, sub: ctx.sub, input });
}
