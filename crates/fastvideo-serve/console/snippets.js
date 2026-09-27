// API snippets (cURL, Python fal-client, JS @fal-ai/client) for one
// endpoint and input, pointed at this server.

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
