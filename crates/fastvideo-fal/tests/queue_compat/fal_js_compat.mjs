// Drive the real `@fal-ai/client` against a local fv fal server.
//
// Run by `tests/queue_compat.rs` (design §7.5). `@fal-ai/client` hard-codes
// fal's hosts, so it is pointed at us both supported ways (fal §12.1):
//
//   - `requestMiddleware` rewriting https://queue.fal.run/ -> <base>/ and
//     https://fal.run/ -> <base>/run/ (and rest.fal.ai -> <base>);
//   - `proxyUrl: {url: "<base>/fal/proxy", when: "always"}`, where the real
//     target travels in `x-fal-target-url`.
//
// Each mode runs queue submit/status/result/cancel, subscribe (polling and
// SSE streaming), run (sync), and a Blob input uploaded through
// `storage/upload/initiate` + PUT by `transformInput`.
//
// usage: node fal_js_compat.mjs <node_modules parent dir> <base url> <key> <png path>

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import path from "node:path";

const [, , modulesDir, base, key, pngPath] = process.argv;
const require = createRequire(path.join(modulesDir, "package.json"));
const { createFalClient, ApiError, ValidationError } = require("@fal-ai/client");

const APP = "minimax/h3-max/text-to-video";
const checks = [];
function check(cond, what) {
  if (!cond) throw new Error(`check failed: ${what}`);
}

function rewrite(url) {
  return url
    .replace(/^https:\/\/queue\.fal\.run\//, `${base}/`)
    .replace(/^https:\/\/fal\.run\//, `${base}/run/`)
    .replace(/^https:\/\/rest\.fal\.ai\//, `${base}/`);
}

async function suite(mode, fal) {
  // submit + status + result
  const { request_id } = await fal.queue.submit(APP, { input: { prompt: `js ${mode}`, seed: 3 } });
  check(/^[0-9a-f-]{36}$/.test(request_id), `request id ${request_id}`);
  let st;
  for (let i = 0; i < 2000; i++) {
    st = await fal.queue.status(APP, { requestId: request_id, logs: true });
    check(["IN_QUEUE", "IN_PROGRESS", "COMPLETED"].includes(st.status), st.status);
    check(st.response_url, "response_url on every status");
    if (st.status === "COMPLETED") break;
    await new Promise((r) => setTimeout(r, 50));
  }
  check(st.status === "COMPLETED" && Array.isArray(st.logs) && st.logs.length > 0, JSON.stringify(st));
  const res = await fal.queue.result(APP, { requestId: request_id });
  check(res.requestId === request_id, "Result.requestId from x-fal-request-id");
  check(res.data.video.content_type === "video/mp4", JSON.stringify(res.data));
  const dl = await fetch(res.data.video.url);
  check(dl.ok && (await dl.arrayBuffer()).byteLength === res.data.video.file_size, "download");
  checks.push(`${mode}: submit/status/result`);

  // subscribe: polling, then SSE streaming
  const updates = [];
  const r1 = await fal.subscribe(APP, { input: { prompt: "js poll" }, logs: true, pollInterval: 50, onQueueUpdate: (u) => updates.push(u.status) });
  check(r1.data.video && updates.at(-1) === "COMPLETED", `polling ${updates}`);
  const streamed = [];
  const r2 = await fal.subscribe(APP, { input: { prompt: "js stream" }, mode: "streaming", onQueueUpdate: (u) => streamed.push(u.status) });
  check(r2.data.video && streamed.includes("COMPLETED"), `streaming ${streamed}`);
  checks.push(`${mode}: subscribe polling + streaming`);

  // run (sync)
  const r3 = await fal.run(APP, { input: { prompt: "js run" } });
  check(r3.data.video && r3.requestId, "run");
  checks.push(`${mode}: run`);

  // Blob input: transformInput uploads it through storage/upload/initiate.
  const png = new Blob([readFileSync(pngPath)], { type: "image/png" });
  const r4 = await fal.subscribe("minimax/h3-max/image-to-video", { input: { prompt: "js blob", image_url: png }, pollInterval: 50 });
  check(r4.data.video, "i2v with an uploaded Blob");
  checks.push(`${mode}: storage upload + i2v`);

  // cancel: the second of two back-to-back jobs
  const a = await fal.queue.submit(APP, { input: { prompt: "js cancel a" } });
  const b = await fal.queue.submit(APP, { input: { prompt: "js cancel b" } });
  await fal.queue.cancel(APP, { requestId: b.request_id });
  let sb;
  do {
    sb = await fal.queue.status(APP, { requestId: b.request_id });
  } while (sb.status !== "COMPLETED");
  check(sb.error_type === "client_cancelled", JSON.stringify(sb));
  let err;
  try {
    await fal.queue.cancel(APP, { requestId: b.request_id });
  } catch (e) {
    err = e;
  }
  check(err instanceof ApiError && err.status === 400, `second cancel: ${err}`);
  await fal.queue.subscribeToStatus(APP, { requestId: a.request_id, pollInterval: 50 });
  checks.push(`${mode}: cancel`);

  // validation errors surface as ValidationError with fal locs
  err = undefined;
  try {
    await fal.queue.submit(APP, { input: { prompt: "p", duration: 99 } });
  } catch (e) {
    err = e;
  }
  check(err instanceof ValidationError && err.fieldErrors[0].loc.join(".") === "body.duration", `422: ${err}`);
  checks.push(`${mode}: ValidationError`);
}

async function main() {
  await suite(
    "requestMiddleware",
    createFalClient({ credentials: key, requestMiddleware: async (req) => ({ ...req, url: rewrite(req.url) }) }),
  );
  await suite("proxyUrl", createFalClient({ credentials: key, proxyUrl: { url: `${base}/fal/proxy`, when: "always" } }));
  console.log(JSON.stringify({ checks }));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
