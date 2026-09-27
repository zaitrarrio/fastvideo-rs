// fal webhooks through the real `@fal-ai/client` (design §4.4, research-fal
// §9.7): `fal.queue.submit(app, {input, webhookUrl})` with
// `requestMiddleware` pointing at fv-serve, a local receiver gets the POST,
// and verifies it the way fal documents for Node: fetch the JWKS (ours, not
// rest.fal.ai), check the timestamp window (±300 s), and verify the hex
// Ed25519 signature over `request_id\nuser_id\ntimestamp\nhex(sha256(body))`
// with node:crypto. Run by tests/compat/run.sh (suite fal-webhook).
//
// usage: node fal_webhook.mjs <modules dir> <base url> <key>
// Needs NODE_EXTRA_CA_CERTS (the compat CA) and fv-serve started with
// FV_CALLBACKS_ALLOW_PRIVATE=1. Prints one JSON summary line last.

import { createHash, createPublicKey, verify as edVerify } from "node:crypto";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import path from "node:path";

const [, , modulesDir, base, key] = process.argv;
const require = createRequire(path.join(modulesDir, "package.json"));
const { createFalClient } = require("@fal-ai/client");

const APP = "minimax/h3-max/text-to-video";
const checks = [];
function check(cond, what) {
  if (!cond) throw new Error(`check failed: ${what}`);
}

function verify(jwks, headers, body) {
  const rid = headers["x-fal-webhook-request-id"];
  const uid = headers["x-fal-webhook-user-id"];
  const ts = headers["x-fal-webhook-timestamp"];
  const sig = headers["x-fal-webhook-signature"];
  if (!rid || !uid || !ts || !sig) return false;
  if (Math.abs(Math.floor(Date.now() / 1000) - Number.parseInt(ts, 10)) > 300) return false;
  const msg = Buffer.from([rid, uid, ts, createHash("sha256").update(body).digest("hex")].join("\n"));
  return jwks.some((k) => {
    if (k.kty !== "OKP" || k.crv !== "Ed25519") return false;
    try {
      const pub = createPublicKey({ key: { kty: "OKP", crv: "Ed25519", x: k.x }, format: "jwk" });
      return edVerify(null, msg, pub, Buffer.from(sig, "hex"));
    } catch {
      return false;
    }
  });
}

// The receiver: records every POST as {path, headers, body}.
const got = [];
const waiters = [];
const server = createServer((req, res) => {
  const parts = [];
  req.on("data", (c) => parts.push(c));
  req.on("end", () => {
    got.push({ path: req.url, headers: req.headers, body: Buffer.concat(parts) });
    res.writeHead(200, { "content-type": "application/json" }).end("{}");
    for (const w of [...waiters]) w();
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const hook = `http://127.0.0.1:${server.address().port}`;
const waitFor = (p, ms = 120000) =>
  new Promise((resolve, reject) => {
    const look = () => {
      const hit = got.find((g) => g.path === p);
      if (hit) {
        waiters.splice(waiters.indexOf(look), 1);
        clearTimeout(timer);
        resolve(hit);
      }
    };
    const timer = setTimeout(() => {
      waiters.splice(waiters.indexOf(look), 1);
      reject(new Error(`no webhook on ${p}; got ${got.map((g) => g.path)}`));
    }, ms);
    waiters.push(look);
    look();
  });

try {
  const jwks = (await (await fetch(`${base}/.well-known/jwks.json`)).json()).keys;
  check(Array.isArray(jwks) && jwks.length > 0, "jwks keys");

  const fal = createFalClient({
    credentials: key,
    requestMiddleware: async (req) => ({
      ...req,
      url: req.url.replace(/^https:\/\/queue\.fal\.run\//, `${base}/`).replace(/^https:\/\/fal\.run\//, `${base}/run/`),
    }),
  });

  const ok = await fal.queue.submit(APP, { input: { prompt: "js webhook ok" }, webhookUrl: `${hook}/js/ok` });
  const bad = await fal.queue.submit(APP, { input: { prompt: "[fake:fail] js webhook" }, webhookUrl: `${hook}/js/err` });
  const okHit = await waitFor("/js/ok");
  check(verify(jwks, okHit.headers, okHit.body), "OK webhook signature verifies (node:crypto Ed25519)");
  const okBody = JSON.parse(okHit.body);
  check(okBody.request_id === ok.request_id && okBody.status === "OK", JSON.stringify(okBody));
  const result = await fal.queue.result(APP, { requestId: ok.request_id });
  check(JSON.stringify(okBody.payload) === JSON.stringify(result.data), "webhook payload equals the result");
  checks.push("queue.submit(webhookUrl): OK webhook, Ed25519 verified");

  const errHit = await waitFor("/js/err");
  check(verify(jwks, errHit.headers, errHit.body), "ERROR webhook signature verifies");
  const errBody = JSON.parse(errHit.body);
  check(errBody.request_id === bad.request_id && errBody.status === "ERROR" && errBody.error, JSON.stringify(errBody));
  check(!verify(jwks, errHit.headers, Buffer.from(errHit.body.toString().replace("ERROR", "OK"))), "tampered body rejected");
  checks.push("queue.submit(webhookUrl): ERROR webhook; tampering rejected");
} finally {
  server.close();
}
console.log(JSON.stringify({ suite: "fal-webhook-js", ok: true, checks }));
