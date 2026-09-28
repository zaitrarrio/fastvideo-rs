#!/usr/bin/env python3
"""fal webhooks (design §4.4, research-fal §9.7): the real `fal-client`
submits with `webhook_url`, a local receiver gets the POST, and verifies it
the way fal's docs describe: fetch the JWKS (`/.well-known/jwks.json` on our
server instead of rest.fal.ai), check the timestamp is within ±300 s, and
verify the hex Ed25519 signature over
`request_id\\nuser_id\\ntimestamp\\nhex(sha256(body))` with PyNaCl.

Needs `SSL_CERT_FILE` (the compat CA), `FAL_QUEUE_RUN_HOST`/`FAL_RUN_HOST`
pointing at the TLS front, and fv-serve started with
`FV_CALLBACKS_ALLOW_PRIVATE=1` so it may reach the loopback receiver.
"""

import base64
import hashlib
import json
import time

import fal_client
import httpx
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from common import Receiver, Suite

APP = "minimax/h3-max/text-to-video"


def b64url(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def unsigned(v):
    """`v` with the query (the `exp`/`sig` of signed file URLs, minted per
    render) cut from every URL, so a webhook payload and a later result compare."""
    if isinstance(v, dict):
        return {k: unsigned(x) for k, x in v.items()}
    if isinstance(v, list):
        return [unsigned(x) for x in v]
    if isinstance(v, str) and v.startswith(("http://", "https://")):
        return v.split("?", 1)[0]
    return v


def fetch_jwks(base):
    r = httpx.get(f"{base}/.well-known/jwks.json", timeout=10)
    r.raise_for_status()
    return r.json()["keys"]


def verify(keys, headers, body):
    """fal's documented verification; returns True when one key verifies."""
    rid = headers.get("x-fal-webhook-request-id")
    uid = headers.get("x-fal-webhook-user-id")
    ts = headers.get("x-fal-webhook-timestamp")
    sig = headers.get("x-fal-webhook-signature")
    if not all((rid, uid, ts, sig)):
        return False
    if abs(int(time.time()) - int(ts)) > 300:
        return False
    message = "\n".join([rid, uid, ts, hashlib.sha256(body).hexdigest()]).encode()
    for k in keys:
        if k.get("kty") != "OKP" or k.get("crv") != "Ed25519":
            continue
        try:
            VerifyKey(b64url(k["x"])).verify(message, bytes.fromhex(sig))
            return True
        except (BadSignatureError, ValueError):
            continue
    return False


def main(s, a):
    keys = fetch_jwks(a.base)
    s.check(keys and all(k["kty"] == "OKP" and k["crv"] == "Ed25519" and k.get("kid") for k in keys), f"jwks {keys}")
    s.ok("JWKS: OKP/Ed25519 keys")

    rx = Receiver(a.hook_host)
    try:
        c = fal_client.SyncClient(key=a.key)
        ok_h = c.submit(APP, {"prompt": "compat: webhook ok"}, webhook_url=rx.url + "/fal/ok")
        bad_h = c.submit(APP, {"prompt": "[fake:fail] compat: webhook error"}, webhook_url=rx.url + "/fal/err")
        got = rx.wait(lambda g: {p for p, _, _ in g} >= {"/fal/ok", "/fal/err"}, timeout=120)
        by_path = {p: (h, b) for p, h, b in got}

        h, raw = by_path["/fal/ok"]
        s.check(verify(keys, h, raw), f"signature verifies: {sorted(h)}")
        s.check(h["x-fal-webhook-request-id"] == ok_h.request_id, "X-Fal-Webhook-Request-Id")
        body = json.loads(raw)
        s.check(body["request_id"] == ok_h.request_id and body["status"] == "OK", body)
        s.check("gateway_request_id" in body and body["payload"]["video"]["content_type"] == "video/mp4", body)
        s.check(unsigned(body["payload"]) == unsigned(ok_h.get()), "webhook payload equals the result (URLs unsigned)")
        s.ok("OK webhook: payload + Ed25519 signature")

        h, raw = by_path["/fal/err"]
        s.check(verify(keys, h, raw), "error webhook signature verifies")
        body = json.loads(raw)
        s.check(body["request_id"] == bad_h.request_id and body["status"] == "ERROR" and body.get("error"), body)
        s.ok("ERROR webhook")

        # Tampering is detected.
        tampered = raw.replace(b"ERROR", b"OK")
        s.check(not verify(keys, h, tampered), "a changed body must not verify")
        stale = dict(h, **{"x-fal-webhook-timestamp": str(int(h["x-fal-webhook-timestamp"]) - 3600)})
        s.check(not verify(keys, stale, raw), "a stale timestamp must not verify")
        s.ok("tampered body / stale timestamp rejected")

        # `?fal_webhook=` on a raw queue submit is the same thing.
        r = httpx.post(
            f"{a.base}/{APP}",
            params={"fal_webhook": rx.url + "/fal/raw"},
            headers={"Authorization": f"Key {a.key}"},
            json={"prompt": "compat: raw fal_webhook"},
            timeout=30,
        )
        s.check(r.status_code == 200, r.text)
        got = rx.wait(lambda g: any(p == "/fal/raw" for p, _, _ in g), timeout=120)
        h, raw = next((h, b) for p, h, b in got if p == "/fal/raw")
        s.check(verify(keys, h, raw) and json.loads(raw)["request_id"] == r.json()["request_id"], "raw webhook")
        s.ok("?fal_webhook= query parameter")
    finally:
        rx.close()


if __name__ == "__main__":
    Suite("fal-webhook").run(main)
