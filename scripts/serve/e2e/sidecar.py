#!/usr/bin/env python3
"""WP-18 GPU E2E sidecar: runs on the Runpod pod next to fv-serve (port 8001,
published through the pod's HTTP proxy). Started by pod-boot.sh.

- `POST /hook/<anything>`: a webhook / callback receiver (no auth). Records
  path, headers and body; a JSON body with `challenge` gets it echoed
  (MiniMax callback handshake). fv-serve reaches it on 127.0.0.1:8001
  (`FV_CALLBACKS_ALLOW_PRIVATE=1`).
- `GET /hooks`: the recorded deliveries (body base64).
- `PUT /bundle`: a tar.gz of test files, extracted under /e2e.
- `POST /exec` `{"cmd": "..."}`: runs a shell command in the background;
  returns `{"id"}`. `GET /exec/<id>`: `{"done", "rc", "secs", "out"}` (last
  64 KiB of combined output).

Everything but `/hook/*` needs `X-Sidecar-Token: $FV_SIDECAR_TOKEN`. The pod
is ephemeral and single-tenant; the token only keeps strangers off `/exec`.
"""

import base64
import hmac
import io
import json
import os
import subprocess
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.get("FV_SIDECAR_TOKEN", "")
HOOKS = []
JOBS = {}
LOCK = threading.Lock()


def run_job(jid, cmd):
    t0 = time.monotonic()
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd="/e2e")
    buf = bytearray()
    for chunk in iter(lambda: p.stdout.read(4096), b""):
        with LOCK:
            buf += chunk
            del buf[:-65536]
            JOBS[jid]["out"] = bytes(buf)
    rc = p.wait()
    with LOCK:
        JOBS[jid].update(done=True, rc=rc, secs=round(time.monotonic() - t0, 2))


class H(BaseHTTPRequestHandler):
    def _send(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self):
        ok = bool(TOKEN) and hmac.compare_digest(self.headers.get("x-sidecar-token", ""), TOKEN)
        if not ok:
            self._send(401, {"error": "token"})
        return ok

    def _body(self):
        n = int(self.headers.get("content-length") or 0)
        return self.rfile.read(n) if n else b""

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/hook/"):
            body = self._body()
            rec = {
                "t": time.time(),
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body_b64": base64.b64encode(body).decode(),
            }
            with LOCK:
                HOOKS.append(rec)
            try:
                j = json.loads(body or b"{}")
            except ValueError:
                j = {}
            if isinstance(j, dict) and "challenge" in j:
                return self._send(200, {"challenge": j["challenge"]})
            return self._send(200, {})
        if not self._authed():
            return
        if self.path == "/exec":
            cmd = json.loads(self._body())["cmd"]
            jid = str(len(JOBS) + 1)
            with LOCK:
                JOBS[jid] = {"done": False, "rc": None, "secs": None, "out": b""}
            threading.Thread(target=run_job, args=(jid, cmd), daemon=True).start()
            return self._send(200, {"id": jid})
        self._send(404, {})

    def do_PUT(self):  # noqa: N802
        if not self._authed():
            return
        if self.path == "/bundle":
            with tarfile.open(fileobj=io.BytesIO(self._body()), mode="r:gz") as t:
                t.extractall("/e2e")
            return self._send(200, {"ok": True})
        self._send(404, {})

    def do_GET(self):  # noqa: N802
        if self.path == "/ping":
            return self._send(200, {"ok": True})
        if not self._authed():
            return
        if self.path == "/hooks":
            with LOCK:
                return self._send(200, list(HOOKS))
        if self.path.startswith("/exec/"):
            with LOCK:
                j = dict(JOBS.get(self.path[6:], {}))
            if not j:
                return self._send(404, {})
            j["out"] = j["out"].decode("utf-8", "replace")
            return self._send(200, j)
        self._send(404, {})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    os.makedirs("/e2e", exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", 8001), H).serve_forever()
