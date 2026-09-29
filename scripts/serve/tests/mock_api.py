#!/usr/bin/env python3
"""One local HTTP server standing in for every API the release scripts call
(scripts/serve/tests/release.test.sh):

- Cloudflare D1: /client/v4/accounts, …/d1/database?name=, …/query (SQLite)
- Runpod REST v1 (/v1/pods, /v1/endpoints, /v1/templates) and GraphQL (/graphql)
- a GHCR-like registry: /token, /v2/<repo>/manifests/<ref>, /v2/<repo>/blobs/<digest>
- GitHub workflow_dispatch: /repos/<o>/<r>/actions/workflows/<f>/dispatches
- fv-serve pods: /pod/<id>/health, /pod/<id>/fv/v1/internal/{drain,status}
- test hooks: GET /__state, POST /__seed (merge), POST /__tag {tag, digest},
  POST /__build {sha, keys} (a build's images and sha tags), POST /__sql {sql}

Credentials are checked (Bearer tokens below) so a script that forgets a
header fails. Usage: mock_api.py <port file>; prints the port there.
"""

import hashlib
import json
import re
import sqlite3
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CF_TOKEN = "test-cf-token"
RUNPOD_KEY = "test-runpod-key"
GH_TOKEN = "test-gh-token"
INTERNAL = "test-internal-token"

LOCK = threading.Lock()
DB = sqlite3.connect(":memory:", check_same_thread=False)
DB.row_factory = sqlite3.Row
STATE = {
    "pods": {},
    "endpoints": {},
    "templates": {},
    "tags": {},
    "manifests": {},
    "blobs": {},
    "dispatches": [],
    "drains": [],
    "patches": [],
    "requests": 0,
    # GPU type -> quoted secure $/hr (GraphQL gpuTypes); others 0.5.
    "gpu_prices": {"NVIDIA H100 80GB HBM3": 2.99},
    # Every POST /v1/pods body's GPU types, in order.
    "pod_creates": [],
}


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S.000 +0000 UTC", time.gmtime())


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, code, body=None, headers=None):
        data = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return {}

    def bearer(self):
        return (self.headers.get("authorization") or "").removeprefix("Bearer ").strip()

    def do_HEAD(self):
        self.route()

    def do_GET(self):
        self.route()

    def do_POST(self):
        self.route()

    def do_PATCH(self):
        self.route()

    def do_DELETE(self):
        self.route()

    def route(self):
        with LOCK:
            STATE["requests"] += 1
            try:
                self._route()
            except Exception as e:  # noqa: BLE001 — a test double
                self.reply(500, {"error": repr(e)})

    def _route(self):
        p = self.path.split("?")[0]
        q = self.path.split("?")[1] if "?" in self.path else ""
        m = self.command
        # --- test hooks
        if p == "/__state":
            return self.reply(200, STATE)
        if p == "/__seed":
            for k, v in self.body().items():
                if isinstance(v, dict) and isinstance(STATE.get(k), dict):
                    STATE[k].update(v)
                else:
                    STATE[k] = v
            return self.reply(200, {"ok": True})
        if p == "/__tag":
            b = self.body()
            STATE["tags"][b["tag"]] = b["digest"]
            return self.reply(200, {"ok": True})
        if p == "/__build":
            b = self.body()
            out = {}
            for k in b["keys"]:
                man_d, man, cfg_d, cfg = image(b["sha"], k)
                STATE["manifests"][man_d] = man
                STATE["blobs"][cfg_d] = cfg
                STATE["tags"][("sha-" if k == "debug" else k + "-sha-") + b["sha"][:7]] = man_d
                out[k] = man_d
            return self.reply(200, out)
        if p == "/__sql":
            rows = [dict(r) for r in DB.execute(self.body()["sql"]).fetchall()]
            return self.reply(200, rows)
        # --- Cloudflare
        if p.startswith("/client/v4/"):
            if self.bearer() != CF_TOKEN:
                return self.reply(403, {"success": False, "errors": [{"message": "bad token"}]})
            if p == "/client/v4/accounts":
                return self.reply(200, {"success": True, "result": [{"id": "acct-test"}]})
            if re.fullmatch(r"/client/v4/accounts/acct-test/d1/database", p):
                return self.reply(200, {"success": True, "result": [{"name": "fv-jobs", "uuid": "db-test"}]})
            if p == "/client/v4/accounts/acct-test/d1/database/db-test/query" and m == "POST":
                b = self.body()
                sql, params = b.get("sql", ""), b.get("params", [])
                try:
                    if not params and sql.count(";") > 1:
                        DB.executescript(sql)
                        rows = []
                    else:
                        cur = DB.execute(sql, params)
                        rows = [dict(r) for r in cur.fetchall()]
                        DB.commit()
                except sqlite3.Error as e:
                    return self.reply(400, {"success": False, "errors": [{"message": str(e)}]})
                return self.reply(200, {"success": True, "result": [{"results": rows, "success": True, "meta": {}}]})
            return self.reply(404, {"success": False, "errors": [{"message": "no route " + p}]})
        # --- Runpod
        if p == "/graphql":
            if self.bearer() != RUNPOD_KEY:
                return self.reply(401, {"errors": ["bad key"]})
            query = self.body().get("query", "")
            gm = re.search(r'gpuTypes\(input: \{id: "([^"]*)"\}\)', query)
            if gm:
                price = STATE["gpu_prices"].get(gm.group(1), 0.5)
                gt = {"id": gm.group(1), "securePrice": price, "communityPrice": price * 0.8}
                if "lowestPrice" in query:
                    gt["lowestPrice"] = {"uninterruptablePrice": price}
                return self.reply(200, {"data": {"gpuTypes": [gt]}})
            return self.reply(200, {"data": {"myself": {"clientBalance": 100.0, "currentSpendPerHr": 1.0}}})
        if p.startswith("/v1/"):
            if self.bearer() != RUNPOD_KEY:
                return self.reply(401, {"error": "bad key"})
            parts = p.split("/")[2:]
            coll = {"pods": "pods", "endpoints": "endpoints", "templates": "templates"}.get(parts[0])
            if not coll:
                return self.reply(404, {"error": "no route"})
            items = STATE[coll]
            if len(parts) == 1:
                if m == "GET":
                    return self.reply(200, list(items.values()))
                if m == "POST":
                    b = self.body()
                    if coll == "pods":
                        STATE["pod_creates"].append(b.get("gpuTypeIds") or [])
                    i = uuid.uuid4().hex[:14]
                    b.update({"id": i, "createdAt": now_str(), "desiredStatus": "RUNNING", "costPerHr": 0.5})
                    items[i] = b
                    return self.reply(200, b)
            else:
                i = parts[1]
                if i not in items:
                    return self.reply(404, {"error": "not found"})
                if m == "GET":
                    return self.reply(200, items[i])
                if m == "DELETE":
                    del items[i]
                    return self.reply(200, {})
                if m == "PATCH":
                    b = self.body()
                    STATE["patches"].append({"coll": coll, "id": i, "keys": sorted(b.keys())})
                    items[i].update(b)
                    return self.reply(200, items[i])
            return self.reply(405, {"error": "method"})
        # --- registry
        if p == "/token":
            return self.reply(200, {"token": "anon"})
        mm = re.fullmatch(r"/v2/(.+)/(manifests|blobs)/(.+)", p)
        if mm:
            if self.bearer() != "anon":
                return self.reply(401, {"errors": ["unauthorized"]})
            kind, ref = mm.group(2), mm.group(3)
            if kind == "manifests":
                d = ref if ref.startswith("sha256:") else STATE["tags"].get(ref)
                if not d or d not in STATE["manifests"]:
                    return self.reply(404, {"errors": ["MANIFEST_UNKNOWN"]})
                return self.reply(200, STATE["manifests"][d], {"docker-content-digest": d})
            if ref not in STATE["blobs"]:
                return self.reply(404, {"errors": ["BLOB_UNKNOWN"]})
            return self.reply(200, STATE["blobs"][ref])
        # --- GitHub
        mm = re.fullmatch(r"/repos/([^/]+)/([^/]+)/actions/workflows/([^/]+)/dispatches", p)
        if mm and m == "POST":
            if self.bearer() != GH_TOKEN:
                return self.reply(401, {"message": "Bad credentials"})
            STATE["dispatches"].append({"workflow": mm.group(3), **self.body()})
            return self.reply(204)
        # --- fv-serve pods
        mm = re.fullmatch(r"/pod/([^/]+)(/.*)", p)
        if mm:
            pod, sub = mm.group(1), mm.group(2)
            if pod not in STATE["pods"]:
                return self.reply(502, {"error": "no such pod"})
            env = STATE["pods"][pod].get("env") or {}
            if sub == "/health":
                return self.reply(200, {"status": "ok", "model_loaded": True, "state": "AVAILABLE", "version": "0.1.0",
                                        "build": {"git_sha": env.get("MOCK_GIT_SHA", "unknown"),
                                                  "image": {"digest": env.get("FV_IMAGE_DIGEST")}}})
            if (self.headers.get("x-fv-internal-token") or "") != INTERNAL:
                return self.reply(401, {"error": "internal token"})
            if sub == "/fv/v1/internal/drain" and m == "POST":
                STATE["drains"].append(pod)
                return self.reply(200, {"draining": True})
            if sub == "/fv/v1/internal/status":
                return self.reply(200, {"object": "fv.worker", "draining": pod in STATE["drains"],
                                        "stats": {"queued_batch": 0, "queued_stream": 0, "running": 0, "sessions": 0}})
        return self.reply(404, {"error": "no route " + p, "q": q})


def image(sha, key):
    """A manifest + config blob for one image of a build (sha, key)."""
    cfg = {"config": {"Labels": {"org.opencontainers.image.revision": sha,
                                 **({"dev.fastvideo.variant": key} if key != "debug" else {})}}}
    cfg_d = "sha256:" + hashlib.sha256(json.dumps(cfg).encode()).hexdigest()
    man = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
           "config": {"digest": cfg_d}, "layers": []}
    man_d = "sha256:" + hashlib.sha256((sha + key).encode()).hexdigest()
    return man_d, man, cfg_d, cfg


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    with open(sys.argv[1], "w") as f:
        f.write(str(srv.server_address[1]))
    srv.serve_forever()


if __name__ == "__main__":
    main()
