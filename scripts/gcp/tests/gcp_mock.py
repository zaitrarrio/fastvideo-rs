#!/usr/bin/env python3
"""A fake Google Cloud API for scripts/gcp/tests/vm.test.sh (docs/serve/deploy-gcp.md).

- POST /token: the OAuth JWT-bearer grant. Checks grant_type and that the
  assertion is a three-part JWT whose claims name the key's account and this
  token endpoint; answers a fixed access token. (The signature itself is
  checked offline by `auth.sh self-test`.)
- /compute/v1/...: the Compute Engine routes vm.sh uses (instances insert /
  get / delete / setMetadata / serialPort / aggregated list, firewalls,
  disks, regions, operations wait). Every call needs the bearer token.
  Operations finish at once; their selfLink points back here.

Test hooks: GET /__state, POST /__seed (merge into STATE).
Usage: gcp_mock.py <port file>
"""

import base64
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = "ya29.mock-access-token-0123456789"
LOCK = threading.Lock()
STATE = {
    "instances": {},  # name -> instance (zone inside)
    "firewalls": {},
    "disks": {},  # name -> disk
    "ops": 0,
    "tokens_issued": 0,
    "requests": [],
    "inserts": [],
    "deletes": [],
}
PORT = 0


def b64json(part):
    part += "=" * (-len(part) % 4)
    return json.loads(base64.urlsafe_b64decode(part))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("content-length") or 0)
        return self.rfile.read(n).decode() if n else ""

    def base(self):
        return f"http://127.0.0.1:{PORT}/compute/v1"

    def op(self, scope, target):
        STATE["ops"] += 1
        name = f"operation-{STATE['ops']}"
        return {"kind": "compute#operation", "name": name, "status": "RUNNING", "targetLink": target,
                "selfLink": f"{self.base()}/{scope}/operations/{name}"}

    def authed(self):
        return self.headers.get("authorization") == f"Bearer {TOKEN}"

    def route(self, method):
        url = urllib.parse.urlparse(self.path)
        p, q = url.path, urllib.parse.parse_qs(url.query)
        raw = self.body() if method in ("POST", "PUT", "PATCH") else ""
        with LOCK:
            STATE["requests"].append({"method": method, "path": p, "auth": self.authed()})
        if p == "/__state":
            with LOCK:
                return self.send(200, STATE)
        if p == "/__seed":
            with LOCK:
                for k, v in json.loads(raw or "{}").items():
                    if isinstance(v, dict) and isinstance(STATE.get(k), dict):
                        STATE[k].update(v)
                    else:
                        STATE[k] = v
            return self.send(200, {})
        if p == "/token":
            form = urllib.parse.parse_qs(raw)
            a = (form.get("assertion") or [""])[0].split(".")
            if form.get("grant_type") != ["urn:ietf:params:oauth:grant-type:jwt-bearer"] or len(a) != 3:
                return self.send(400, {"error": "invalid_grant", "error_description": "bad assertion"})
            try:
                claims = b64json(a[1])
            except ValueError:
                return self.send(400, {"error": "invalid_grant", "error_description": "claims"})
            if claims.get("aud") != f"http://127.0.0.1:{PORT}/token" or not claims.get("iss"):
                return self.send(400, {"error": "invalid_grant", "error_description": "aud/iss"})
            with LOCK:
                STATE["tokens_issued"] += 1
            return self.send(200, {"access_token": TOKEN, "expires_in": 3599, "token_type": "Bearer"})
        if not p.startswith("/compute/v1/"):
            return self.send(404, {"error": {"code": 404, "message": "no route"}})
        if not self.authed():
            return self.send(401, {"error": {"code": 401, "message": "Request had invalid authentication credentials."}})
        parts = p[len("/compute/v1/"):].split("/")
        body = json.loads(raw) if raw else None
        with LOCK:
            return self.compute(method, parts, q, body)

    def compute(self, method, parts, q, body):
        # projects/{p}/...
        if parts[:1] != ["projects"] or len(parts) < 3:
            # the boot image family
            return self.send(200, {"name": "ubuntu-accelerator-2404", "status": "READY"})
        project, rest = parts[1], parts[2:]
        if rest[-1:] == ["wait"] and "operations" in rest:
            return self.send(200, {"name": rest[-2], "status": "DONE"})
        if rest[:2] == ["aggregated", "instances"]:
            flt = (q.get("filter") or [""])[0]
            want = flt.split("=", 1)[1] if flt.startswith("labels.fv-owner=") else None
            by_zone = {}
            for i in STATE["instances"].values():
                if want is not None and i.get("labels", {}).get("fv-owner") != want:
                    continue
                by_zone.setdefault(f"zones/{i['_zone']}", {"instances": []})["instances"].append(public(i))
            return self.send(200, {"items": by_zone})
        if rest[:2] == ["global", "firewalls"]:
            if len(rest) == 2 and method == "GET":
                return self.send(200, {"items": list(STATE["firewalls"].values())})
            if len(rest) == 2 and method == "POST":
                if body["name"] in STATE["firewalls"]:
                    return self.send(409, {"error": {"code": 409, "message": "already exists"}})
                STATE["firewalls"][body["name"]] = body
                return self.send(200, self.op(f"projects/{project}/global", body["name"]))
            name = rest[2]
            if name not in STATE["firewalls"]:
                return self.send(404, {"error": {"code": 404, "message": f"firewall {name} not found"}})
            if method == "DELETE":
                del STATE["firewalls"][name]
                return self.send(200, self.op(f"projects/{project}/global", name))
            return self.send(200, STATE["firewalls"][name])
        if rest[:1] == ["regions"]:
            return self.send(200, {"name": rest[1], "quotas": [{"metric": "NVIDIA_RTX_PRO_6000_GPUS", "usage": 0, "limit": 1}]})
        if rest[:1] == ["zones"] and len(rest) >= 3:
            zone, kind = rest[1], rest[2]
            scope = f"projects/{project}/zones/{zone}"
            if kind == "machineTypes":
                return self.send(200, {"name": rest[3]})
            if kind == "disks":
                d = STATE["disks"].get(rest[3]) if len(rest) > 3 else None
                if d is None or d.get("_zone", zone) != zone:
                    return self.send(404, {"error": {"code": 404, "message": "disk not found"}})
                return self.send(200, d)
            if kind == "instances":
                if len(rest) == 3 and method == "POST":
                    name = body["name"]
                    if name in STATE["instances"]:
                        return self.send(409, {"error": {"code": 409, "message": "already exists"}})
                    inst = dict(body)
                    inst.update({"_zone": zone, "zone": f"{self.base()}/{scope}", "status": "RUNNING",
                                 "creationTimestamp": "2026-10-06T00:00:00Z",
                                 "networkInterfaces": [{"accessConfigs": [{"natIP": "127.0.0.1"}]}]})
                    inst.setdefault("metadata", {})["fingerprint"] = "fp1"
                    STATE["instances"][name] = inst
                    STATE["inserts"].append(body)
                    return self.send(200, self.op(scope, name))
                name = rest[3]
                inst = STATE["instances"].get(name)
                if inst is None or inst["_zone"] != zone:
                    return self.send(404, {"error": {"code": 404, "message": f"instance {name} not found"}})
                if len(rest) == 4 and method == "DELETE":
                    del STATE["instances"][name]
                    STATE["deletes"].append(name)
                    return self.send(200, self.op(scope, name))
                if len(rest) == 4:
                    return self.send(200, public(inst))
                if rest[4] == "setMetadata":
                    inst["metadata"] = {"items": body.get("items", []), "fingerprint": "fp2"}
                    return self.send(200, self.op(scope, name))
                if rest[4] == "serialPort":
                    return self.send(200, {"contents": "FV-GCP STARTED\nFV-GCP READY after 1s from boot\n", "next": "40"})
        if not rest:
            return self.send(200, {"name": project, "quotas": [{"metric": "GPUS_ALL_REGIONS", "usage": 0, "limit": 1}]})
        return self.send(404, {"error": {"code": 404, "message": "no route " + "/".join(rest)}})

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_DELETE(self):
        self.route("DELETE")


def public(i):
    out = {k: v for k, v in i.items() if not k.startswith("_")}
    out.setdefault("zone", f"http://127.0.0.1:{PORT}/compute/v1/projects/p/zones/{i.get('_zone', '')}")
    return out


def main():
    global PORT
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    PORT = srv.server_address[1]
    with open(sys.argv[1], "w") as f:
        f.write(str(PORT))
    srv.serve_forever()


if __name__ == "__main__":
    main()
