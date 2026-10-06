#!/usr/bin/env python3
"""A fake CloudRift API (rift-server 0.62 shapes, docs/ops/cloudrift.md) for
scripts/gpu/tests/cloudrift.test.sh. Every call is POST /api/v1/<path> with
{"version", "data"}; the answer is {"version", "data"}. X-API-Key is checked
(instance-types/list is public, as on the real server).

The same server also stands in for the rented containers' public port 8000
(fv-serve: /healthz, /fv/v1/capabilities, /fv/v1/jobs): instances report
host 127.0.0.1 and map container port 8000 to this server's port.

Test hooks: GET /__state, POST /__seed (merge into STATE).
Usage: cloudrift_mock.py <port file>
"""

import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "test-cloudrift-key-0123456789"
RENT_MIN_VERSION = "2026-09-08"
LOCK = threading.Lock()
STATE = {
    "balance": 50.0,
    "gpu_util": 40.0,
    "instances": {},
    "rents": [],
    "terminates": [],
    "requests": [],
    "jobs": {},
    "catalog": [
        {"name": "rtxpro6000-t", "brand_short": "RTX PRO 6000", "price": 139.36, "free": {"us-t-1": 1}},
        {"name": "rtx59-t", "brand_short": "RTX 5090", "price": 65.0, "free": {}},
        {"name": "rtx49-t", "brand_short": "RTX 4090", "price": 39.0, "free": {"eu-t-1": 2}},
    ],
}
PORT = 0


def catalog():
    out = []
    for t in STATE["catalog"]:
        out.append({
            "name": t["name"], "brand_short": t["brand_short"], "manufacturer": "NVIDIA", "cost_per_hour": int(t["price"]),
            "datacenters": [{"name": dc, "provider_name": "p"} for dc in t["free"]],
            "variants": [{
                "name": f"{t['name']}.{g}", "gpu_count": g, "cpu_count": 8 * g, "logical_cpu_count": 16 * g,
                "dram": 64 * g << 30, "vram": 96 << 30, "disk": 500 << 30, "cost_per_hour": t["price"] * g,
                "nodes": 1, "nodes_per_dc": dict(t["free"]), "available_nodes": sum(t["free"].values()),
                "available_nodes_per_dc": dict(t["free"]), "cpu_sharing_policy": "Dedicated",
                "ip_availability_per_dc": {dc: {"public_ips": True} for dc in t["free"]}, "volume_types_per_dc": {},
            } for g in (1, 2)],
        })
    return out


def select(sel):
    insts = list(STATE["instances"].values())
    if "ById" in sel:
        return [i for i in insts if i["id"] in sel["ById"]]
    if "ByTags" in sel:
        need = sel["ByTags"].get("all", [])
        return [i for i in insts if all(t in i["tags"] for t in need)]
    if "ByStatus" in sel:
        return [i for i in insts if i["status"] in sel["ByStatus"].get("statuses", [])]
    return insts


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj, raw=False):
        b = (obj if raw else json.dumps(obj)).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n).decode() if n else ""
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return raw

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/__state":
            with LOCK:
                return self.send(200, STATE)
        if p == "/healthz":
            return self.send(200, {"status": "ok"})
        if p == "/fv/v1/capabilities":
            return self.send(200, {"models": [{"caps": {"id": "fake-wan"}}]})
        if p.startswith("/fv/v1/jobs/"):
            jid = p.rsplit("/", 1)[1]
            return self.send(200, {"id": jid, "model": "fake-wan", "status": "succeeded"})
        self.send(404, {"error": "no route"})

    def do_POST(self):
        p = self.path.split("?")[0]
        b = self.body()
        with LOCK:
            STATE["requests"].append({"path": p, "key_ok": self.headers.get("X-API-Key") == KEY,
                                      "auth_header": bool(self.headers.get("authorization"))})
        if p == "/__seed":
            with LOCK:
                STATE.update(b or {})
            return self.send(200, {})
        if p == "/fv/v1/jobs":
            jid = "job_" + uuid.uuid4().hex[:8]
            return self.send(201, {"id": jid, "status": "queued"})
        if not p.startswith("/api/v1/"):
            return self.send(404, {"error": "no route"})
        path = p[len("/api/v1/"):]
        if not isinstance(b, dict) or "version" not in b or "data" not in b:
            return self.send(400, "request must be {version, data}", raw=True)
        v, d = b["version"], b["data"]
        if path == "instance-types/list":
            return self.send(200, {"version": "2025-01-29", "data": {"instance_types": catalog()}})
        if self.headers.get("X-API-Key") != KEY:
            return self.send(401, "User cannot be authenticated from the request", raw=True)
        with LOCK:
            if path == "auth/me":
                return self.send(200, {"version": v, "data": {"email": "owner@example.com", "id": "u1", "provider": "CloudRift", "totp_enabled": False}})
            if path == "account/info":
                return self.send(200, {"version": v, "data": {"balance": STATE["balance"]}})
            if path == "instances/rent":
                if v < RENT_MIN_VERSION and v != "~upcoming":
                    return self.send(400, "unsupported version", raw=True)
                sel = d.get("selector", {}).get("ByInstanceTypeAndLocation", {})
                docker = d.get("config", {}).get("Docker")
                if not sel.get("instance_type") or not docker or not d.get("with_public_ip"):
                    return self.send(400, "bad rent request", raw=True)
                iid = str(uuid.uuid4())
                maps = []
                for spec in docker.get("ports", []):
                    host, cont = spec.split("/")[0].split(":")
                    maps.append([int(cont), PORT if int(cont) == 8000 else int(host)])
                price = next((t["price"] for t in STATE["catalog"] if sel["instance_type"].startswith(t["name"])), 100.0)
                STATE["instances"][iid] = {
                    "id": iid, "status": "Initializing", "instance_name": d.get("name"), "tags": d.get("tags", []),
                    "host_address": "127.0.0.1", "port_mappings": maps, "node_id": "n1", "node_mode": "Container",
                    "node_status": "Ready", "containers": [], "virtual_machines": [], "ssh_key_auth": False,
                    "created_at": "2026-10-06T00:00:00Z",
                    "resource_info": {"cost_per_hour": price, "instance_type": sel["instance_type"], "provider_name": "p"},
                    "_payload": d,
                }
                STATE["rents"].append(d)
                return self.send(201, {"version": v, "data": {"instance_ids": [iid]}})
            if path == "instances/list":
                out = []
                for i in select(d.get("selector", {})):
                    if i["status"] == "Initializing":
                        i["status"] = "Active"
                    out.append({k: x for k, x in i.items() if not k.startswith("_")})
                return self.send(200, {"version": v, "data": {"instances": out}})
            if path == "instances/terminate":
                done = []
                for i in select(d.get("selector", {})):
                    if i["status"] != "Inactive":
                        i["status"] = "Inactive"
                        done.append({k: x for k, x in i.items() if not k.startswith("_")})
                        STATE["terminates"].append(i["id"])
                return self.send(201, {"version": v, "data": {"terminated": done}})
            if path == "instances/metrics":
                ids = d.get("selector", {}).get("ById", [])
                return self.send(200, {"version": v, "data": {"metrics": [
                    {"instance_id": i, "node_id": "n1", "gpus": [{"gpu_index": "0", "gpu_utilization_percent": STATE["gpu_util"]}]} for i in ids]}})
        return self.send(404, f"no route {path}", raw=True)


def main():
    global PORT
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    PORT = srv.server_address[1]
    with open(sys.argv[1], "w") as f:
        f.write(str(PORT))
    srv.serve_forever()


if __name__ == "__main__":
    main()
