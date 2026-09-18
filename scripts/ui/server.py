#!/usr/bin/env python3
"""Local control panel for renting a GPU, deploying FastVideo and generating a clip.

Searching, listing and destroying go straight to Vast's REST API. Deploying and
generating shell out to scripts/gpu/validate.sh, because those need ssh, rsync
and the gpucheck binary on the far end — which is also why this has to run on
the machine holding the credentials and the ssh key rather than in a browser.

Binds to loopback only. The API key is read from .env into this process, sent
only to Vast, and scrubbed from anything the page receives.

    python3 scripts/ui/server.py [--port 8733]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
# Vast's REST API. Verified against the live service: /api/v0/instances/ answers
# 410 Gone (the published docs still list it), so listing goes through v1, while
# bundles, asks and delete are still v0.
VAST_BASE = os.environ.get("VAST_API_BASE", "https://console.vast.ai")
API_OFFERS = "/api/v0/bundles/"
API_INSTANCES = "/api/v1/instances/"
API_RENT = "/api/v0/asks/{offer}/"
API_INSTANCE = "/api/v0/instances/{id}/"
VALIDATE = ROOT / "scripts" / "gpu" / "validate.sh"
CLIPS = ROOT / "artifacts" / "clips"
UI_DIR = Path(__file__).resolve().parent
# Anything that smells like a credential never reaches the browser.
SECRET = re.compile(r"(api[_-]?key|token|secret|password)\s*[=:]\s*\S+", re.I)


class Job:
    """One background command, with its output tailed by the browser.

    Only one runs at a time: every action here either rents, deploys to, or
    generates on a single instance, and overlapping them would interleave ssh
    sessions on the same box.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lines: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.label = ""
        self.started = 0.0
        self.rc: int | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, label: str, cmd: list[str], env: dict[str, str]) -> tuple[bool, str]:
        with self.lock:
            if self.running():
                return False, f"{self.label} is still running"
            self.lines = [f"$ {' '.join(shlex.quote(c) for c in cmd)}"]
            self.label, self.started, self.rc = label, time.time(), None
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        threading.Thread(target=self._pump, daemon=True).start()
        return True, label

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            clean = SECRET.sub(lambda m: m.group(0).split("=")[0] + "=<redacted>", line.rstrip())
            with self.lock:
                self.lines.append(clean)
                # Keep the tail bounded; a clip run emits thousands of lines.
                if len(self.lines) > 4000:
                    del self.lines[:1000]
        self.proc.wait()
        with self.lock:
            self.rc = self.proc.returncode

    def snapshot(self, since: int) -> dict:
        with self.lock:
            return {
                "label": self.label,
                "running": self.running(),
                "rc": self.rc,
                "elapsed": round(time.time() - self.started, 1) if self.started else 0,
                "from": since,
                "lines": self.lines[since:],
                "total": len(self.lines),
            }

    def stop(self) -> bool:
        with self.lock:
            if self.running() and self.proc:
                self.proc.terminate()
                return True
        return False


JOB = Job()


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def api_key() -> str:
    """The key from the environment or .env. It never leaves this process."""
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("VAST_API_KEY"):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("VAST_API_KEY is not set: add it to .env (chmod 600)")


def vast(method: str, path: str, params: dict | None = None, body: dict | None = None, timeout: int = 60):
    url = VAST_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {api_key()}",
        "Accept": "application/json",
        # Vast rejects the stock urllib agent on some routes.
        "User-Agent": "fastvideo-rs-ui/1",
        **({"Content-Type": "application/json"} if data else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def offers(tier: str, gpu: str) -> list[dict]:
    """Cheapest matching offers, straight from the bundles endpoint.

    Mirrors the filters `validate.sh tier_query` uses, so what the page shows is
    what a run would actually rent.
    """
    need = {"gen": 24, "clip": 24, "compare": 24, "parity": 16, "kernels": 8}.get(tier, 24)
    q = {
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": need * 1000},
        "compute_cap": {"gte": 800},
        "cuda_max_good": {"gte": 12.4},
        "reliability2": {"gt": 0.97},
        "rentable": {"eq": True},
        "verified": {"eq": True},
        "direct_port_count": {"gte": 1},
        "inet_down": {"gte": 200},
        "disk_space": {"gte": 100},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 24,
    }
    if gpu:
        q["gpu_name"] = {"eq": gpu.replace("_", " ")}
    try:
        body = vast("GET", API_OFFERS, {"q": json.dumps(q)})
    except Exception as e:  # noqa: BLE001 - surfaced in the page, not swallowed
        return [{"error": f"{type(e).__name__}: {e}"}]
    out = []
    for o in body.get("offers", []):
        out.append({
            "id": o.get("id"),
            "machine": o.get("machine_id"),
            "gpu": o.get("gpu_name"),
            "vram": round((o.get("gpu_ram") or 0) / 1024),
            "dph": round(o.get("dph_total") or 0, 4),
            "sm": o.get("compute_cap"),
            "reliability": round(o.get("reliability2") or 0, 3),
            "inet": round(o.get("inet_down") or 0),
            "where": o.get("geolocation") or "",
            "cuda": o.get("cuda_max_good"),
        })
    return out


def instances() -> list[dict]:
    try:
        body = vast("GET", API_INSTANCES)
    except Exception as e:  # noqa: BLE001
        return [{"error": f"{type(e).__name__}: {e}"}]
    raw = body.get("instances", body.get("instances_found", [])) if isinstance(body, dict) else []
    if isinstance(raw, int):  # v1 returns a count under that name when empty
        raw = body.get("instances", [])
    out = []
    for i in raw or []:
        out.append({
            "id": i.get("id"),
            "gpu": i.get("gpu_name"),
            "status": i.get("actual_status") or i.get("cur_state"),
            "dph": i.get("dph_total"),
            "label": i.get("label"),
            "ssh": f"{i.get('ssh_host')}:{i.get('ssh_port')}" if i.get("ssh_host") else "",
        })
    return out


def clips() -> list[dict]:
    out = []
    if CLIPS.is_dir():
        for mp4 in sorted(CLIPS.glob("*/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            out.append({
                "name": f"{mp4.parent.name}/{mp4.name}",
                "size_mb": round(mp4.stat().st_size / 1e6, 2),
                "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(mp4.stat().st_mtime)),
            })
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # quiet; the job log is what matters
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/offers":
            self._json(offers(q.get("tier", ["gen"])[0], q.get("gpu", [""])[0]))
        elif u.path == "/api/instances":
            self._json(instances())
        elif u.path == "/api/clips":
            self._json(clips())
        elif u.path == "/api/job":
            self._json(JOB.snapshot(int(q.get("since", ["0"])[0])))
        elif u.path.startswith("/clip/"):
            rel = u.path[len("/clip/"):]
            path = (CLIPS / rel).resolve()
            # Never serve outside the clips directory.
            if not str(path).startswith(str(CLIPS.resolve())) or not path.is_file():
                self._json({"error": "not found"}, 404)
                return
            self._send(200, path.read_bytes(), "video/mp4")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or "{}") if length else {}
        if u.path == "/api/generate":
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                self._json({"error": "prompt is empty"}, 400)
                return
            env = {
                "FV_PROMPT": prompt,
                "FV_FRAMES": str(body.get("frames", 129)),
                "FV_STEPS": str(body.get("steps", 3)),
                "FV_HEIGHT": str(body.get("height", 448)),
                "FV_WIDTH": str(body.get("width", 832)),
                "FV_VSA": "1" if body.get("vsa", True) else "0",
                "FV_VAE_CHUNK": str(body.get("vae_chunk", 2)),
            }
            if body.get("gpu"):
                env["FV_OFFER_QUERY_EXTRA"] = f"gpu_name={body['gpu']}"
            cmd = ["bash", str(VALIDATE), "run", "gen"]
            # Reusing an instance skips a five-minute deploy, but a box that has
            # already generated accumulates GPU memory and later runs OOM on it.
            if body.get("instance"):
                cmd += ["--instance", str(body["instance"])]
            if body.get("keep"):
                cmd.append("--keep")
            ok, msg = JOB.start(f"generate: {prompt[:60]}", cmd, env)
            self._json({"started": ok, "message": msg}, 200 if ok else 409)
        elif u.path == "/api/destroy":
            try:
                res = vast("DELETE", API_INSTANCE.format(id=int(body.get("id"))))
                self._json({"ok": bool(res.get("success", True)), "output": json.dumps(res)[:300]})
            except Exception as e:  # noqa: BLE001
                self._json({"ok": False, "output": f"{type(e).__name__}: {e}"}, 502)
        elif u.path == "/api/stop":
            self._json({"stopped": JOB.stop()})
        else:
            self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8733)
    ap.add_argument("--list-instances", action="store_true",
                    help="print running instances and exit (no server)")
    args = ap.parse_args()
    if args.list_instances:
        rows = instances()
        if not rows:
            print("no running instances")
        for i in rows:
            if "error" in i:
                print(f"error: {i['error']}")
            else:
                print(f"{i['id']}\t{i['gpu']}\t{i['status']}\t${i['dph']}/hr\t{i['label'] or ''}")
        return
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fastvideo control panel → http://127.0.0.1:{args.port}")
    print("loopback only; the Vast key stays in .env and never reaches the page")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
