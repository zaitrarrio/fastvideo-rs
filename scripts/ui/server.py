#!/usr/bin/env python3
"""Local control panel for renting a GPU, deploying FastVideo and generating a clip.

Runs on the machine that holds the Vast credentials and the ssh key, because
every useful action here shells out: `vastai` to search and rent, ssh/rsync to
deploy, and the gpucheck binary on the far end to generate. A page served from
elsewhere could not do any of it.

Binds to loopback only. The Vast API key stays in .env, is read by the scripts
themselves, and is never sent to the browser.

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
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


def offers(tier: str, gpu: str) -> list[dict]:
    """Cheapest matching offers, parsed out of `validate.sh offers`."""
    env = {"FV_OFFER_QUERY_EXTRA": f"gpu_name={gpu}"} if gpu else {}
    p = subprocess.run(
        ["bash", str(VALIDATE), "offers", tier],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=120,
    )
    out = []
    for line in (p.stdout or "").splitlines():
        m = re.search(
            r"offer (\d+)\s+(.+?)\s+(\d+)GB\s+\$([\d.]+)/hr\s+sm(\d+)\s+rel ([\d.]+)\s+(\d+)Mbps\s+(.+?)\s+worst-case",
            line,
        )
        if m:
            out.append({
                "id": m.group(1), "gpu": m.group(2).strip(), "vram": int(m.group(3)),
                "dph": float(m.group(4)), "sm": m.group(5), "reliability": float(m.group(6)),
                "inet": int(m.group(7)), "where": m.group(8).strip(),
            })
    return out


def instances() -> list[dict]:
    rc, out = run(["vastai", "show", "instances", "--raw"])
    if rc != 0:
        return []
    try:
        raw = json.loads(out)
    except json.JSONDecodeError:
        return []
    return [{
        "id": i.get("id"),
        "gpu": i.get("gpu_name"),
        "status": i.get("actual_status"),
        "dph": i.get("dph_total"),
        "label": i.get("label"),
    } for i in raw]


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
            rc, out = run(["vastai", "destroy", "instance", "-y", str(body.get("id"))])
            self._json({"ok": rc == 0, "output": SECRET.sub("<redacted>", out)})
        elif u.path == "/api/stop":
            self._json({"stopped": JOB.stop()})
        else:
            self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8733)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fastvideo control panel → http://127.0.0.1:{args.port}")
    print("loopback only; the Vast key stays in .env and never reaches the page")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
