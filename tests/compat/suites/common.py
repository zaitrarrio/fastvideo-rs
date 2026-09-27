"""Shared bits of the Python compat suites (tests/compat/run.sh).

Every suite takes `--base <url> --key <api key>`, exits non-zero on the first
failed check and prints one JSON summary line last.
"""

import argparse
import json
import struct
import sys
import time
import traceback
import zlib


class Suite:
    def __init__(self, name):
        self.name = name
        self.checks = []
        self.info = {}

    def check(self, cond, what):
        if not cond:
            raise AssertionError(what)

    def ok(self, name):
        self.checks.append(name)
        print(f"  ok  {name}", file=sys.stderr, flush=True)

    def run(self, fn):
        ap = argparse.ArgumentParser()
        ap.add_argument("--base", required=True, help="server base URL (https with the compat CA)")
        ap.add_argument("--key", required=True, help="API key")
        ap.add_argument("--hook-host", default="127.0.0.1", help="host the server reaches local receivers on")
        args = ap.parse_args()
        try:
            fn(self, args)
        except Exception as e:  # noqa: BLE001 - one failure report for everything
            traceback.print_exc()
            print(json.dumps({"suite": self.name, "ok": False, "checks": self.checks, "error": f"{type(e).__name__}: {e}"}))
            sys.exit(1)
        print(json.dumps({"suite": self.name, "ok": True, "checks": self.checks, **self.info}))


def poll(fn, done, timeout=120, interval=0.1):
    """Calls `fn` until `done(result)`; returns the last result."""
    t0 = time.monotonic()
    while True:
        r = fn()
        if done(r):
            return r
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"timed out polling; last: {r!r}")
        time.sleep(interval)


def png(width, height, rgb=(200, 40, 40)):
    """A solid-colour RGB PNG."""
    def chunk(t, data):
        c = zlib.crc32(t + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", c)

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def is_mp4(body):
    return len(body) >= 8 and body[4:8] == b"ftyp"


class Receiver:
    """A local HTTP server recording POSTs (webhook / callback receiver).

    `reply(path, headers, body) -> (status, json_or_None)` answers each POST;
    the default answers 200 `{}`. Received requests are in `.got` as
    `(path, headers dict with lower-case names, raw body bytes)`.
    """

    def __init__(self, host="127.0.0.1", reply=None):
        import http.server
        import threading

        self.got = []
        self.cond = threading.Condition()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                n = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(n)
                headers = {k.lower(): v for k, v in self.headers.items()}
                status, out = (reply or (lambda *_: (200, {})))(self.path, headers, body)
                data = json.dumps(out).encode() if out is not None else b""
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                with outer.cond:
                    outer.got.append((self.path, headers, body))
                    outer.cond.notify_all()

            def log_message(self, *_):
                pass

        self.server = http.server.ThreadingHTTPServer((host, 0), H)
        self.url = f"http://{host}:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def wait(self, pred, timeout=60):
        """Waits until `pred(self.got)` holds; returns `self.got`."""
        with self.cond:
            if not self.cond.wait_for(lambda: pred(self.got), timeout):
                raise AssertionError(f"receiver: timed out; got {[(p, b[:200]) for p, _, b in self.got]}")
            return list(self.got)

    def close(self):
        self.server.shutdown()
