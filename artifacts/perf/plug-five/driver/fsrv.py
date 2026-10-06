#!/usr/bin/env python3
"""GET /f/<name>: a file from /root/plg/out, with X-Sidecar-Token (the pod's
sidecar token). Nothing else is served."""
import hmac, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
TOKEN = os.environ.get("FV_SIDECAR_TOKEN", "")
ROOT = "/root/plg/out"
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200); self.end_headers(); return
        if not (TOKEN and hmac.compare_digest(self.headers.get("x-sidecar-token", ""), TOKEN)):
            self.send_response(401); self.end_headers(); return
        name = self.path[3:] if self.path.startswith("/f/") else ""
        p = os.path.join(ROOT, os.path.basename(name))
        if not name or not os.path.isfile(p):
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("content-length", str(os.path.getsize(p))); self.end_headers()
        with open(p, "rb") as f:
            while (b := f.read(1 << 20)):
                self.wfile.write(b)
    def log_message(self, *_):
        pass
ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
