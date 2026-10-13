"""A small in-memory stand-in for R2's S3 API (tests and local benchmarks of r2-weights.py / r2-fetch.py).

Path-style requests (/<bucket>/<key>): PUT (If-None-Match: *), HEAD, GET with
Range, DELETE, ListObjectsV2, and multipart uploads. Header-signed requests
must carry an x-amz-content-sha256 that matches the body (as R2 checks) and a
valid SigV4 signature for SECRET; presigned GETs are checked the same way,
expiry included. Faults for the fetcher's tests: corrupt / cut / fail / stall the
next N ranged GETs, or refuse presigned URLs dated before a cut-off (403).
POST /plan answers with `plan_fn()` (a fake fv-control plan endpoint).
"""
import datetime
import hashlib
import hmac
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACCESS = "fake-access-key"
SECRET = "fake-secret"


def _hmac(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _sig(secret, date, region, to_sign):
    k = _hmac(("AWS4" + secret).encode(), date)
    for p in (region, "s3", "aws4_request"):
        k = _hmac(k, p)
    return hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()


def _cq(pairs):
    return "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}" for k, v in sorted(pairs))


class State:
    def __init__(self):
        self.objects = {}  # (bucket, key) -> (bytes, meta)
        self.uploads = {}  # id -> (bucket, key, meta, {n: bytes})
        self.lock = threading.Lock()
        self.faults = {"corrupt": 0, "cut": 0, "err500": 0, "stall": 0}
        self.stall_skip = 0  # ranged GETs served normally before a "stall" fault may hit one
        self.stall_s = 20.0  # how long a stalled GET trickles before it gives up
        self.get_delay_s = 0.0  # every ranged GET of a tree object waits this long first (a slow fetch)
        self.die_after = None  # after this many ranged GETs of tree objects, every one is a 404 (the fetch dies)
        self.reject_before = None  # presigned URLs with X-Amz-Date before this: 403
        self.gets = []  # (key, range)
        self.plan_fn = None
        self.plan_calls = 0
        self.bad_auth = 0
        self.n = 0

    def take(self, fault):
        with self.lock:
            if self.faults.get(fault, 0) > 0:
                self.faults[fault] -= 1
                return True
        return False


def make_handler(st):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def reply(self, code, body=b"", headers=None):
            self.send_response(code)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def err(self, code, s3code, msg=""):
            self.reply(code, f"<Error><Code>{s3code}</Code><Message>{msg}</Message></Error>".encode(), {"content-type": "application/xml"})

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def auth_ok(self, u, q, body):
            raw_path = u.path
            if "X-Amz-Signature" in q:
                amz_date = q["X-Amz-Date"]
                when = datetime.datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc)
                if datetime.datetime.now(datetime.timezone.utc) > when + datetime.timedelta(seconds=int(q["X-Amz-Expires"])):
                    return False
                if st.reject_before and when < st.reject_before:
                    return False
                cred = q["X-Amz-Credential"].split("/")
                pairs = [(k, v) for k, v in urllib.parse.parse_qsl(u.query, keep_blank_values=True) if k != "X-Amz-Signature"]
                canonical = "\n".join([self.command, raw_path, _cq(pairs), f"host:{self.headers['Host']}\n", "host", "UNSIGNED-PAYLOAD"])
                scope = "/".join(cred[1:])
                to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
                return cred[0] == ACCESS and hmac.compare_digest(_sig(SECRET, cred[1], cred[2], to_sign), q["X-Amz-Signature"])
            a = self.headers.get("Authorization") or ""
            m = re.match(r"AWS4-HMAC-SHA256 Credential=([^/]+)/(\d{8})/([^/]+)/s3/aws4_request, SignedHeaders=([^,]+), Signature=([0-9a-f]+)", a)
            if not m or m.group(1) != ACCESS:
                return False
            psha = self.headers.get("x-amz-content-sha256") or ""
            if psha != "UNSIGNED-PAYLOAD" and psha != hashlib.sha256(body).hexdigest():
                return "sha"
            signed = m.group(4).split(";")
            canon_h = "".join(f"{h}:{(self.headers.get(h) or '').strip()}\n" for h in signed)
            pairs = urllib.parse.parse_qsl(u.query, keep_blank_values=True)
            canonical = "\n".join([self.command, raw_path, _cq(pairs), canon_h, ";".join(signed), psha])
            amz_date = self.headers.get("x-amz-date")
            to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, f"{m.group(2)}/{m.group(3)}/s3/aws4_request", hashlib.sha256(canonical.encode()).hexdigest()])
            return hmac.compare_digest(_sig(SECRET, m.group(2), m.group(3), to_sign), m.group(5))

        def handle_any(self):
            u = urllib.parse.urlsplit(self.path)
            q = dict(urllib.parse.parse_qsl(u.query, keep_blank_values=True))
            body = self.body() if self.command in ("PUT", "POST") else b""
            if u.path == "/plan" and self.command == "POST":
                st.plan_calls += 1
                return self.reply(200, json.dumps(st.plan_fn()).encode(), {"content-type": "application/json"})
            ok = self.auth_ok(u, q, body)
            if ok == "sha":
                return self.err(400, "XAmzContentSHA256Mismatch", "the body does not match x-amz-content-sha256")
            if not ok:
                st.bad_auth += 1
                return self.err(403, "AccessDenied", "bad signature or expired")
            parts = urllib.parse.unquote(u.path).lstrip("/").split("/", 1)
            bucket, key = parts[0], (parts[1] if len(parts) > 1 else "")
            if not key and self.command == "GET":
                return self.list(bucket, q)
            if self.command == "PUT" and "uploadId" in q:
                with st.lock:
                    up = st.uploads.get(q["uploadId"])
                    if not up:
                        return self.err(404, "NoSuchUpload")
                    up[3][int(q["partNumber"])] = body
                return self.reply(200, b"", {"ETag": '"%s"' % hashlib.md5(body).hexdigest()})
            if self.command == "POST" and "uploads" in q:
                with st.lock:
                    st.n += 1
                    uid = f"up{st.n}"
                    meta = {k[11:].lower(): v for k, v in self.headers.items() if k.lower().startswith("x-amz-meta-")}
                    st.uploads[uid] = (bucket, key, meta, {})
                return self.reply(200, f"<InitiateMultipartUploadResult><UploadId>{uid}</UploadId></InitiateMultipartUploadResult>".encode())
            if self.command == "POST" and "uploadId" in q:
                with st.lock:
                    up = st.uploads.pop(q["uploadId"], None)
                    if not up:
                        return self.err(404, "NoSuchUpload")
                    nums = [int(x) for x in re.findall(r"<PartNumber>(\d+)</PartNumber>", body.decode())]
                    st.objects[(bucket, key)] = (b"".join(up[3][n] for n in nums), up[2])
                return self.reply(200, b"<CompleteMultipartUploadResult/>")
            if self.command == "DELETE" and "uploadId" in q:
                with st.lock:
                    st.uploads.pop(q["uploadId"], None)
                return self.reply(204)
            if self.command == "PUT":
                with st.lock:
                    if self.headers.get("If-None-Match") == "*" and (bucket, key) in st.objects:
                        return self.err(412, "PreconditionFailed")
                    meta = {k[11:].lower(): v for k, v in self.headers.items() if k.lower().startswith("x-amz-meta-")}
                    st.objects[(bucket, key)] = (body, meta)
                return self.reply(200, b"", {"ETag": '"x"'})
            if self.command == "DELETE":
                with st.lock:
                    st.objects.pop((bucket, key), None)
                return self.reply(204)
            obj = st.objects.get((bucket, key))
            if obj is None:
                return self.err(404, "NoSuchKey")
            data, meta = obj
            hdrs = {"ETag": '"e"', "Accept-Ranges": "bytes", **{"x-amz-meta-" + k: v for k, v in meta.items()}}
            if self.command == "HEAD":
                self.send_response(200)
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return
            rng = self.headers.get("Range")
            if st.take("err500"):
                return self.err(500, "InternalError")
            if rng:
                a, b = rng.split("=")[1].split("-")
                a, b = int(a), min(int(b), len(data) - 1)
                st.gets.append((key, a))
                chunk = data[a:b + 1]
                if "/weights/" in key:
                    if st.die_after is not None and sum(1 for k, x in st.gets if x is not None and "/weights/" in k) > st.die_after:
                        return self.err(404, "NoSuchKey", "gone")
                    if st.get_delay_s:
                        time.sleep(st.get_delay_s)
                if len(st.gets) > st.stall_skip and "/weights/" in key and st.take("stall"):
                    # a straggler: headers and a few bytes, then (almost) nothing, as R2 does now and then
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    try:
                        self.wfile.write(chunk[:100])
                        self.wfile.flush()
                        t = time.time()
                        while time.time() - t < st.stall_s:
                            time.sleep(0.2)
                            self.wfile.write(chunk[100:101])  # a byte now and then: no socket timeout
                            self.wfile.flush()
                    except OSError:
                        pass
                    self.close_connection = True
                    return
                if st.take("corrupt"):
                    chunk = bytes([chunk[0] ^ 0xFF]) + chunk[1:]
                if st.take("cut"):
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk[: len(chunk) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                return self.reply(206, chunk, dict(hdrs, **{"Content-Range": f"bytes {a}-{b}/{len(data)}"}))
            st.gets.append((key, None))
            return self.reply(200, data, hdrs)

        def list(self, bucket, q):
            prefix = q.get("prefix", "")
            keys = sorted(k for (b, k) in st.objects if b == bucket and k.startswith(prefix))
            xml = "".join(f"<Contents><Key>{k}</Key><Size>{len(st.objects[(bucket, k)][0])}</Size></Contents>" for k in keys)
            self.reply(200, f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>{xml}</ListBucketResult>'.encode())

        do_GET = do_PUT = do_POST = do_HEAD = do_DELETE = handle_any

    return H


def serve():
    st = State()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, st, f"http://127.0.0.1:{srv.server_address[1]}"
