#!/usr/bin/env python3
"""Minimal S3 (SigV4) client for the R2 build-artifact bucket. Stdlib only, so
it runs unchanged in this container, on the build pod's driver and on GitHub
runners (docs/dev/build-pod.md "Release artifacts").

  r2.py head <key>                exit 0 if the object exists, 1 if not (3: error)
  r2.py get  <key> <file>         download (to <file>.part, then rename)
  r2.py put  <key> <file>         upload (single PUT, payload sha256 signed)
  r2.py ls   <prefix>             keys and sizes under a prefix

Credentials never go through argv. They come from the environment, or from
the KEY=VALUE file named by FV_R2_ARTIFACTS_ENV_FILE (mode 600):

  FV_R2_ARTIFACTS_ENDPOINT           https://<account id>.r2.cloudflarestorage.com
  FV_R2_ARTIFACTS_ACCESS_KEY_ID
  FV_R2_ARTIFACTS_SECRET_ACCESS_KEY
  FV_R2_ARTIFACTS_BUCKET             default fv-build-artifacts

The secret is never printed; errors show the HTTP status and R2's error code.
"""

import datetime
import hashlib
import hmac
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

REGION = "auto"
SERVICE = "s3"
UA = "fv-r2-artifacts/1"


def load_env():
    path = os.environ.get("FV_R2_ARTIFACTS_ENV_FILE")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    cfg = {
        "endpoint": os.environ.get("FV_R2_ARTIFACTS_ENDPOINT", "").rstrip("/"),
        "key": os.environ.get("FV_R2_ARTIFACTS_ACCESS_KEY_ID", ""),
        "secret": os.environ.get("FV_R2_ARTIFACTS_SECRET_ACCESS_KEY", ""),
        "bucket": os.environ.get("FV_R2_ARTIFACTS_BUCKET", "") or "fv-build-artifacts",
    }
    missing = [n for n, k in (("FV_R2_ARTIFACTS_ENDPOINT", "endpoint"), ("FV_R2_ARTIFACTS_ACCESS_KEY_ID", "key"),
                              ("FV_R2_ARTIFACTS_SECRET_ACCESS_KEY", "secret")) if not cfg[k]]
    if missing:
        fail(f"r2: missing {', '.join(missing)} (environment or FV_R2_ARTIFACTS_ENV_FILE)")
    return cfg


def _hmac(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def quote_path(path):
    return urllib.parse.quote(path, safe="/-_.~")


def sign(method, host, path, query, headers, payload_sha, key, secret, now, region=REGION, service=SERVICE):
    """SigV4 for one request. `query` is a dict; `headers` a dict of extra
    headers (lower-case names). Returns the headers to send (incl. Authorization)."""
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date = now.strftime("%Y%m%d")
    hdrs = {k.lower(): str(v).strip() for k, v in headers.items()}
    hdrs["host"] = host
    hdrs["x-amz-date"] = amz_date
    hdrs["x-amz-content-sha256"] = payload_sha
    signed = sorted(hdrs)
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}"
        for k, v in sorted(query.items())
    )
    canonical = "\n".join(
        [method, quote_path(path), canonical_query, "".join(f"{h}:{hdrs[h]}\n" for h in signed), ";".join(signed),
         payload_sha]
    )
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = _hmac(("AWS4" + secret).encode("utf-8"), date)
    k = _hmac(k, region)
    k = _hmac(k, service)
    k = _hmac(k, "aws4_request")
    sig = hmac.new(k, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    hdrs["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={key}/{scope}, SignedHeaders={';'.join(signed)}, Signature={sig}"
    )
    del hdrs["host"]  # urllib sets it
    return hdrs


def request(cfg, method, key, query=None, body=None, payload_sha=None, timeout=900):
    query = query or {}
    u = urllib.parse.urlsplit(cfg["endpoint"])
    path = f"/{cfg['bucket']}/{key}" if key else f"/{cfg['bucket']}"
    if payload_sha is None:
        payload_sha = hashlib.sha256(body or b"").hexdigest()
    now = datetime.datetime.now(datetime.timezone.utc)
    extra = {"content-length": str(os.fstat(body.fileno()).st_size)} if hasattr(body, "fileno") else {}
    hdrs = sign(method, u.netloc, path, query, extra, payload_sha, cfg["key"], cfg["secret"], now)
    hdrs["user-agent"] = UA
    url = f"{u.scheme}://{u.netloc}{quote_path(path)}"
    if query:
        url += "?" + urllib.parse.urlencode(sorted(query.items()), quote_via=urllib.parse.quote)
    req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
    return urllib.request.urlopen(req, timeout=timeout)


def err(e):
    body = e.read()[:400].decode("utf-8", "replace") if hasattr(e, "read") else ""
    code = ""
    try:
        code = ET.fromstring(body).findtext("Code") or ""
    except ET.ParseError:
        pass
    return f"HTTP {getattr(e, 'code', '?')} {code}".strip()


def fail(msg):
    """Errors exit 3, so `head`'s 1 always means "not found"."""
    print(msg, file=sys.stderr)
    sys.exit(3)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_head(cfg, key):
    try:
        with request(cfg, "HEAD", key, timeout=60):
            return 0
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 1
        fail(f"r2: HEAD {key}: {err(e)}")


def cmd_get(cfg, key, dest):
    tmp = dest + ".part"
    try:
        with request(cfg, "GET", key) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if e.code == 404:
            print(f"r2: {key}: not found", file=sys.stderr)
            return 1
        fail(f"r2: GET {key}: {err(e)}")
    os.replace(tmp, dest)
    return 0


def cmd_put(cfg, key, src):
    sha = file_sha256(src)
    try:
        with open(src, "rb") as f, request(cfg, "PUT", key, body=f, payload_sha=sha):
            pass
    except urllib.error.HTTPError as e:
        fail(f"r2: PUT {key}: {err(e)}")
    print(f"r2: put {key} ({os.path.getsize(src)} bytes, sha256 {sha[:16]}…)", file=sys.stderr)
    return 0


def cmd_ls(cfg, prefix):
    token = None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        try:
            with request(cfg, "GET", "", query=q, timeout=60) as r:
                root = ET.fromstring(r.read())
        except urllib.error.HTTPError as e:
            fail(f"r2: LIST {prefix}: {err(e)}")
        for c in root.iter(ns + "Contents"):
            print(f"{c.findtext(ns + 'Key')}\t{c.findtext(ns + 'Size')}\t{c.findtext(ns + 'LastModified')}")
        if root.findtext(ns + "IsTruncated") != "true":
            return 0
        token = root.findtext(ns + "NextContinuationToken")


def main(argv):
    if len(argv) < 2 or argv[0] not in ("head", "get", "put", "ls"):
        print(__doc__, file=sys.stderr)
        return 2
    cfg = load_env()
    cmd, args = argv[0], argv[1:]
    if cmd == "head":
        return cmd_head(cfg, args[0])
    if cmd == "get":
        return cmd_get(cfg, args[0], args[1])
    if cmd == "put":
        return cmd_put(cfg, args[0], args[1])
    return cmd_ls(cfg, args[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
