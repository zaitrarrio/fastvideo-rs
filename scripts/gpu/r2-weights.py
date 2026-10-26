#!/usr/bin/env python3
"""The R2 weights store: publish, verify, list and hand out weight trees (docs/ops/r2-weights.md).

Layout in the bucket (default `fv-weights`):

  weights/<dest>/<rev>/<path>        the tree's files, byte for byte
  parts/<dest>/<rev>/<path>.json     per-file record written right after the file: size, sha256, per-part sha256
  manifests/<dest>/<rev>.json        the tree's manifest, written LAST: the tree is published once it exists
  verified/<dest>/<rev>/<stamp>.json a verification record (`verify --deep --record`)

Add-only, like the weight volumes: nothing here overwrites or deletes an
object. A file is uploaded as a multipart upload (invisible until completed)
or a single PUT with `If-None-Match: *`; its SHA-256 is computed while it
streams and checked against what is known (Hub LFS SHA-256, the git blob
SHA-1 of a small Hub file, scripts/gpu/weights-sha256.tsv) BEFORE the upload
is completed; a mismatch aborts it. R2 also checks every part against the
SHA-256 in its signature (x-amz-content-sha256). An object that already
exists is reused only when its size and recorded SHA-256 match, else the run
stops: it never replaces it. The manifest is written with `If-None-Match: *`.

Sources:
  publish-hub DEST     streamed from the Hub at the pinned revision (weights-revisions.tsv,
                       globs from weights-manifest.tsv): ranged GETs from the Hub CDN feed the
                       multipart upload part by part; no local copy. HF_TOKEN for gated repos.
  publish-dir DEST DIR a local directory (the EU volume mounted on a CPU pod), e.g. the derived
                       trees that have no Hub source. --rev is required unless DEST has a Hub row.

Other commands: status (every known tree: published or not), list, show DEST,
verify [DEST...] [--deep] [--record], compare-dir DEST DIR (the volume copy
against the manifest), plan DEST... (presigned plan for r2-fetch.py),
create-bucket, record (weights-r2.tsv rows of the published manifests).

Credentials (never printed): FV_WEIGHTS_R2_ACCESS_KEY_ID / FV_WEIGHTS_R2_SECRET_ACCESS_KEY,
else S3 credentials derived from a Cloudflare API token with R2 rights
(--cf-token-file, default /root/.config/fv/cf_api_token: access key = the
token's id, secret = SHA-256 of the token; developers.cloudflare.com/r2/api/tokens).
Endpoint FV_WEIGHTS_R2_ENDPOINT (default https://<FV_CF_ACCOUNT_ID>.r2.cloudflarestorage.com),
bucket FV_WEIGHTS_R2_BUCKET (default fv-weights), --bucket overrides.
"""
import argparse
import concurrent.futures
import datetime
import fnmatch
import hashlib
import hmac
import importlib.util
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r2_fetch", HERE / "r2-fetch.py")
r2_fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2_fetch)

MANIFEST_FORMAT = "fv-weights-manifest/1"
PART_SIZE = 64 << 20
DEFAULT_ACCOUNT = "0cd06fd37ee4e07de370821cb3852a8a"  # control/wrangler.toml [env.staging.vars] CF_ACCOUNT_ID
DEFAULT_BUCKET = "fv-weights"
HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        sys.stderr.write(time.strftime("%H:%M:%S ") + msg + "\n")
        sys.stderr.flush()


class Fail(Exception):
    pass


# --------------------------------------------------------------------------- repo data files
def read_tsv(name):
    rows = []
    with open(HERE / name) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    return rows


def hub_rows():
    """dest -> (repo, revision, globs) for every plain Hub tree (weights-manifest.tsv + weights-revisions.tsv)."""
    revs = {}
    for r in read_tsv("weights-revisions.tsv"):
        if len(r) >= 5 and r[4] == "hub":
            revs[r[0]] = (r[1], r[2])
    out = {}
    for r in read_tsv("weights-manifest.tsv"):
        if len(r) < 3 or r[1].startswith("url:"):
            continue
        if r[0] in revs and revs[r[0]][0] == r[1]:
            out[r[0]] = (r[1], revs[r[0]][1], r[2].split())
    return out


def _split_dest(full):
    """auxiliary/tae/taeh3.safetensors -> (auxiliary/tae, taeh3.safetensors): the longest known dest prefix."""
    dests = sorted([d for d, _ in VOLUME_ONLY] + list(hub_rows()), key=len, reverse=True)
    for d in dests:
        if full.startswith(d + "/"):
            return d, full[len(d) + 1:]
    return None, None


def recorded_hashes():
    """(dest, path) -> (bytes or None, algo, [hex alternatives]) recorded in the repo: weights-sha256.tsv, the
    auxiliary/ url rows of weights-manifest.tsv, and verify-weights.sh's FP8_TREES / DIT_PREQUANT_TREES."""
    out = {}
    for r in read_tsv("weights-sha256.tsv"):
        if len(r) < 4 or ":" not in r[3]:
            continue
        algo, hexes = r[3].split(":", 1)
        out[(r[0], r[1])] = (None if r[2] == "-" else int(r[2]), algo, [h.lower() for h in hexes.split("|")])
    for r in read_tsv("weights-manifest.tsv"):
        if len(r) >= 3 and r[1].startswith("url:"):
            sha = re.search(r"sha256:([0-9a-f]{64})", " ".join(r[2:]))
            size = re.search(r"size:(\d+)", " ".join(r[2:]))
            dest, path = _split_dest(r[0])
            if sha and dest:
                out.setdefault((dest, path), (int(size.group(1)) if size else None, "sha256", [sha.group(1)]))
    vw = HERE / "verify-weights.sh"
    if vw.is_file():
        for m in re.finditer(r"^(?:[A-Z_]+=\")?([\w./-]+)\t(\d+)\t([0-9a-f]{64})\"?$", vw.read_text(), re.M):
            dest, path = _split_dest(m.group(1))
            if dest:
                out.setdefault((dest, path), (int(m.group(2)), "sha256", [m.group(3)]))
    return out


# Trees that exist only on the EU volume (no plain Hub source): docs/ops/r2-weights.md §6. Published with publish-dir.
VOLUME_ONLY = [
    ("h3-base/text_encoder_fp8", "derived (fv-gpucheck quantize-text-encoder), verify-weights.sh text-fp8"),
    ("ltx25/text_encoder_fp8", "derived (fv-gpucheck quantize-text-encoder), verify-weights.sh text-fp8"),
    ("h3-base/transformer_prequant_4step-vsa_mxfp8", "derived (fv-gpucheck quantize-dit), verify-weights.sh dit-prequant"),
    ("h3-base/transformer_prequant_sol-h3_mxfp8", "derived (fv-gpucheck quantize-dit), verify-weights.sh dit-prequant"),
    ("longlive-1.3b-safetensors", "derived (convert-longlive.py); per-volume metadata order"),
    ("mmaudio-44k-v2", "three repos at main + converted safetensors/ (fetch-mmaudio.py); DFN5B revision UNKNOWN"),
    ("h3-ref2va", "two repos merged (fetch-h3-ref2va.py; Hub-reproducible, copied from the volume for byte identity)"),
    ("auxiliary/tae", "pinned URLs (weights-manifest.tsv aux rows)"),
    ("auxiliary/lpips", "pinned URLs (weights-manifest.tsv aux rows)"),
    ("qwen3-4b-instruct-gguf", "converted to GGUF Q8_0 (llama.cpp b11195) from Qwen/Qwen3-4B-Instruct-2507 @ cdbee75; R2 only, no volume; superseded by qwen3-vl-4b-instruct-gguf"),
]


# --------------------------------------------------------------------------- S3 (SigV4, stdlib)
def _hmac(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def quote_key(k):
    return urllib.parse.quote(k, safe="/-_.~")


def _signing_key(secret, date, region="auto", service="s3"):
    k = _hmac(("AWS4" + secret).encode("utf-8"), date)
    for part in (region, service, "aws4_request"):
        k = _hmac(k, part)
    return k


def sign_headers(method, host, path, query, headers, payload_sha, key, secret, when, region="auto"):
    """SigV4 header auth for one request (path already URI-encoded); returns the headers to send."""
    amz_date = when.strftime("%Y%m%dT%H%M%SZ")
    date = amz_date[:8]
    hdrs = {k.lower(): str(v).strip() for k, v in headers.items()}
    hdrs.update({"host": host, "x-amz-date": amz_date, "x-amz-content-sha256": payload_sha})
    signed = sorted(hdrs)
    cq = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}" for k, v in sorted(query.items()))
    canonical = "\n".join([method, path, cq, "".join(f"{h}:{hdrs[h]}\n" for h in signed), ";".join(signed), payload_sha])
    scope = f"{date}/{region}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    sig = hmac.new(_signing_key(secret, date, region), to_sign.encode(), hashlib.sha256).hexdigest()
    hdrs["authorization"] = f"AWS4-HMAC-SHA256 Credential={key}/{scope}, SignedHeaders={';'.join(signed)}, Signature={sig}"
    del hdrs["host"]
    return hdrs


def presign(endpoint, bucket, key, access, secret, expires, when=None):
    """A presigned GET URL (path style, query auth, UNSIGNED-PAYLOAD), as fv-control's r2presign.ts makes."""
    u = urllib.parse.urlsplit(endpoint)
    return presign_path(u.scheme, u.netloc, "/" + quote_key(f"{bucket}/{key}"), access, secret, expires, when)


def presign_path(scheme, host, path, access, secret, expires, when=None, region="auto"):
    when = when or datetime.datetime.now(datetime.timezone.utc)
    amz_date = when.strftime("%Y%m%dT%H%M%SZ")
    date = amz_date[:8]
    scope = f"{date}/{region}/s3/aws4_request"
    q = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{access}/{scope}", "X-Amz-Date": amz_date, "X-Amz-Expires": str(int(expires)), "X-Amz-SignedHeaders": "host"}
    cq = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}" for k, v in sorted(q.items()))
    canonical = "\n".join(["GET", path, cq, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"])
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    sig = hmac.new(_signing_key(secret, date, region), to_sign.encode(), hashlib.sha256).hexdigest()
    return f"{scheme}://{host}{path}?{cq}&X-Amz-Signature={sig}"


class S3Error(Fail):
    def __init__(self, status, code, msg):
        super().__init__(f"HTTP {status} {code}: {msg}")
        self.status = status
        self.code = code


class S3:
    def __init__(self, endpoint, bucket, access, secret):
        self.endpoint = endpoint.rstrip("/")
        self.u = urllib.parse.urlsplit(self.endpoint)
        self.bucket = bucket
        self.access = access
        self.secret = secret
        self.conns = r2_fetch.Conns(timeout=float(os.environ.get("FV_R2_TIMEOUT_S", "120")))
        self.ops = {"A": 0, "B": 0}
        self.oplock = threading.Lock()

    def _count(self, cls):
        with self.oplock:
            self.ops[cls] += 1

    def request(self, method, key="", query=None, headers=None, body=b"", payload_sha=None, bucket_level=False, ok=(200,), retries=5):
        path = "/" + quote_key(self.bucket) + ("" if bucket_level else "/" + quote_key(key))
        query = query or {}
        body = body if body is not None else b""
        psha = payload_sha or hashlib.sha256(body).hexdigest()
        last = None
        for attempt in range(retries + 1):
            hdrs = sign_headers(method, self.u.netloc, path, query, dict(headers or {}, **({"content-length": str(len(body))} if method in ("PUT", "POST") else {})), psha, self.access, self.secret, datetime.datetime.now(datetime.timezone.utc))
            qs = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}" for k, v in sorted(query.items()))
            url = f"{self.endpoint}{path}" + (f"?{qs}" if qs else "")
            try:
                r = self.conns.request(method, url, hdrs, body if method in ("PUT", "POST") else None)
                data = r.read() if method != "HEAD" else b""
                r.close()
            except (OSError, r2_fetch.http.client.HTTPException) as e:
                last = Fail(f"{method} {key}: {e}")
                time.sleep(min(30, 2 ** attempt))
                continue
            self._count("B" if method in ("GET", "HEAD") else "A")
            if r.status in ok:
                return r, data
            code, msg = "", data[:300].decode("utf-8", "replace")
            try:
                x = ElementTree.fromstring(data)
                code = x.findtext("Code") or ""
                msg = x.findtext("Message") or msg
            except ElementTree.ParseError:
                pass
            err = S3Error(r.status, code, f"{method} {self.bucket}/{key}: {msg}")
            if r.status >= 500 or r.status == 429:
                last = err
                time.sleep(min(30, 2 ** attempt))
                continue
            raise err
        raise last

    def head(self, key):
        """(size, metadata dict) or None."""
        try:
            r, _ = self.request("HEAD", key)
        except S3Error as e:
            if e.status == 404:
                return None
            raise
        meta = {k[len("x-amz-meta-"):].lower(): v for k, v in r.getheaders() if k.lower().startswith("x-amz-meta-")}
        return int(r.getheader("Content-Length") or 0), meta

    def get(self, key, rng=None):
        try:
            _, data = self.request("GET", key, headers={"range": rng} if rng else None, ok=(200, 206))
            return data
        except S3Error as e:
            if e.status == 404:
                return None
            raise

    def put_new(self, key, body, meta=None, content_type="application/octet-stream"):
        """PUT only if absent (If-None-Match: *). True when written, False when it already existed (412)."""
        h = {"if-none-match": "*", "content-type": content_type}
        for k, v in (meta or {}).items():
            h["x-amz-meta-" + k] = v
        try:
            self.request("PUT", key, headers=h, body=body)
            return True
        except S3Error as e:
            if e.status == 412:
                return False
            raise

    def list(self, prefix, delimiter=None):
        token = None
        while True:
            q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["continuation-token"] = token
            _, data = self.request("GET", query=q, bucket_level=True)
            x = ElementTree.fromstring(data)
            ns = x.tag.split("}")[0] + "}" if x.tag.startswith("{") else ""
            for c in x.findall(f"{ns}Contents"):
                yield c.findtext(f"{ns}Key"), int(c.findtext(f"{ns}Size") or 0)
            for c in x.findall(f"{ns}CommonPrefixes"):
                yield c.findtext(f"{ns}Prefix"), -1
            if (x.findtext(f"{ns}IsTruncated") or "false") != "true":
                return
            token = x.findtext(f"{ns}NextContinuationToken")

    # multipart
    def mp_create(self, key, meta):
        h = {"content-type": "application/octet-stream"}
        for k, v in (meta or {}).items():
            h["x-amz-meta-" + k] = v
        _, data = self.request("POST", key, query={"uploads": ""}, headers=h)
        x = ElementTree.fromstring(data)
        ns = x.tag.split("}")[0] + "}" if x.tag.startswith("{") else ""
        return x.findtext(f"{ns}UploadId")

    def mp_part(self, key, upload_id, n, body, sha):
        r, _ = self.request("PUT", key, query={"partNumber": str(n), "uploadId": upload_id}, body=body, payload_sha=sha)
        return r.getheader("ETag")

    def mp_complete(self, key, upload_id, etags):
        xml = "<CompleteMultipartUpload>" + "".join(f"<Part><PartNumber>{i + 1}</PartNumber><ETag>{e}</ETag></Part>" for i, e in enumerate(etags)) + "</CompleteMultipartUpload>"
        self.request("POST", key, query={"uploadId": upload_id}, body=xml.encode())

    def mp_abort(self, key, upload_id):
        try:
            self.request("DELETE", key, query={"uploadId": upload_id}, ok=(200, 204), retries=2)
        except Fail as e:
            log(f"abort {key}: {e} (R2 expires incomplete uploads after 7 days)")


def credentials(args):
    access = os.environ.get("FV_WEIGHTS_R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("FV_WEIGHTS_R2_SECRET_ACCESS_KEY", "")
    if not (access and secret):
        f = Path(args.cf_token_file)
        if not f.is_file():
            raise Fail("no R2 credentials: set FV_WEIGHTS_R2_ACCESS_KEY_ID / FV_WEIGHTS_R2_SECRET_ACCESS_KEY or --cf-token-file")
        tok = f.read_text().strip()
        req = urllib.request.Request("https://api.cloudflare.com/client/v4/user/tokens/verify", headers={"authorization": "Bearer " + tok})
        with urllib.request.urlopen(req, timeout=30) as r:
            access = json.load(r)["result"]["id"]
        secret = hashlib.sha256(tok.encode()).hexdigest()
    acct = os.environ.get("FV_CF_ACCOUNT_ID", DEFAULT_ACCOUNT)
    endpoint = os.environ.get("FV_WEIGHTS_R2_ENDPOINT") or f"https://{acct}.r2.cloudflarestorage.com"
    bucket = args.bucket or os.environ.get("FV_WEIGHTS_R2_BUCKET") or DEFAULT_BUCKET
    return S3(endpoint, bucket, access, secret)


# --------------------------------------------------------------------------- keys
def obj_key(prefix, dest, rev, path):
    return f"{prefix}weights/{dest}/{rev}/{path}"


def parts_key(prefix, dest, rev, path):
    return f"{prefix}parts/{dest}/{rev}/{path}.json"


def manifest_key(prefix, dest, rev):
    return f"{prefix}manifests/{dest}/{rev}.json"


def check_name(dest, rev):
    r2_fetch.safe_rel(dest)
    if not rev or "/" in rev or rev.startswith(".") or len(rev) > 100:
        raise Fail(f"bad revision {rev!r}")


# --------------------------------------------------------------------------- sources
class Source:
    """One file to publish: path, size, read(offset, length) -> bytes, and what its bytes must hash to."""

    def __init__(self, path, size, reader, sha256=None, sha1_blob=None, recorded=None):
        self.path = path
        self.size = size
        self.read = reader
        self.sha256 = sha256
        self.sha1_blob = sha1_blob
        self.recorded = recorded  # (algo, [hex...]) from weights-sha256.tsv


def hub_headers():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not tok:
        p = Path(os.path.expanduser("~/.cache/huggingface/token"))
        tok = p.read_text().strip() if p.is_file() else ""
    return {"authorization": "Bearer " + tok} if tok else {}


def hub_list(repo, rev):
    """Every file of repo@rev: (path, size, lfs sha256 or None, git oid)."""
    out = []
    url = f"{HUB}/api/models/{repo}/tree/{rev}?recursive=1&expand=0"
    while url:
        req = urllib.request.Request(url, headers=hub_headers())
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r)
            link = r.headers.get("Link") or ""
        for it in items:
            if it.get("type") != "file":
                continue
            lfs = it.get("lfs") or {}
            out.append((it["path"], int(it.get("size", lfs.get("size", 0))), (lfs.get("oid") or "").removeprefix("sha256:") or None, it.get("oid")))
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return out


class HubReader:
    """Ranged reads of one Hub file: the resolve URL's redirect target (the CDN, which serves Range) is reused."""

    def __init__(self, conns, repo, rev, path):
        self.conns = conns
        self.url = f"{HUB}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
        self.target = None
        self.lock = threading.Lock()

    def resolve(self):
        with self.lock:
            if self.target and time.time() - self.target[1] < 600:
                return self.target[0]
            url = self.url
            for _ in range(5):
                r = self.conns.request("HEAD", url, hub_headers() if urllib.parse.urlsplit(url).netloc == urllib.parse.urlsplit(HUB).netloc else {})
                r.read()
                if r.status in (301, 302, 303, 307, 308):
                    url = urllib.parse.urljoin(url, r.getheader("Location"))
                    continue
                if r.status != 200:
                    raise Fail(f"Hub {self.url}: HTTP {r.status}")
                break
            self.target = (url, time.time())
            return url

    def __call__(self, off, n):
        for attempt in range(6):
            url = self.resolve()
            same = urllib.parse.urlsplit(url).netloc == urllib.parse.urlsplit(HUB).netloc
            try:
                r = self.conns.request("GET", url, dict({"Range": f"bytes={off}-{off + n - 1}"}, **(hub_headers() if same else {})))
                data = r.read()
                if r.status == 206 and len(data) == n:
                    return data
                if r.status == 200 and off == 0 and len(data) == n:
                    return data
                if r.status in (403, 410):
                    self.target = None  # the CDN URL expired: resolve again
                err = f"HTTP {r.status}, {len(data)} of {n} bytes"
            except (OSError, r2_fetch.http.client.HTTPException) as e:
                u = urllib.parse.urlsplit(url)
                self.conns.drop(u.scheme, u.netloc)
                err = str(e)
            time.sleep(min(30, 2 ** attempt))
        raise Fail(f"Hub read {self.url} @{off}+{n}: {err}")


def file_reader(p):
    def read(off, n):
        with open(p, "rb") as f:
            return os.pread(f.fileno(), n, off)
    return read


def hub_sources(dest, conns, rev_override=None):
    rows = hub_rows()
    if dest not in rows:
        raise Fail(f"{dest}: no Hub row (weights-manifest.tsv + weights-revisions.tsv, fetcher hub); use publish-dir")
    repo, rev, globs = rows[dest]
    rev = rev_override or rev
    globs = globs + ["model_index.json"]  # as fetch-hub-tree.py
    rec = recorded_hashes()
    want = [x for x in hub_list(repo, rev) if any(fnmatch.fnmatch(x[0], g) for g in globs)]
    if not want:
        raise Fail(f"{repo}@{rev}: nothing matches {' '.join(globs)}")
    out = []
    for path, size, lfs, oid in sorted(want):
        r = rec.get((dest, path))
        out.append(Source(path, size, HubReader(conns, repo, rev, path), sha256=lfs, sha1_blob=None if lfs else oid, recorded=(r[1], r[2]) if r else None))
    return rev, {"kind": "hub", "repo": repo, "revision": rev, "globs": globs}, out


SKIP_NAMES = {".complete", ".fv-revision", ".fv-source", ".fv-manifest.json"}


def dir_sources(dest, root, globs=None, exclude=(), hidden=False):
    root = Path(root)
    if not root.is_dir():
        raise Fail(f"{root}: not a directory")
    rec = recorded_hashes()
    out = []
    # followlinks: the HF-cache layout's top-level links may point at snapshot directories
    for dp, dns, fns in os.walk(root, followlinks=True):
        rel_dir = os.path.relpath(dp, root)
        # never descend into fetchers' temp folders, HF-cache internals or caches
        dns[:] = sorted(d for d in dns if (hidden or not d.startswith(".")) and not d.startswith("models--"))
        for fn in sorted(fns):
            rel = fn if rel_dir == "." else f"{rel_dir}/{fn}"
            if not hidden and (fn in SKIP_NAMES or fn.startswith(".")):
                continue
            if globs and not any(fnmatch.fnmatch(rel, g) for g in globs):
                continue
            if any(fnmatch.fnmatch(rel, g) for g in exclude):
                continue
            full = Path(dp) / fn
            real = full.resolve()  # HF-cache layout: top-level links into blobs/
            if not real.is_file():
                continue
            r = rec.get((dest, rel))
            out.append(Source(rel, real.stat().st_size, file_reader(real), recorded=(r[1], r[2]) if r else None))
    if not out:
        raise Fail(f"{root}: no files")
    return out


def hub_dir_sources(dest, root, conns):
    """A Hub tree copied from a directory (the volume), checked against the Hub at the pinned revision: the file set is
    the Hub listing's (manifest globs + model_index.json), every file must exist with the listed size, and each one's
    bytes must hash to the Hub's LFS SHA-256 (or git blob SHA-1) before its upload completes. HF-cache layouts are
    read through their top-level links, else models--*/snapshots/<rev>/. Files under the globs that the revision does
    not list are left out (reported)."""
    repo, rev, globs = hub_rows()[dest]
    globs = globs + ["model_index.json"]
    want = {p: (size, lfs, oid) for p, size, lfs, oid in hub_list(repo, rev) if any(fnmatch.fnmatch(p, g) for g in globs)}
    if not want:
        raise Fail(f"{repo}@{rev}: nothing matches {' '.join(globs)}")
    root = Path(root)
    try:
        # hidden=True: a Hub repo may hold dot files (.gitattributes); the Hub listing decides what is kept
        have = {s.path: s for s in dir_sources(dest, root, globs, hidden=True)}
    except Fail:
        have = {}  # nothing at the top level: an HF-cache layout without top-level links (the snapshot below)
    snaps = sorted(root.glob(f"models--*/snapshots/{rev}")) if root.is_dir() else []
    rec = recorded_hashes()
    out, missing = [], []
    for p, (size, lfs, oid) in sorted(want.items()):
        # the pinned revision's own snapshot (HF-cache layout) first, else the plain / top-level file
        src = None
        for sd in snaps:
            f = (sd / p).resolve()
            if f.is_file():
                r = rec.get((dest, p))
                src = Source(p, f.stat().st_size, file_reader(f), recorded=(r[1], r[2]) if r else None)
                break
        src = src or have.get(p)
        if src is None:
            missing.append(p)
            continue
        if src.size != size:
            raise Fail(f"{dest}/{p}: {src.size} bytes in {root}, the Hub lists {size} at {rev[:12]}")
        src.sha256 = lfs
        src.sha1_blob = None if lfs else oid
        out.append(src)
    if missing:
        raise Fail(f"{dest}: {len(missing)} file(s) of {repo}@{rev[:12]} not in {root}: {', '.join(missing[:8])}")
    extra = sorted(set(have) - set(want))
    if extra:
        log(f"{dest}: left out {len(extra)} file(s) under the globs that {rev[:12]} does not list: {', '.join(extra[:8])}")
    return rev, {"kind": "hub-via-dir", "repo": repo, "revision": rev, "globs": globs, "path": str(root.resolve())}, out


# --------------------------------------------------------------------------- publish
class Publisher:
    def __init__(self, s3, args):
        self.s3 = s3
        self.prefix = args.prefix
        self.files_parallel = args.files
        self.window = args.window
        self.part_size = PART_SIZE
        self.dry = args.dry_run
        self.bytes = 0
        self.skipped = 0
        self.checked = {}
        self.blk = threading.Lock()
        self.t0 = time.time()

    def add(self, n):
        with self.blk:
            self.bytes += n

    @staticmethod
    def checks(src):
        """What a file's bytes were checked against before its upload completed (none: its hash is recorded from the copy)."""
        return (["hub-lfs-sha256"] if src.sha256 else []) + (["hub-git-blob-sha1"] if src.sha1_blob else []) + ([f"repo-{src.recorded[0]}"] if src.recorded else [])

    def expected_ok(self, src, sha256, sha1_blob, other):
        """Raises when the bytes do not match what is known about them."""
        if src.sha256 and sha256 != src.sha256.lower():
            raise Fail(f"{src.path}: sha256 {sha256} != Hub LFS {src.sha256}")
        if src.sha1_blob and sha1_blob and sha1_blob != src.sha1_blob:
            raise Fail(f"{src.path}: git blob sha1 {sha1_blob} != Hub {src.sha1_blob}")
        if src.recorded:
            algo, hexes = src.recorded
            got = sha256 if algo == "sha256" else other.get(algo)
            if got and got not in hexes:
                raise Fail(f"{src.path}: {algo} {got} not in weights-sha256.tsv ({', '.join(h[:16] + '…' for h in hexes)})")

    def adopt(self, key, pkey, src):
        whole_r2, whole_src, parts = hashlib.sha256(), hashlib.sha256(), []
        for i in range(max(1, -(-src.size // self.part_size)) if src.size else 0):
            off = i * self.part_size
            n = min(self.part_size, src.size - off)
            a = self.s3.get(key, f"bytes={off}-{off + n - 1}") if n else b""
            b = src.read(off, n)
            if a is None or len(a) != n or len(b) != n:
                raise Fail(f"{key}: short read while checking an object without its record")
            whole_r2.update(a)
            whole_src.update(b)
            parts.append(hashlib.sha256(a).hexdigest())
        sha = whole_r2.hexdigest()
        if sha != whole_src.hexdigest():
            raise Fail(f"{key} exists without its parts record and its bytes differ from {src.path}: add-only, not replacing it")
        self.expected_ok(src, sha, None, {})
        rec = {"path": src.path, "size": src.size, "sha256": sha, "part_size": self.part_size, "parts": parts, "checked": self.checks(src), "adopted": True}
        self.s3.put_new(pkey, json.dumps(rec).encode(), content_type="application/json")
        log(f"  {src.path}: object without a record: same bytes as the source, record written")
        return rec

    def publish_file(self, dest, rev, src, key=None, pkey=None):
        key = key or obj_key(self.prefix, dest, rev, src.path)
        pkey = pkey or parts_key(self.prefix, dest, rev, src.path)
        have = self.s3.head(key)
        if have is not None:
            rec = self.s3.get(pkey)
            if have[0] != src.size:
                raise Fail(f"{key} exists with {have[0]} bytes, the source has {src.size}: add-only, not replacing it")
            if rec is None:
                # An earlier run stopped between the object and its record: the object is kept only if its bytes are
                # the source's (both read and hashed here), then the record is written; otherwise the run stops.
                rec = self.adopt(key, pkey, src)
            else:
                rec = json.loads(rec)
            self.expected_ok(src, rec["sha256"], None, {})
            self.checked[src.path] = rec.get("checked", self.checks(src))
            self.skipped += 1
            self.add(src.size)
            return rec
        if self.dry:
            return {"path": src.path, "size": src.size, "sha256": src.sha256 or "?", "parts": []}
        want_algos = set()
        if src.recorded and src.recorded[0] not in ("sha256",):
            want_algos.add(src.recorded[0])
        whole = hashlib.sha256()
        extra = {a: hashlib.new(a) for a in want_algos}
        blob = hashlib.sha1(f"blob {src.size}\0".encode()) if src.sha1_blob else None
        nparts = max(1, -(-src.size // self.part_size))
        part_shas = []
        meta = {"sha256": src.sha256} if src.sha256 else ({"sha256": src.recorded[1][0]} if src.recorded and src.recorded[0] == "sha256" and len(src.recorded[1]) == 1 else {})
        if src.size <= self.part_size:
            data = src.read(0, src.size) if src.size else b""
            if len(data) != src.size:
                raise Fail(f"{src.path}: read {len(data)} of {src.size} bytes")
            for h in (whole, *extra.values(), *( [blob] if blob else [])):
                h.update(data)
            sha = whole.hexdigest()
            part_shas = [sha] if src.size else []
            self.expected_ok(src, sha, blob.hexdigest() if blob else None, {a: h.hexdigest() for a, h in extra.items()})
            if not self.s3.put_new(key, data, meta={"sha256": sha}):
                raise Fail(f"{key} appeared while uploading (412): add-only, stopping")
            self.add(src.size)
        else:
            uid = self.s3.mp_create(key, meta)
            etags = [None] * nparts
            try:
                with concurrent.futures.ThreadPoolExecutor(self.window) as ex:
                    def job(i):
                        off = i * self.part_size
                        n = min(self.part_size, src.size - off)
                        data = src.read(off, n)
                        if len(data) != n:
                            raise Fail(f"{src.path}: read {len(data)} of {n} bytes at {off}")
                        psha = hashlib.sha256(data).hexdigest()
                        etags[i] = self.s3.mp_part(key, uid, i + 1, data, psha)
                        return data, psha
                    pending = deque()
                    nxt = 0
                    for i in range(nparts):
                        while nxt < nparts and len(pending) < self.window:
                            pending.append(ex.submit(job, nxt))
                            nxt += 1
                        data, psha = pending.popleft().result()
                        for h in (whole, *extra.values(), *([blob] if blob else [])):
                            h.update(data)
                        part_shas.append(psha)
                        self.add(len(data))
                        del data
                sha = whole.hexdigest()
                self.expected_ok(src, sha, blob.hexdigest() if blob else None, {a: h.hexdigest() for a, h in extra.items()})
                if self.s3.head(key) is not None:
                    raise Fail(f"{key} appeared while uploading: add-only, stopping")
                self.s3.mp_complete(key, uid, etags)
            except BaseException:
                self.s3.mp_abort(key, uid)
                raise
            got = self.s3.head(key)
            if not got or got[0] != src.size:
                raise Fail(f"{key}: {got[0] if got else 'no object'} bytes after completing, expected {src.size}")
        rec = {"path": src.path, "size": src.size, "sha256": sha, "part_size": self.part_size, "parts": part_shas, "checked": self.checks(src)}
        self.checked[src.path] = rec["checked"]
        for a, h in extra.items():
            rec[a] = h.hexdigest()
        if not self.s3.put_new(pkey, json.dumps(rec).encode(), content_type="application/json"):
            old = json.loads(self.s3.get(pkey) or b"{}")
            if old.get("sha256") != sha:
                raise Fail(f"{pkey} exists with another sha256")
        return rec

    def publish(self, dest, rev, source, files):
        check_name(dest, rev)
        mkey = manifest_key(self.prefix, dest, rev)
        if self.s3.head(mkey) is not None:
            m = json.loads(self.s3.get(mkey))
            log(f"{dest}@{rev}: already published ({len(m['files'])} files, {m['total_bytes'] / 1e9:.2f} GB): nothing to do")
            return m
        total = sum(s.size for s in files)
        log(f"{dest}@{rev}: {len(files)} files, {total / 1e9:.2f} GB -> r2://{self.s3.bucket}/{obj_key(self.prefix, dest, rev, '')}")
        recs = {}
        stop = threading.Event()

        def one(src):
            if stop.is_set():
                return None
            t = time.time()
            r = self.publish_file(dest, rev, src)
            el = time.time() - t
            log(f"  {src.path}: {src.size / 1e6:.1f} MB {'' if not el else f'{src.size / el / 1e6:.0f} MB/s'} sha256 {r['sha256'][:16]}…")
            return r

        # big files first so the long ones overlap
        order = sorted(files, key=lambda s: -s.size)
        with concurrent.futures.ThreadPoolExecutor(self.files_parallel) as ex:
            futs = {ex.submit(one, s): s for s in order}
            try:
                for f in concurrent.futures.as_completed(futs):
                    r = f.result()
                    if r:
                        recs[r["path"]] = r
            except BaseException:
                stop.set()
                raise
        if self.dry:
            log(f"{dest}@{rev}: dry run, nothing written")
            return None
        manifest = {
            "format": MANIFEST_FORMAT,
            "dest": dest,
            "rev": rev,
            "part_size": self.part_size,
            "prefix": obj_key(self.prefix, dest, rev, ""),
            "source": source,
            "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_bytes": total,
            "files": [{k: recs[p][k] for k in ("path", "size", "sha256", "parts") if k in recs[p]} for p in sorted(recs)],
        }
        if not self.s3.put_new(mkey, json.dumps(manifest, indent=1, sort_keys=True).encode(), content_type="application/json"):
            raise Fail(f"{mkey} appeared while publishing: add-only, not replacing it")
        el = time.time() - self.t0
        none = sorted(p for p, c in self.checked.items() if not c)
        kinds = {}
        for c in self.checked.values():
            for k in c or ["none"]:
                kinds[k] = kinds.get(k, 0) + 1
        log(f"{dest}@{rev}: hash checks {json.dumps(kinds, sort_keys=True)}" + (f"; NO recorded hash (SHA-256 recorded from this copy): {', '.join(none)}" if none else ""))
        log(f"{dest}@{rev}: PUBLISHED {mkey} ({total / 1e9:.2f} GB; {self.skipped} files already there; {el:.0f}s, {(self.bytes / max(el, 1e-6)) / 1e6:.0f} MB/s)")
        return manifest


# --------------------------------------------------------------------------- read side
def manifests(s3, prefix):
    out = {}
    for key, size in s3.list(f"{prefix}manifests/"):
        if size < 0 or not key.endswith(".json"):
            continue
        rest = key[len(f"{prefix}manifests/"):-len(".json")]
        dest, rev = rest.rsplit("/", 1)
        out.setdefault(dest, []).append(rev)
    return out


def load_manifest(s3, prefix, dest, rev=None):
    if rev is None:
        revs = manifests(s3, prefix).get(dest) or []
        hub = hub_rows().get(dest)
        if hub and hub[1] in revs:
            rev = hub[1]
        elif len(revs) == 1:
            rev = revs[0]
        elif not revs:
            raise Fail(f"{dest}: not published")
        else:
            raise Fail(f"{dest}: several revisions published ({', '.join(revs)}): pass --rev")
    data = s3.get(manifest_key(prefix, dest, rev))
    if data is None:
        raise Fail(f"{dest}@{rev}: not published")
    m = json.loads(data)
    if m.get("format") != MANIFEST_FORMAT:
        raise Fail(f"{dest}@{rev}: manifest format {m.get('format')}")
    return m


def verify_tree(s3, prefix, m, deep, workers):
    """Every manifest file is in the bucket with its size; --deep: and hashes to its part and file SHA-256s."""
    bad = []
    rec = recorded_hashes()

    def one_shallow(f):
        key = obj_key(prefix, m["dest"], m["rev"], f["path"])
        h = s3.head(key)
        if h is None:
            return f"{f['path']}: missing"
        if h[0] != f["size"]:
            return f"{f['path']}: {h[0]} bytes, manifest {f['size']}"
        r = rec.get((m["dest"], f["path"]))
        if r and r[1] == "sha256" and f["sha256"] not in r[2]:
            return f"{f['path']}: manifest sha256 not in weights-sha256.tsv"
        return None

    def one(f):
        key = obj_key(prefix, m["dest"], m["rev"], f["path"])
        h = s3.head(key)
        if h is None:
            return f"{f['path']}: missing"
        if h[0] != f["size"]:
            return f"{f['path']}: {h[0]} bytes, manifest {f['size']}"
        r = rec.get((m["dest"], f["path"]))
        if r and r[1] == "sha256" and f["sha256"] not in r[2]:
            return f"{f['path']}: manifest sha256 not in weights-sha256.tsv"
        if not deep:
            return None
        ps = m["part_size"]
        whole = hashlib.sha256()
        t0 = time.time()

        def get_part(i):
            a = i * ps
            b = min(f["size"], a + ps)
            data = s3.get(key, f"bytes={a}-{b - 1}")
            if data is None or len(data) != b - a:
                return i, None, "short"
            return i, data, None if hashlib.sha256(data).hexdigest() == f["parts"][i] else "sha256 mismatch"

        # One stream is slow across regions: `window` ranged GETs of this file in flight, consumed in order for the
        # whole-file SHA-256.
        with concurrent.futures.ThreadPoolExecutor(window) as px:
            pending = deque()
            nxt = 0
            for _ in range(len(f["parts"])):
                while nxt < len(f["parts"]) and len(pending) < window:
                    pending.append(px.submit(get_part, nxt))
                    nxt += 1
                i, data, err = pending.popleft().result()
                if err:
                    for x in pending:
                        x.cancel()
                    return f"{f['path']}: part {i} {err}"
                whole.update(data)
        if whole.hexdigest() != f["sha256"]:
            return f"{f['path']}: sha256 mismatch"
        el = time.time() - t0
        log(f"  verified {m['dest']}/{f['path']}: {f['size'] / 1e9:.2f} GB in {el:.0f}s ({f['size'] / max(el, 1e-6) / 1e6:.0f} MB/s)")
        return None

    window = max(1, int(os.environ.get("FV_VERIFY_WINDOW", "8")))
    if deep and os.environ.get("FV_VERIFY_WHOLE_FILE") != "1":
        # Every 64 MiB part against the manifest's part SHA-256, all parts of the tree at once (workers x window GETs
        # in flight, no ordering): R2's slow tail GETs (seconds to tens of seconds, seen 2026-10-12) then cost one
        # thread each instead of stalling a file. The part hashes are the ones the publisher computed from the same
        # bytes as each file's SHA-256 (checked before the upload completed); FV_VERIFY_WHOLE_FILE=1 re-hashes whole files.
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            for f, err in zip(m["files"], ex.map(lambda f: one_shallow(f), m["files"])):
                if err:
                    bad.append(err)
        if bad:
            return bad
        ps = m["part_size"]
        t0 = time.time()

        def part(job):
            f, i = job
            a = i * ps
            b = min(f["size"], a + ps)
            data = s3.get(obj_key(prefix, m["dest"], m["rev"], f["path"]), f"bytes={a}-{b - 1}")
            if data is None or len(data) != b - a:
                return f"{f['path']}: part {i} short"
            return None if hashlib.sha256(data).hexdigest() == f["parts"][i] else f"{f['path']}: part {i} sha256 mismatch"

        jobs = [(f, i) for f in m["files"] for i in range(len(f["parts"]))]
        with concurrent.futures.ThreadPoolExecutor(workers * window) as ex:
            for err in ex.map(part, jobs):
                if err:
                    bad.append(err)
        el = time.time() - t0
        log(f"  parts of {m['dest']}: {len(jobs)} in {el:.0f}s ({m['total_bytes'] / max(el, 1e-6) / 1e6:.0f} MB/s)")
        return bad
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for f, err in zip(m["files"], ex.map(one, m["files"])):
            if err:
                bad.append(err)
    return bad


def cmd_status(s3, args):
    pub = manifests(s3, args.prefix)
    hub = hub_rows()
    rows = []
    seen = set()
    for dest, (repo, rev, _) in hub.items():
        rows.append((dest, "hub", rev, "published" if rev in pub.get(dest, []) else ("other rev: " + ",".join(pub[dest]) if dest in pub else "-")))
        seen.add(dest)
    for dest, why in VOLUME_ONLY:
        rows.append((dest, "volume", ",".join(pub.get(dest, [])) or "-", "published" if dest in pub else "-  " + why))
        seen.add(dest)
    for dest in sorted(set(pub) - seen):
        rows.append((dest, "other", ",".join(pub[dest]), "published"))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:<6}  {r[2][:12]:<12}  {r[3]}")
    n = sum(1 for r in rows if r[3] == "published")
    print(f"{n}/{len(rows)} published in r2://{s3.bucket}/{args.prefix}")


def cmd_record(s3, args):
    """weights-r2.tsv rows (dest, rev, files, bytes, manifest sha256) for every published manifest."""
    for dest, revs in sorted(manifests(s3, args.prefix).items()):
        for rev in sorted(revs):
            raw = s3.get(manifest_key(args.prefix, dest, rev))
            m = json.loads(raw)
            print("\t".join([dest, rev, str(len(m["files"])), str(m["total_bytes"]), hashlib.sha256(raw).hexdigest(), m["source"].get("kind", "?")]))


def pinned_manifests(path):
    """(dest, rev) -> manifest SHA-256 of a weights-r2.tsv (column 5), for `plan --pin`."""
    pins = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 5 or not re.fullmatch(r"[0-9a-f]{64}", f[4]):
            raise Fail(f"{path}:{n}: not dest<TAB>rev<TAB>files<TAB>bytes<TAB>manifest sha256")
        pins[(f[0], f[1])] = f[4]
    return pins


def cmd_plan(s3, args):
    pins = pinned_manifests(args.pin) if args.pin else None
    trees = []
    for d in args.dests:
        dest, _, rev = d.partition("@")
        m = load_manifest(s3, args.prefix, dest, rev or None)
        if pins is not None:
            # As fv-control's r2Manifest: the manifest's bytes are the ones the record pins.
            want = pins.get((dest, m["rev"]))
            if want is None:
                raise Fail(f"{dest}@{m['rev']}: not in {args.pin} (--pin)")
            got = hashlib.sha256(s3.get(manifest_key(args.prefix, dest, m["rev"])) or b"").hexdigest()
            if got != want:
                raise Fail(f"{dest}@{m['rev']}: manifest sha256 {got[:16]}… is not {want[:16]}… ({args.pin})")
        files = [dict(f, url=presign(s3.endpoint, s3.bucket, obj_key(args.prefix, dest, m["rev"], f["path"]), s3.access, s3.secret, args.ttl)) for f in m["files"]]
        trees.append({"dest": dest, "rev": m["rev"], "part_size": m["part_size"], "total_bytes": m["total_bytes"], "files": files})
    plan = {"format": r2_fetch.PLAN_FORMAT, "expires_at": int(time.time() + args.ttl), "trees": trees}
    out = json.dumps(plan)
    if args.out:
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # presigned URLs are bearer tokens
        with os.fdopen(fd, "w") as f:
            f.write(out)
        log(f"plan: {len(trees)} tree(s), {sum(t['total_bytes'] for t in trees) / 1e9:.2f} GB, URLs valid {args.ttl}s -> {args.out}")
    else:
        print(out)


def cmd_compare_dir(s3, args):
    """The directory's copy against the manifest: every file present with its size and (unless --sizes-only) every
    64 MiB part's SHA-256 equal to the manifest's, the parts read in parallel (--workers): one stream from a network
    volume is slow, many are not. The manifest's part hashes are of the bytes R2 holds (verify --deep checks that)."""
    m = load_manifest(s3, args.prefix, args.dest, args.rev)
    root = Path(args.dir)
    snaps = sorted(root.glob(f"models--*/snapshots/{m['rev']}"))  # HF-cache layout: the pinned snapshot first
    ps = m["part_size"]
    bad = []
    jobs = []
    for f in m["files"]:
        p = next((q for q in ((sd / f["path"]).resolve() for sd in snaps) if q.is_file()), None) or (root / f["path"]).resolve()
        if not p.is_file():
            bad.append(f"{f['path']}: not in {root}")
            continue
        if p.stat().st_size != f["size"]:
            bad.append(f"{f['path']}: {p.stat().st_size} bytes on disk, {f['size']} in R2")
            continue
        if not args.sizes_only:
            jobs += [(f, p, i) for i in range(len(f["parts"]))]

    def part_ok(job):
        f, p, i = job
        off = i * ps
        n = min(ps, f["size"] - off)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            got = 0
            while got < n:
                b = os.pread(fh.fileno(), min(16 << 20, n - got), off + got)
                if not b:
                    break
                h.update(b)
                got += len(b)
        return None if got == n and h.hexdigest() == f["parts"][i] else f"{f['path']}: part {i} differs"

    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for err in ex.map(part_ok, jobs):
            if err and err.split(": part")[0] not in {b.split(":")[0] for b in bad}:
                bad.append(err)
    for line in bad:
        print("DIFF " + line)
    gb = sum(f["size"] for f in m["files"]) / 1e9
    print(f"{args.dest}@{m['rev']}: {len(m['files']) - len(bad)}/{len(m['files'])} files equal ({'sizes' if args.sizes_only else 'sha256 of every part'}; {gb:.2f} GB, {time.time() - t:.0f}s)")
    return 1 if bad else 0


def cmd_archive(s3, args):
    """Every regular file of DIR (hidden ones too) under archive/<label>/<name>/<path>, add-only, with its SHA-256 in
    archive/<label>/.parts/<name>/<path>.json and, last, an index archive/<label>/<name>.index.json."""
    r2_fetch.safe_rel(args.name)
    base = f"{args.prefix}archive/{args.label}/"
    ikey = f"{base}{args.name}.index.json"
    if s3.head(ikey) is not None:
        log(f"archive {args.name}: already done ({ikey})")
        return 0
    files = dir_sources(args.name, args.dir, hidden=True)
    pub = Publisher(s3, args)
    t0 = time.time()
    recs = {}
    with concurrent.futures.ThreadPoolExecutor(args.files) as ex:
        futs = {ex.submit(pub.publish_file, args.name, "-", f, f"{base}{args.name}/{f.path}", f"{base}.parts/{args.name}/{f.path}.json"): f for f in sorted(files, key=lambda f: -f.size)}
        for fu in concurrent.futures.as_completed(futs):
            r = fu.result()
            recs[r["path"]] = r
    if args.dry_run:
        return 0
    total = sum(r["size"] for r in recs.values())
    idx = {"format": "fv-archive-index/1", "name": args.name, "source": str(Path(args.dir).resolve()), "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "total_bytes": total,
           "files": [{k: recs[p][k] for k in ("path", "size", "sha256")} for p in sorted(recs)]}
    if not s3.put_new(ikey, json.dumps(idx, indent=1).encode(), content_type="application/json"):
        raise Fail(f"{ikey} appeared while archiving")
    el = time.time() - t0
    log(f"archive {args.name}: {len(recs)} files, {total / 1e9:.2f} GB -> {base}{args.name}/ ({el:.0f}s, {total / max(el, 1e-6) / 1e6:.0f} MB/s)")
    return 0


def cmd_compare_archive(s3, args):
    """An archive (archive/<label>/<name>/) against its directory: the same set of files (hidden ones too), each with
    its size, each R2 object present with that size, each file's parts record agreeing with the index, and (unless
    --sizes-only) every 64 MiB part of the file on disk hashing to the parts record's SHA-256, read in parallel."""
    base = f"{args.prefix}archive/{args.label}/"
    raw = s3.get(f"{base}{args.name}.index.json")
    if raw is None:
        raise Fail(f"no index {base}{args.name}.index.json")
    idx = json.loads(raw)
    disk = {f.path: f.size for f in dir_sources(args.name, args.dir, hidden=True)}
    root = Path(args.dir)
    bad = [f"{p}: on disk, not in the archive" for p in sorted(set(disk) - {f["path"] for f in idx["files"]})]

    def meta(f):
        if f["path"] not in disk:
            return f, None, f"{f['path']}: in the archive, not on disk"
        if disk[f["path"]] != f["size"]:
            return f, None, f"{f['path']}: {disk[f['path']]} bytes on disk, {f['size']} in the index"
        h = s3.head(f"{base}{args.name}/{f['path']}")
        if h is None or h[0] != f["size"]:
            return f, None, f"{f['path']}: R2 object {'missing' if h is None else f'{h[0]} bytes'}"
        rec = json.loads(s3.get(f"{base}.parts/{args.name}/{f['path']}.json") or b"{}")
        if rec.get("sha256") != f["sha256"] or rec.get("size") != f["size"]:
            return f, None, f"{f['path']}: parts record disagrees with the index"
        return f, rec, None

    t = time.time()
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for f, rec, err in ex.map(meta, idx["files"]):
            if err:
                bad.append(err)
            elif not args.sizes_only:
                parts = rec.get("parts") or []
                ps = rec.get("part_size") or PART_SIZE
                if len(parts) != -(-f["size"] // ps):
                    bad.append(f"{f['path']}: parts record has {len(parts)} parts")
                    continue
                jobs += [(f, (root / f["path"]).resolve(), i, ps, parts[i]) for i in range(len(parts))]

    def part_ok(job):
        f, p, i, ps, want = job
        off = i * ps
        n = min(ps, f["size"] - off)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            got = 0
            while got < n:
                b = os.pread(fh.fileno(), min(16 << 20, n - got), off + got)
                if not b:
                    break
                h.update(b)
                got += len(b)
        return None if got == n and h.hexdigest() == want else f"{f['path']}: part {i} differs"

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for err in ex.map(part_ok, jobs):
            if err:
                bad.append(err)
    for line in bad:
        print("DIFF " + line)
    nbad = len({b.split(":")[0] for b in bad})
    print(f"archive/{args.label}/{args.name}: {len(idx['files']) - nbad}/{len(idx['files'])} files equal, {len(disk)} on disk "
          f"({'sizes' if args.sizes_only else 'sha256 of every part'}; {idx['total_bytes'] / 1e9:.2f} GB, {time.time() - t:.0f}s)")
    return 1 if bad else 0


def cmd_create_bucket(args):
    tok = Path(args.cf_token_file).read_text().strip()
    acct = os.environ.get("FV_CF_ACCOUNT_ID", DEFAULT_ACCOUNT)
    bucket = args.bucket or os.environ.get("FV_WEIGHTS_R2_BUCKET") or DEFAULT_BUCKET
    body = json.dumps({"name": bucket, "locationHint": args.location}).encode()
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{acct}/r2/buckets", data=body, method="POST", headers={"authorization": "Bearer " + tok, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.load(r)["result"]
        print(f"created {res.get('name')} location {res.get('location')} at {res.get('creation_date')}")
    except urllib.error.HTTPError as e:
        raise Fail(f"create bucket {bucket}: HTTP {e.code}: {e.read(400).decode('utf-8', 'replace')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--prefix", default=os.environ.get("FV_WEIGHTS_R2_PREFIX", ""), help="a key prefix for tests (e.g. test/<name>/); empty in production")
    ap.add_argument("--cf-token-file", default="/root/.config/fv/cf_api_token")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("publish-hub", "publish-dir"):
        p = sub.add_parser(name)
        p.add_argument("dest")
        if name == "publish-dir":
            p.add_argument("dir")
            p.add_argument("--globs", default="", help="space-separated globs (default: the Hub row's, else every file)")
            p.add_argument("--exclude", default="", help="space-separated globs to leave out")
            p.add_argument("--note", default="", help="where the directory is (recorded in the manifest)")
            p.add_argument("--no-hub-check", action="store_true", help="a Hub dest WITHOUT the Hub listing check (not recommended)")
        p.add_argument("--rev", default="", help="publish-hub: override the pinned revision; publish-dir: required without a Hub row (e.g. eu-20261012)")
        p.add_argument("--files", type=int, default=4, help="files uploaded at once")
        p.add_argument("--window", type=int, default=4, help="parts in flight per file (memory: files x (window+1) x 64 MiB)")
        p.add_argument("--only", default="", help="space-separated globs: publish only these paths (tests); the manifest is not written")
        p.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    sub.add_parser("list")
    sub.add_parser("record")
    p = sub.add_parser("show")
    p.add_argument("dest")
    p.add_argument("--rev", default=None)
    p = sub.add_parser("verify")
    p.add_argument("dests", nargs="*", help="default: every published tree")
    p.add_argument("--deep", action="store_true", help="read every byte and check the part and file SHA-256s")
    p.add_argument("--record", action="store_true", help="write verified/<dest>/<rev>/<stamp>.json (add-only) for each tree that passes")
    p.add_argument("--skip-recorded", action="store_true", help="--deep: skip a tree that already has a verified/ record (a resumed run)")
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("compare-dir")
    p.add_argument("dest")
    p.add_argument("dir")
    p.add_argument("--rev", default=None)
    p.add_argument("--sizes-only", action="store_true")
    p.add_argument("--workers", type=int, default=24, help="parts read and hashed at once")
    p = sub.add_parser("plan")
    p.add_argument("dests", nargs="+", help="dest or dest@rev")
    p.add_argument("--pin", default="", help="a weights-r2.tsv: refuse a manifest whose SHA-256 is not the one recorded there (column 5)")
    p.add_argument("--ttl", type=int, default=3600)
    p.add_argument("--out", default="")
    p = sub.add_parser("archive", help="copy a directory as-is under archive/<label>/<name>/ (no manifest, no preset wiring)")
    p.add_argument("name")
    p.add_argument("dir")
    p.add_argument("--label", default="eu-volume")
    p.add_argument("--files", type=int, default=4)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("compare-archive", help="an archive against its directory (file set, sizes, every part's SHA-256)")
    p.add_argument("name")
    p.add_argument("dir")
    p.add_argument("--label", default="eu-volume")
    p.add_argument("--sizes-only", action="store_true")
    p.add_argument("--workers", type=int, default=24)
    p = sub.add_parser("put-temp", help="stage a file under tmp/ of a NON-weights bucket and print a presigned GET URL")
    p.add_argument("file")
    p.add_argument("--key", required=True)
    p.add_argument("--ttl", type=int, default=3600)
    p = sub.add_parser("rm-temp", help="delete the tmp/<prefix>* objects of a NON-weights bucket")
    p.add_argument("--prefix", dest="tmp_prefix", required=True)
    p = sub.add_parser("create-bucket")
    p.add_argument("--location", default="wnam", help="R2 location hint: wnam enam weur eeur apac oc")
    args = ap.parse_args(argv)
    if args.prefix and not args.prefix.endswith("/"):
        args.prefix += "/"
    try:
        if args.cmd == "create-bucket":
            return cmd_create_bucket(args) or 0
        s3 = credentials(args)
        if args.cmd in ("publish-hub", "publish-dir"):
            pub = Publisher(s3, args)
            if args.cmd == "publish-hub":
                rev, source, files = hub_sources(args.dest, s3.conns, args.rev or None)
            elif hub_rows().get(args.dest) and not args.no_hub_check:
                if args.rev and args.rev != hub_rows()[args.dest][1]:
                    raise Fail(f"{args.dest}: a Hub tree keeps its pinned revision {hub_rows()[args.dest][1]}, not {args.rev}")
                rev, source, files = hub_dir_sources(args.dest, args.dir, s3.conns)
                source["note"] = args.note
            else:
                hub = hub_rows().get(args.dest)
                rev = args.rev or (hub[1] if hub else "")
                if not rev:
                    raise Fail(f"{args.dest}: no Hub row, so --rev is required (e.g. eu-{time.strftime('%Y%m%d')})")
                globs = args.globs.split() or (hub[2] + ["model_index.json"] if hub else None)
                files = dir_sources(args.dest, args.dir, globs, args.exclude.split())
                source = {"kind": "dir", "path": str(Path(args.dir).resolve()), "note": args.note, **({"repo": hub[0], "revision": hub[1]} if hub else {})}
            if args.only:
                files = [f for f in files if any(fnmatch.fnmatch(f.path, g) for g in args.only.split())]
                for f in files:
                    pub.publish_file(args.dest, rev, f)
                log(f"--only: {len(files)} file(s) uploaded; no manifest written ({s3.ops['A']} class A, {s3.ops['B']} class B ops)")
                return 0
            pub.publish(args.dest, rev, source, files)
            log(f"ops: {s3.ops['A']} class A, {s3.ops['B']} class B")
            return 0
        if args.cmd in ("put-temp", "rm-temp"):
            # Only scratch objects, only outside the weights bucket: the weights bucket is add-only.
            if s3.bucket == DEFAULT_BUCKET:
                raise Fail(f"{args.cmd}: not in the weights bucket {DEFAULT_BUCKET}")
            if args.cmd == "put-temp":
                if not args.key.startswith("tmp/"):
                    raise Fail("put-temp: the key must start with tmp/")
                s3.request("PUT", args.key, headers={"content-type": "application/octet-stream"}, body=Path(args.file).read_bytes())
                print(presign(s3.endpoint, s3.bucket, args.key, s3.access, s3.secret, args.ttl))
                return 0
            if not args.tmp_prefix.startswith("tmp/"):
                raise Fail("rm-temp: the prefix must start with tmp/")
            for k, size in list(s3.list(args.tmp_prefix)):
                if size >= 0 and k.startswith(args.tmp_prefix):
                    s3.request("DELETE", k, ok=(200, 204))
                    log(f"deleted {s3.bucket}/{k}")
            return 0
        if args.cmd == "archive":
            return cmd_archive(s3, args)
        if args.cmd == "compare-archive":
            return cmd_compare_archive(s3, args)
        if args.cmd == "status":
            return cmd_status(s3, args) or 0
        if args.cmd == "list":
            for dest, revs in sorted(manifests(s3, args.prefix).items()):
                print(dest, " ".join(revs))
            return 0
        if args.cmd == "record":
            return cmd_record(s3, args) or 0
        if args.cmd == "show":
            print(json.dumps(load_manifest(s3, args.prefix, args.dest, args.rev), indent=1))
            return 0
        if args.cmd == "verify":
            # FV_WEIGHTS_R2_VERIFY_ACCESS_KEY_ID / _SECRET_ACCESS_KEY: every read goes through that (read-only) key, e.g.
            # fv-control's, to prove it reads everything; the verified/ records are written with the main key.
            rs3 = s3
            if os.environ.get("FV_WEIGHTS_R2_VERIFY_ACCESS_KEY_ID") and os.environ.get("FV_WEIGHTS_R2_VERIFY_SECRET_ACCESS_KEY"):
                rs3 = S3(s3.endpoint, s3.bucket, os.environ["FV_WEIGHTS_R2_VERIFY_ACCESS_KEY_ID"], os.environ["FV_WEIGHTS_R2_VERIFY_SECRET_ACCESS_KEY"])
                log("verify: reading with the FV_WEIGHTS_R2_VERIFY_* key")
            pub = manifests(rs3, args.prefix)
            dests = args.dests or sorted(pub)
            fails = 0
            for d in dests:
                dest, _, rev = d.partition("@")
                for r in ([rev] if rev else pub.get(dest, [None])):
                    try:
                        m = load_manifest(rs3, args.prefix, dest, r)
                    except Fail as e:
                        print(f"FAIL {dest}: {e}")
                        fails += 1
                        continue
                    if args.deep and args.skip_recorded and any(size >= 0 for _, size in rs3.list(f"{args.prefix}verified/{dest}/{m['rev']}/")):
                        print(f"ok   {dest}@{m['rev']}: {len(m['files'])} files, {m['total_bytes'] / 1e9:.2f} GB (sha256, recorded earlier)")
                        continue
                    t = time.time()
                    bad = verify_tree(rs3, args.prefix, m, args.deep, args.workers)
                    what = "sha256" if args.deep else "size"
                    if bad:
                        fails += 1
                        print(f"FAIL {dest}@{m['rev']}: " + "; ".join(bad[:10]))
                    else:
                        print(f"ok   {dest}@{m['rev']}: {len(m['files'])} files, {m['total_bytes'] / 1e9:.2f} GB ({what}, {time.time() - t:.0f}s)")
                        if args.record and args.deep:
                            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                            s3.put_new(f"{args.prefix}verified/{dest}/{m['rev']}/{stamp}.json", json.dumps({"dest": dest, "rev": m["rev"], "files": len(m["files"]), "total_bytes": m["total_bytes"], "check": "sha256 of every file" if os.environ.get("FV_VERIFY_WHOLE_FILE") == "1" else "sha256 of every 64 MiB part (manifest part hashes)", "read_with": "verify key" if rs3 is not s3 else "main key", "at": stamp}).encode(), content_type="application/json")
            return 1 if fails else 0
        if args.cmd == "compare-dir":
            return cmd_compare_dir(s3, args)
        if args.cmd == "plan":
            return cmd_plan(s3, args) or 0
    except Fail as e:
        log(f"FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
