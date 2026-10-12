#!/usr/bin/env python3
"""Measure R2 fetch throughput on a pod (docs/ops/r2-weights.md §5.4): run by r2-fetch-bench.sh on a Runpod CPU pod.

Python 3.8+ stdlib only. Reads with a READ-ONLY R2 key pair from the environment (FV_R2_RO_ID / FV_R2_RO_SECRET,
never printed): it presigns its own GET URLs (SigV4 query auth, as fv-control does) and never writes to R2.

Steps, each appended to $FV_BENCH_OUT/results.jsonl (and a line to log.txt):
  host     CPUs, model, SHA-NI, one core's SHA-256 rate, N threads' SHA-256 rate, a 4 GB pwrite+fsync
  raw      ranged GETs to nowhere (no hash, no write) for FV_BENCH_RAW_S seconds per (streams, MiB per GET)
  fetch    r2-fetch.py on the tree(s) of FV_BENCH_TREES (dest@rev ...) for each "concurrency:get_parts:flags" of
           FV_BENCH_FETCH, from empty each time; wall time, MB/s and the fetcher's CPU seconds
Env: FV_WEIGHTS_R2_ENDPOINT, FV_WEIGHTS_R2_BUCKET (default fv-weights), FV_BENCH_TREES, FV_BENCH_RAW
("32:64 64:64 ..."), FV_BENCH_RAW_S (20), FV_BENCH_FETCH ("32:1 64:4:--no-full-hash ..."), FV_BENCH_OUT (/srv),
FV_BENCH_ROOT (/bench).
"""
import datetime
import hashlib
import hmac
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FV_BENCH_OUT", "/srv")
ROOT = os.environ.get("FV_BENCH_ROOT", "/bench")
EP = os.environ.get("FV_WEIGHTS_R2_ENDPOINT", "").rstrip("/")
BUCKET = os.environ.get("FV_WEIGHTS_R2_BUCKET", "fv-weights")


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f:
        f.write(line + "\n")


def record(kind, **kw):
    kw = {"kind": kind, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    with open(os.path.join(OUT, "results.jsonl"), "a") as f:
        f.write(json.dumps(kw) + "\n")
    log(f"{kind}: " + json.dumps({k: v for k, v in kw.items() if k not in ("kind", "at")}))


# ---------------------------------------------------------------- SigV4 presigned GET (as control/src/r2weights.ts)
def _enc(s, safe="-_.~"):
    return urllib.parse.quote(s, safe=safe)


def presign(key, ttl=3600):
    ak, sk = os.environ["FV_R2_RO_ID"], os.environ["FV_R2_RO_SECRET"]
    u = urllib.parse.urlsplit(EP)
    now = datetime.datetime.now(datetime.timezone.utc)
    amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/auto/s3/aws4_request"
    path = "/" + _enc(BUCKET) + "/" + "/".join(_enc(p) for p in key.split("/"))
    q = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{ak}/{scope}", "X-Amz-Date": amz, "X-Amz-Expires": str(ttl), "X-Amz-SignedHeaders": "host"}
    cq = "&".join(f"{_enc(k)}={_enc(q[k])}" for k in sorted(q))
    canon = "\n".join(["GET", path, cq, f"host:{u.netloc}\n", "host", "UNSIGNED-PAYLOAD"])
    sts = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(canon.encode()).hexdigest()])
    k = ("AWS4" + sk).encode()
    for p in (day, "auto", "s3", "aws4_request"):
        k = hmac.new(k, p.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    return f"{u.scheme}://{u.netloc}{path}?{cq}&X-Amz-Signature={sig}"


def make_plan(trees):
    out = []
    for t in trees:
        dest, rev = t.split("@", 1)
        with urllib.request.urlopen(presign(f"manifests/{dest}/{rev}.json", 600), timeout=60) as r:
            m = json.load(r)
        files = [dict(f, url=presign(f"weights/{dest}/{rev}/{f['path']}")) for f in m["files"]]
        out.append({"dest": dest, "rev": rev, "part_size": m["part_size"], "total_bytes": m["total_bytes"], "files": files})
    return {"format": "fv-weights-plan/1", "expires_at": int(time.time()) + 3600, "trees": out}


# ---------------------------------------------------------------- host
def host():
    info = {}
    try:
        cpu = open("/proc/cpuinfo").read()
        info["model"] = next((ln.split(":", 1)[1].strip() for ln in cpu.splitlines() if ln.startswith("model name")), "?")
        info["sha_ni"] = " sha_ni" in cpu
    except OSError:
        pass
    info["cpus"] = os.cpu_count()
    buf = os.urandom(1 << 20) * 256
    t = time.time()
    for _ in range(4):
        hashlib.sha256(buf).hexdigest()
    info["sha256_1core_MBps"] = round(4 * len(buf) / (time.time() - t) / 1e6)
    n = max(1, os.cpu_count() or 1)

    def hh():
        for _ in range(2):
            hashlib.sha256(buf).hexdigest()
    th = [threading.Thread(target=hh) for _ in range(n)]
    t = time.time()
    for x in th:
        x.start()
    for x in th:
        x.join()
    info["sha256_allcores_MBps"] = round(2 * n * len(buf) / (time.time() - t) / 1e6)
    os.makedirs(ROOT, exist_ok=True)
    gb = int(os.environ.get("FV_BENCH_DD_GB", "4"))
    if gb:
        p = os.path.join(ROOT, "ddtest")
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        t = time.time()
        for i in range(gb * 4):
            os.pwrite(fd, buf, i * len(buf))
        os.fsync(fd)
        os.close(fd)
        info["pwrite_fsync_MBps"] = round(gb * 4 * len(buf) / (time.time() - t) / 1e6)
        os.unlink(p)
    du = shutil.disk_usage(ROOT)
    info["disk_free_gb"] = round(du.free / 1e9, 1)
    record("host", **info)


# ---------------------------------------------------------------- raw GETs
def raw(plan, streams, mib, secs, hashed=False):
    F = load_fetch()
    conns = F.Conns(timeout=60)
    files = [f for t in plan["trees"] for f in t["files"] if f["size"] >= mib << 20]
    ranges = [(f["url"], off, min(f["size"], off + (mib << 20))) for f in files for off in range(0, f["size"], mib << 20)]
    lock = threading.Lock()
    state = {"i": 0, "bytes": 0, "gets": 0, "errors": 0, "lat": []}
    stop = time.time() + secs

    def w():
        buf = bytearray(4 << 20)
        mv = memoryview(buf)
        while time.time() < stop:
            with lock:
                url, a, b = ranges[state["i"] % len(ranges)]
                state["i"] += 1
            t0 = time.time()
            try:
                r = conns.request("GET", url, {"Range": f"bytes={a}-{b - 1}"})
                h = hashlib.sha256() if hashed else None
                n = 0
                first = None
                while True:
                    k = r.readinto(mv)
                    if not k:
                        break
                    if first is None:
                        first = time.time() - t0
                    if h:
                        h.update(mv[:k])
                    n += k
                    with lock:
                        state["bytes"] += k
                    if time.time() > stop + 30:
                        break
                with lock:
                    state["gets"] += 1
                    state["lat"].append((first or 0, time.time() - t0))
            except Exception:  # noqa: BLE001
                conns.drop("https", urllib.parse.urlsplit(url).netloc)
                with lock:
                    state["errors"] += 1
    t = time.time()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    th = [threading.Thread(target=w, daemon=True) for _ in range(streams)]
    for x in th:
        x.start()
    for x in th:
        x.join(secs + 60)
    el = time.time() - t
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    lat = sorted(x[0] for x in state["lat"])
    tot = sorted(x[1] for x in state["lat"])
    pick = lambda a, q: round(a[min(len(a) - 1, int(q * len(a)))], 2) if a else None  # noqa: E731
    record("raw", streams=streams, mib=mib, hashed=hashed, secs=round(el, 1), MBps=round(state["bytes"] / el / 1e6), gets=state["gets"], errors=state["errors"],
           ttfb_p50=pick(lat, 0.5), ttfb_p90=pick(lat, 0.9), get_s_p50=pick(tot, 0.5), get_s_p90=pick(tot, 0.9),
           per_stream_MBps=round(state["bytes"] / el / 1e6 / streams, 1), cpu_s=round((r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime), 1))


_F = None


def load_fetch():
    global _F
    if _F is None:
        spec = importlib.util.spec_from_file_location("r2_fetch", os.path.join(HERE, "r2-fetch.py"))
        _F = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_F)
    return _F


# ---------------------------------------------------------------- full fetches
def fetch(plan_path, conc, gp, flags):
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT, exist_ok=True)
    st = os.path.join(OUT, "fetch-status.json")
    cmd = [sys.executable, os.path.join(HERE, "r2-fetch.py"), "--root", ROOT, "--plan", plan_path, "--concurrency", str(conc), "--get-parts", str(gp), "--status", st, "--min-free-gb", "1"] + flags
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = time.time()
    with open(os.path.join(OUT, f"fetch-c{conc}-g{gp}{''.join(flags).replace('-', '')}.log"), "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=lf)
    el = time.time() - t
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    s = json.load(open(st)) if os.path.exists(st) else {}
    record("fetch", concurrency=conc, get_parts=gp, flags=" ".join(flags), rc=rc, wall_s=round(el, 1), GB=round(s.get("bytes_done", 0) / 1e9, 2),
           MBps_wall=round(s.get("bytes_done", 0) / el / 1e6), MBps_fetcher=s.get("mb_per_s"), retries=s.get("retries"), corrupt=s.get("corrupt_parts"),
           user_s=round(r1.ru_utime - r0.ru_utime, 1), sys_s=round(r1.ru_stime - r0.ru_stime, 1))
    shutil.rmtree(ROOT, ignore_errors=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    trees = os.environ.get("FV_BENCH_TREES", "").split()
    log(f"bench: trees {trees}")
    plan = make_plan(trees)
    pp = os.path.join("/root", "plan.json")
    fd = os.open(pp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # presigned URLs: never in OUT (served)
    with os.fdopen(fd, "w") as f:
        json.dump(plan, f)
    record("plan", trees=trees, GB=round(sum(t["total_bytes"] for t in plan["trees"]) / 1e9, 2), files=sum(len(t["files"]) for t in plan["trees"]))
    host()
    secs = float(os.environ.get("FV_BENCH_RAW_S", "20"))
    for spec in os.environ.get("FV_BENCH_RAW", "32:64 64:64 128:64 64:256").split():
        p = spec.split(":")
        raw(plan, int(p[0]), int(p[1]), secs, hashed=len(p) > 2 and p[2] == "h")
    for spec in os.environ.get("FV_BENCH_FETCH", "32:1").split():
        p = spec.split(":")
        fetch(pp, int(p[0]), int(p[1]), [x for x in p[2:] if x])
    log("bench: done")


if __name__ == "__main__":
    try:
        main()
    finally:
        open(os.path.join(OUT, "DONE"), "w").write("1\n")
