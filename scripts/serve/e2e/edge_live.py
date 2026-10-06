#!/usr/bin/env python3
"""Stage-3 live checks of an edge cluster (docs/serve/edge-control-plane.md
§9): real clients against the edge URL only, with an h3 front (fasth3) and
an ltx front (ltx25-distill-sol) behind it.

    FV_KEY=<api key> edge_live.py <edge url> [checks...] [--latency N]

Checks: models, native, minimax, openai, ltx, fal, fal-ltx, sse, cancel,
upload, reactor (default: all of them). `--latency N`: N serial native t2v
jobs and 3N status polls, with submit, queue and completion times. Prints
one JSON line per check and a summary; exits 1 when one fails. The key is
read from FV_KEY and never printed.
"""

import json
import os
import sys
import threading
import time
import urllib.parse

import requests

BASE = sys.argv[1].rstrip("/")
ARGS = sys.argv[2:]
LAT = 0
if "--latency" in ARGS:
    i = ARGS.index("--latency")
    LAT = int(ARGS[i + 1])
    del ARGS[i : i + 2]
KEY = os.environ["FV_KEY"]
H = {"Authorization": f"Bearer {KEY}"}
HJ = {**H, "Content-Type": "application/json"}
HOST = urllib.parse.urlparse(BASE).netloc
os.environ["FAL_KEY"] = KEY
os.environ["FAL_QUEUE_RUN_HOST"] = HOST
os.environ["FAL_RUN_HOST"] = f"{HOST}/run"
PROMPT = "A paper lantern drifts over a night river, soft reflections"
results = []
lock = threading.Lock()


def mp4(b):
    return len(b) > 1000 and b[4:8] == b"ftyp"


def poll(fn, done, timeout=900, every=2.0):
    t0 = time.monotonic()
    while True:
        v = fn()
        if done(v):
            return v
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"not done after {timeout}s: {str(v)[:300]}")
        time.sleep(every)


def check(name):
    def wrap(fn):
        def run():
            t0 = time.monotonic()
            rec = {"check": name}
            try:
                rec.update(fn() or {})
                rec["ok"] = True
            except Exception as e:  # noqa: BLE001 - every failure is a result
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            rec["wall_s"] = round(time.monotonic() - t0, 2)
            with lock:
                results.append(rec)
                print(json.dumps(rec), flush=True)

        run.check = name
        return run

    return wrap


def download(url, auth=False):
    r = requests.get(url, headers=H if auth else {}, timeout=120)
    assert r.status_code == 200 and mp4(r.content), f"download {r.status_code} {len(r.content)} bytes from {url[:80]}"
    return len(r.content)


@check("models")
def models():
    ids = [m["id"] for m in requests.get(f"{BASE}/v1/models", headers=H, timeout=30).json()["data"]]
    assert "fasth3" in ids and "ltx25-distill-sol" in ids, ids
    caps = requests.get(f"{BASE}/fv/v1/capabilities", headers=H, timeout=30)
    assert caps.status_code == 200, caps.status_code
    st = requests.get(f"{BASE}/fv/v1/status", timeout=30).json()
    return {"models": len(ids), "pools": sorted(p["id"] for p in st["pools"]), "state": st["state"]}


def native_job(model, extra=None):
    t0 = time.monotonic()
    r = requests.post(f"{BASE}/fv/v1/jobs", headers=HJ, json={"model": model, "prompt": PROMPT, **(extra or {})}, timeout=60)
    assert r.status_code in (200, 201, 202), f"submit {r.status_code} {r.text[:300]}"
    submit_s = time.monotonic() - t0
    jid = r.json()["id"]
    j = poll(lambda: requests.get(f"{BASE}/fv/v1/jobs/{jid}", headers=H, timeout=30).json(), lambda j: j["status"] in ("succeeded", "failed", "cancelled"))
    assert j["status"] == "succeeded", j
    return jid, j, submit_s, time.monotonic() - t0


@check("native")
def native():
    out = {}
    # Both fronts at once: h3 and ltx.
    res = {}

    def one(m):
        res[m] = native_job(m)

    ts = [threading.Thread(target=one, args=(m,)) for m in ("fasth3", "ltx25-distill-sol")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for m in ("fasth3", "ltx25-distill-sol"):
        assert m in res, f"{m} did not finish"
        jid, j, sub, tot = res[m]
        url = (j.get("output") or {}).get("url") or (j.get("output") or {}).get("video_url")
        assert url, j
        out[m] = {"submit_s": round(sub, 2), "done_s": round(tot, 1), "queue_s": j.get("metrics", {}).get("queue_s"), "bytes": download(url)}
    return out


@check("minimax")
def minimax():
    r = requests.post(f"{BASE}/v2/video_generation", headers=HJ, json={"model": "MiniMax-H3", "content": [{"type": "text", "text": PROMPT}], "resolution": "768P", "duration": 5, "ratio": "16:9"}, timeout=60)
    assert r.status_code == 200, f"create {r.status_code} {r.text[:300]}"
    tid = r.json()["task_id"]
    t = poll(lambda: requests.get(f"{BASE}/v2/query/video_generation/{tid}", headers=H, timeout=30).json()["task"], lambda t: t["status"] in ("succeeded", "failed"))
    assert t["status"] == "succeeded", t
    return {"bytes": download(t["content"]["url"])}


@check("openai")
def openai_videos():
    from openai import OpenAI

    c = OpenAI(base_url=f"{BASE}/v1", api_key=KEY, max_retries=0)
    v = c.videos.create(model="h3-turbo", prompt=PROMPT, seconds="8", size="1344x768")
    v = poll(lambda: c.videos.retrieve(v.id), lambda v: v.status in ("completed", "failed"))
    assert v.status == "completed", v
    body = c.videos.download_content(v.id).read()
    assert mp4(body), len(body)
    return {"bytes": len(body)}


@check("ltx")
def ltx():
    r = requests.post(f"{BASE}/v2/text-to-video", headers=HJ, json={"prompt": PROMPT, "model": "ltx-2-5-fast", "duration": 6, "resolution": "1280x720"}, timeout=60)
    assert r.status_code == 202, f"submit {r.status_code} {r.text[:300]}"
    jid = r.json()["id"]
    st = poll(lambda: requests.get(f"{BASE}/v2/text-to-video/{jid}", headers=H, timeout=30).json(), lambda j: j["status"] in ("completed", "failed"))
    assert st["status"] == "completed", st
    return {"bytes": download(st["result"]["video_url"])}


@check("fal")
def fal():
    import fal_client

    logs = []
    t0 = time.monotonic()
    r = fal_client.subscribe("minimax/h3-turbo/text-to-video", arguments={"prompt": PROMPT, "resolution": "768P", "duration": 5, "seed": 11}, with_logs=True, on_queue_update=lambda u: logs.append(type(u).__name__))
    wall = time.monotonic() - t0
    url = r["video"]["url"]
    return {"subscribe_s": round(wall, 1), "updates": sorted(set(logs)), "bytes": download(url)}


@check("fal-ltx")
def fal_ltx():
    """Submit, then status and the result by id: the result answers from the edge whichever front holds it."""
    import fal_client

    h = fal_client.submit("fastvideo/ltx-turbo/text-to-video", arguments={"prompt": PROMPT, "duration": 6})
    poll(lambda: h.status(), lambda s: type(s).__name__ == "Completed", every=1.1)
    r = h.get()
    url = r["video"]["url"]
    return {"request_id": h.request_id, "bytes": download(url)}


@check("sse")
def sse():
    """fal status over SSE until COMPLETED."""
    r = requests.post(f"https://{HOST}/minimax/h3-turbo/text-to-video", headers={"Authorization": f"Key {KEY}", "Content-Type": "application/json"}, json={"prompt": PROMPT, "resolution": "768P", "duration": 5}, timeout=60)
    assert r.status_code == 200, f"submit {r.status_code} {r.text[:200]}"
    body = r.json()
    t0 = time.monotonic()
    seen = []
    with requests.get(f"{body['status_url']}/stream", headers={"Authorization": f"Key {KEY}"}, stream=True, timeout=600) as s:
        assert s.status_code == 200, s.status_code
        for line in s.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                st = json.loads(line[5:]).get("status")
                if not seen or seen[-1] != st:
                    seen.append(st)
                if st == "COMPLETED":
                    break
    assert seen and seen[-1] == "COMPLETED", seen
    res = requests.get(body["response_url"], headers={"Authorization": f"Key {KEY}"}, timeout=60).json()
    return {"statuses": seen, "completed_s": round(time.monotonic() - t0, 1), "bytes": download(res["video"]["url"])}


@check("cancel")
def cancel():
    jobs = [requests.post(f"{BASE}/fv/v1/jobs", headers=HJ, json={"model": "fasth3", "prompt": f"{PROMPT} {i}"}, timeout=60).json()["id"] for i in range(3)]
    # The last of three on one GPU is still queued: cancel it.
    last = jobs[-1]
    r = requests.delete(f"{BASE}/fv/v1/jobs/{last}", headers=H, timeout=30)
    assert r.status_code in (200, 202, 204), f"cancel {r.status_code} {r.text[:200]}"
    j = poll(lambda: requests.get(f"{BASE}/fv/v1/jobs/{last}", headers=H, timeout=30).json(), lambda j: j["status"] in ("succeeded", "failed", "cancelled"))
    assert j["status"] == "cancelled", j
    for jid in jobs[:-1]:
        poll(lambda: requests.get(f"{BASE}/fv/v1/jobs/{jid}", headers=H, timeout=30).json(), lambda j: j["status"] in ("succeeded", "failed", "cancelled"))
    return {"cancelled": last}


def tiny_png(w=16, h=16):
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes((40, 90, 200)) * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


@check("upload")
def upload():
    """fal storage upload through the edge, read back through the edge."""
    png = tiny_png()
    # What fal clients do: initiate (the edge forwards it to a front), PUT, then read the file URL.
    r = requests.post(f"{BASE}/storage/upload/initiate", headers={"Authorization": f"Key {KEY}", "Content-Type": "application/json"}, json={"content_type": "image/png", "file_name": "edge.png"}, timeout=30)
    assert r.status_code == 200, f"initiate {r.status_code} {r.text[:200]}"
    up = r.json()
    p = requests.put(up["upload_url"], data=png, headers={"Content-Type": "image/png"}, timeout=60)
    assert p.status_code in (200, 201, 204), f"put {p.status_code} {p.text[:200]}"
    url = up["file_url"]
    g = requests.get(url, timeout=60)
    assert g.status_code == 200 and g.content[:4] == b"\x89PNG", f"read back {g.status_code}"
    return {"url_host": urllib.parse.urlparse(url).netloc}


@check("reactor")
def reactor():
    r = requests.post(f"{BASE}/start_session", headers=HJ, json={}, timeout=60)
    assert r.status_code == 200, f"start {r.status_code} {r.text[:300]}"
    s = requests.get(f"{BASE}/session", headers=H, timeout=30)
    assert s.status_code == 200, f"session {s.status_code}"
    stop = requests.post(f"{BASE}/stop_session", headers=HJ, json={"reason": "edge live test"}, timeout=30)
    assert stop.status_code == 200, f"stop {stop.status_code}"
    return {"session": r.json().get("session_id"), "transport": (r.json().get("selected_transport") or {}).get("protocol")}


def latency(n):
    """n serial t2v jobs (fasth3): submit time, queue time, time to succeeded; 3n status polls."""
    subs, queues, dones, polls = [], [], [], []
    last = None
    for i in range(n):
        jid, j, sub, tot = native_job("fasth3", {"seed": 100 + i})
        subs.append(sub)
        queues.append((j.get("metrics") or {}).get("queue_s") or 0)
        dones.append(tot)
        last = jid
    for _ in range(3 * n):
        t0 = time.monotonic()
        assert requests.get(f"{BASE}/fv/v1/jobs/{last}", headers=H, timeout=30).status_code == 200
        polls.append(time.monotonic() - t0)

    def q(v):
        v = sorted(v)
        return {"p50": round(v[len(v) // 2], 3), "max": round(v[-1], 3)}

    rec = {"check": "latency", "jobs": n, "submit_s": q(subs), "queue_s": q(queues), "done_s": q(dones), "poll_s": q(polls), "ok": True}
    results.append(rec)
    print(json.dumps(rec), flush=True)


ALL = [models, native, minimax, openai_videos, ltx, fal, fal_ltx, sse, cancel, upload, reactor]
chosen = [c for c in ALL if not ARGS or c.check in ARGS]
for c in chosen:
    c()
if LAT:
    latency(LAT)
bad = [r["check"] for r in results if not r["ok"]]
print(json.dumps({"summary": {"passed": len(results) - len(bad), "failed": bad}}))
sys.exit(1 if bad else 0)
