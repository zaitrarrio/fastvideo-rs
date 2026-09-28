#!/usr/bin/env python3
"""WP-18 GPU E2E, batch APIs (docs/serve/design.md §7.6): real generations
through every batch API of one fv-serve pod, one JSON record per check.

    batch.py --base https://<pod>-8000.proxy.runpod.net --key <api key> \
        --app minimax/h3-turbo --model fasth3 --out artifacts/serve/e2e/h3-turbo \
        [--only fal.t2v,minimax] [--sidecar https://<pod>-8001.proxy.runpod.net]

The sidecar token comes from FV_SIDECAR_TOKEN. Webhooks and MiniMax
callbacks go to the sidecar on the pod (http://127.0.0.1:8001/hook/...;
fv-serve runs with FV_CALLBACKS_ALLOW_PRIVATE=1) and are read back from
its `/hooks`. Clients: requests, openai 3.6.0, PyNaCl (tests/compat venv).
Writes <out>/batch.json (appending to earlier records of other checks) and
at most two sample MP4s under 5 MB to <out>/samples/.
"""

import argparse
import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

import requests
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "scripts/gpu/fixtures/ti2v-beach-832x480.jpg"
PROMPT = "A red fox trots through fresh snow at dawn, its breath visible in the cold air, cinematic"
RES = os.environ.get("FV_E2E_RES", "480P")  # the fake engine is 768P only
HOOK = "http://127.0.0.1:8001/hook"
TERMINAL_FAL = {"COMPLETED"}

a = None
RESULTS = []
SAMPLES = []


def log(*m):
    print(*m, file=sys.stderr, flush=True)


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def rec(name, fn):
    if a.only and not any(name.startswith(o) for o in a.only):
        return None
    log(f"--- {name}")
    t0 = time.monotonic()
    try:
        detail = fn() or {}
        ok = True
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        detail, ok = {"error": f"{type(e).__name__}: {e}"[:1500]}, False
    r = {"check": name, "ok": ok, "wall_s": round(time.monotonic() - t0, 2), **detail}
    RESULTS.append(r)
    log(("PASS " if ok else "FAIL ") + json.dumps(r)[:600])
    return r


def fal_h():
    return {"Authorization": f"Key {a.key}"}


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,profile,width,height,nb_frames,r_frame_rate,sample_rate,channels:format=duration,size",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), {})
    au = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return {
        "video": f'{v.get("codec_name")}/{v.get("profile")} {v.get("width")}x{v.get("height")} {v.get("nb_frames")}f @{v.get("r_frame_rate")}',
        "audio": f'{au["codec_name"]} {au["sample_rate"]}Hz {au["channels"]}ch' if au else None,
        "duration_s": float(j["format"].get("duration", 0)),
        "bytes": int(j["format"].get("size", 0)),
        "width": v.get("width"), "height": v.get("height"),
    }


def download(url, name, headers=None, keep=False):
    t0 = time.monotonic()
    r = requests.get(url, headers=headers or {}, timeout=300)
    r.raise_for_status()
    dl = time.monotonic() - t0
    tmp = pathlib.Path(a.out) / "tmp" / f"{name}.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(r.content)
    check(r.content[4:8] == b"ftyp", f"{name}: not an MP4")
    p = ffprobe(tmp)
    p["download_s"] = round(dl, 2)
    p["host"] = url.split("/")[2]
    if keep and len(SAMPLES) < 2 and len(r.content) < 5 * 1024 * 1024:
        dst = pathlib.Path(a.out) / "samples" / f"{name}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(r.content)
        SAMPLES.append(os.path.relpath(dst, ROOT))
        p["sample"] = SAMPLES[-1]
    tmp.unlink()
    return p


# ------------------------------------------------------------------ fal

def fal_submit(sub, body, params=None):
    t0 = time.monotonic()
    r = requests.post(f"{a.base}/{a.app}/{sub}", headers=fal_h(), json=body, params=params or {}, timeout=60)
    return r, time.monotonic() - t0


def fal_status(rid, logs=False):
    r = requests.get(f"{a.base}/{a.app}/requests/{rid}/status", headers=fal_h(), params={"logs": int(logs)}, timeout=30)
    r.raise_for_status()
    return r.json()


def fal_wait(rid, timeout=900):
    t0 = time.monotonic()
    seen, first_progress = [], None
    while True:
        s = fal_status(rid)
        if not seen or seen[-1] != s["status"]:
            seen.append(s["status"])
            if s["status"] == "IN_PROGRESS" and first_progress is None:
                first_progress = time.monotonic() - t0
        if s["status"] in TERMINAL_FAL:
            return s, seen, first_progress
        check(time.monotonic() - t0 < timeout, f"fal {rid}: timeout; statuses {seen}")
        time.sleep(1)


def fal_result(rid):
    r = requests.get(f"{a.base}/{a.app}/requests/{rid}", headers=fal_h(), timeout=60)
    return r


def fal_run_job(sub, body, name, keep=False, params=None):
    r, submit_s = fal_submit(sub, body, params)
    check(r.status_code == 200, f"submit {r.status_code} {r.text[:300]}")
    sub_j = r.json()
    rid = sub_j["request_id"]
    check({"request_id", "response_url", "status_url", "cancel_url"} <= set(sub_j), f"submit body {sub_j}")
    st, seen, fp = fal_wait(rid)
    st_logs = fal_status(rid, logs=True)
    res = fal_result(rid)
    check(res.status_code == 200, f"result {res.status_code} {res.text[:400]}")
    j = res.json()
    url = j["video"]["url"]
    probe = download(url, name, keep=keep)
    return {"request_id": rid, "submit_s": round(submit_s, 2), "statuses": seen,
            "to_in_progress_s": round(fp, 2) if fp else None, "fal_timings": j.get("timings") or st.get("metrics"),
            "status_metrics": st.get("metrics"), "logs_n": len(st_logs.get("logs") or []),
            "seed": j.get("seed"), "mp4": probe}


def t_fal_t2v_480():
    d = fal_run_job("text-to-video", {"prompt": PROMPT, "resolution": RES, "duration": 5, "aspect_ratio": "16:9", "seed": 1},
                    "fal-t2v-480p", keep=True)
    check(d["mp4"]["height"] == 480 or d["mp4"]["width"] == 480 or min(d["mp4"]["width"], d["mp4"]["height"]) == 480, d["mp4"])
    check(d["logs_n"] >= 0, "logs")
    return d


def t_fal_t2v_768():
    d = fal_run_job("text-to-video", {"prompt": PROMPT, "duration": 5, "seed": 2}, "fal-t2v-768p", keep=True)
    check(min(d["mp4"]["width"], d["mp4"]["height"]) == 768, d["mp4"])
    return d


STATE = {}


def t_fal_upload():
    t0 = time.monotonic()
    r = requests.post(f"{a.base}/storage/upload/initiate", headers=fal_h(), params={"storage_type": "fal-cdn-v3"},
                      json={"content_type": "image/jpeg", "file_name": "beach.jpg"}, timeout=30)
    check(r.status_code == 200, f"initiate {r.status_code} {r.text[:300]}")
    j = r.json()
    data = FIXTURE.read_bytes()
    p = requests.put(j["upload_url"], data=data, headers={"content-type": "image/jpeg"}, timeout=60)
    check(p.status_code in (200, 201, 204), f"PUT {p.status_code} {p.text[:200]}")
    g = requests.get(j["file_url"], timeout=60)
    check(g.status_code == 200 and g.content == data, f"GET file_url {g.status_code} {len(g.content)}")
    STATE["file_url"] = j["file_url"]
    return {"upload_host": j["upload_url"].split("/")[2], "file_host": j["file_url"].split("/")[2],
            "bytes": len(data), "roundtrip_s": round(time.monotonic() - t0, 2)}


def t_fal_i2v():
    check("file_url" in STATE, "no uploaded file (fal.upload failed)")
    return fal_run_job("image-to-video", {"prompt": "The waves roll onto the beach as the camera slowly pushes in",
                                          "image_url": STATE["file_url"], "resolution": RES, "duration": 5, "seed": 1},
                       "fal-i2v-480p")


def t_fal_r2v():
    img = STATE.get("file_url") or "https://raw.githubusercontent.com/zaitrarrio/fastvideo-rs/main/scripts/gpu/fixtures/ti2v-beach-832x480.jpg"
    r, _ = fal_submit("reference-to-video", {"prompt": "Image 1 at sunset", "reference_image_urls": [img], "resolution": RES, "duration": 5})
    if 400 <= r.status_code < 500:
        body = r.json()
        check("detail" in body, f"4xx without fal detail: {r.text[:300]}")
        return {"refused": r.status_code, "detail": body["detail"]}
    check(r.status_code == 200, f"r2v {r.status_code} {r.text[:300]}")
    rid = r.json()["request_id"]
    st, seen, _ = fal_wait(rid)
    res = fal_result(rid)
    out = {"accepted": True, "statuses": seen, "result_status": res.status_code, "result": res.text[:400]}
    check(res.status_code == 200 or 400 <= res.status_code < 500, out)
    return out


def t_fal_cancel():
    body = {"prompt": PROMPT, "resolution": RES, "duration": 5, "seed": 5}
    ra, _ = fal_submit("text-to-video", body)
    rb, _ = fal_submit("text-to-video", body)
    check(ra.status_code == 200 and rb.status_code == 200, "submits")
    A, B = ra.json()["request_id"], rb.json()["request_id"]
    c = requests.put(f"{a.base}/{a.app}/requests/{B}/cancel", headers=fal_h(), timeout=30)
    cb = {"code": c.status_code, "body": c.text[:200]}
    sb = fal_status(B)
    # cancel the running one too (it ends at the next denoise step)
    time.sleep(2)
    c2 = requests.put(f"{a.base}/{a.app}/requests/{A}/cancel", headers=fal_h(), timeout=30)
    sa, seen_a, _ = fal_wait(A)
    ra2 = fal_result(A)
    rb2 = fal_result(B)
    # cancel after completion
    c3 = requests.put(f"{a.base}/{a.app}/requests/{A}/cancel", headers=fal_h(), timeout=30)
    out = {"queued_cancel": cb, "queued_status": sb.get("status"), "queued_error_type": sb.get("error_type"),
           "running_cancel": {"code": c2.status_code, "body": c2.text[:200]}, "running_final": sa.get("status"),
           "running_error_type": sa.get("error_type"), "running_result": ra2.status_code, "queued_result": rb2.status_code,
           "queued_result_body": rb2.text[:200], "after_done_cancel": {"code": c3.status_code, "body": c3.text[:200]}}
    check(c.status_code == 202, out)
    check(sb.get("status") == "COMPLETED" and sb.get("error_type") == "client_cancelled", out)
    check(c3.status_code == 400, out)
    return out


def t_fal_sync():
    t0 = time.monotonic()
    r = requests.post(f"{a.base}/run/{a.app}/text-to-video", headers=fal_h(),
                      json={"prompt": PROMPT, "resolution": RES, "duration": 5, "seed": 3}, timeout=110)
    wall = time.monotonic() - t0
    check(r.status_code == 200, f"sync {r.status_code} {r.text[:300]}")
    j = r.json()
    return {"sync_wall_s": round(wall, 2), "timings": j.get("timings"), "mp4": download(j["video"]["url"], "fal-sync")}


def b64url(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify_fal(keys, h, body):
    rid, uid, ts, sig = (h.get(k) for k in ("x-fal-webhook-request-id", "x-fal-webhook-user-id", "x-fal-webhook-timestamp", "x-fal-webhook-signature"))
    if not all((rid, uid, ts, sig)) or abs(int(time.time()) - int(ts)) > 300:
        return False
    msg = "\n".join([rid, uid, ts, hashlib.sha256(body).hexdigest()]).encode()
    for k in keys:
        if k.get("kty") == "OKP" and k.get("crv") == "Ed25519":
            try:
                VerifyKey(b64url(k["x"])).verify(msg, bytes.fromhex(sig))
                return True
            except (BadSignatureError, ValueError):
                pass
    return False


def hooks():
    r = requests.get(f"{a.sidecar}/hooks", headers={"X-Sidecar-Token": os.environ["FV_SIDECAR_TOKEN"]}, timeout=30)
    r.raise_for_status()
    return [(x["path"], x["headers"], base64.b64decode(x["body_b64"]), x["t"]) for x in r.json()]


def wait_hooks(pred, timeout=600):
    t0 = time.monotonic()
    while True:
        g = hooks()
        if pred(g):
            return g
        check(time.monotonic() - t0 < timeout, f"hooks: timeout; paths {[p for p, *_ in g]}")
        time.sleep(2)


def t_fal_webhook():
    keys = requests.get(f"{a.base}/.well-known/jwks.json", timeout=30).json()["keys"]
    tag = f"/hook/fal-{int(time.time())}"
    t0 = time.time()
    r, _ = fal_submit("text-to-video", {"prompt": PROMPT, "resolution": RES, "duration": 5, "seed": 4}, params={"fal_webhook": HOOK[:-5] + tag})
    check(r.status_code == 200, f"submit {r.status_code} {r.text[:300]}")
    rid = r.json()["request_id"]
    g = wait_hooks(lambda g: any(p == tag for p, *_ in g))
    _, h, raw, t = next(x for x in g if x[0] == tag)
    body = json.loads(raw)
    ok_sig = verify_fal(keys, h, raw)
    tampered = verify_fal(keys, h, raw.replace(b'"OK"', b'"ERROR"'))
    out = {"request_id": rid, "delivery_after_submit_s": round(t - t0, 2), "status": body.get("status"),
           "signature_ok": ok_sig, "tampered_rejected": not tampered, "jwks_keys": len(keys),
           "payload_video": bool(body.get("payload", {}).get("video", {}).get("url"))}
    check(ok_sig and not tampered and body.get("request_id") == rid and body.get("status") == "OK", out)
    return out


def t_fal_errors():
    r, _ = fal_submit("text-to-video", {"prompt": "x", "resolution": "4K"})
    r2 = requests.post(f"{a.base}/{a.app}/text-to-video", json={"prompt": "x"}, timeout=30)
    out = {"bad_resolution": r.status_code, "detail": r.json().get("detail"), "no_key": r2.status_code}
    check(r.status_code == 422 and r2.status_code == 401, out)
    return out


def t_fal_unserved():
    """A configured fal app whose tier this process does not serve."""
    other = "minimax/h3-max" if a.app != "minimax/h3-max" else "minimax/h3-turbo"
    r = requests.post(f"{a.base}/{other}/text-to-video", headers=fal_h(), json={"prompt": "x", "resolution": RES}, timeout=30)
    out = {"app": other, "submit": r.status_code, "body": r.text[:300]}
    if r.status_code == 200:
        rid = r.json()["request_id"]
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            s = requests.get(f"{a.base}/{other}/requests/{rid}/status", headers=fal_h(), timeout=30).json()
            if s["status"] == "COMPLETED":
                break
            time.sleep(1)
        res = requests.get(f"{a.base}/{other}/requests/{rid}", headers=fal_h(), timeout=30)
        out.update(result=res.status_code, result_body=res.text[:300])
    check(400 <= r.status_code < 500 or out.get("result") == 200 or 400 <= out.get("result", 0) < 500, out)
    return out


# -------------------------------------------------------------- MiniMax

def mm_h():
    return {"Authorization": f"Bearer {a.key}", "Content-Type": "application/json"}


def t_minimax():
    tag = f"/hook/minimax-{int(time.time())}"
    t0 = time.monotonic()
    r = requests.post(f"{a.base}/v2/video_generation", headers=mm_h(), timeout=60, json={
        "model": "MiniMax-H3-Turbo", "content": [{"type": "text", "text": PROMPT}],
        "resolution": RES, "duration": 5, "ratio": "16:9", "callback_url": HOOK[:-5] + tag})
    check(r.status_code == 200 and set(r.json()) == {"task_id"}, f"create {r.status_code} {r.text[:300]}")
    tid = r.json()["task_id"]
    seen = []
    while True:
        q = requests.get(f"{a.base}/v2/query/video_generation/{tid}", headers=mm_h(), timeout=30)
        q.raise_for_status()
        task = q.json()["task"]
        if not seen or seen[-1] != task["status"]:
            seen.append(task["status"])
        if task["status"] in ("succeeded", "failed", "cancelled"):
            break
        check(time.monotonic() - t0 < 900, f"timeout {seen}")
        time.sleep(1)
    wall = time.monotonic() - t0
    check(task["status"] == "succeeded", task)
    probe = download(task["content"]["url"], "minimax")
    lst = requests.get(f"{a.base}/v2/query/video_generation", headers=mm_h(), params={"page_num": 1, "page_size": 10}, timeout=30)
    ids = [t["id"] for t in lst.json().get("items", [])]
    g = wait_hooks(lambda g: any(p == tag and json.loads(b).get("task", {}).get("status") == "succeeded" for p, _, b, _ in g), timeout=120)
    bodies = [json.loads(b) for p, _, b, _ in g if p == tag]
    cb_statuses = [b["task"]["status"] for b in bodies if "task" in b]
    out = {"task_id": tid, "wall_s_to_succeeded": round(wall, 2), "statuses": seen, "usage": task.get("usage"),
           "mp4": probe, "list_status": lst.status_code, "listed": tid in ids,
           "callback_challenge_first": "challenge" in bodies[0], "callback_statuses": cb_statuses}
    check(lst.status_code == 200 and tid in ids, out)
    check("challenge" in bodies[0] and cb_statuses[-1] == "succeeded", out)
    e = requests.post(f"{a.base}/v2/video_generation", headers=mm_h(), json={"model": "MiniMax-H3-Turbo", "content": [], "resolution": RES, "duration": 5, "ratio": "16:9"}, timeout=30)
    out["error_2013"] = e.status_code == 400 and e.json()["error"]["message"].endswith("(2013)")
    check(out["error_2013"], e.text[:300])
    return out


# ------------------------------------------------------------ FastVideo

def t_openai():
    import openai
    from openai import OpenAI
    c = OpenAI(base_url=f"{a.base}/v1", api_key=a.key, max_retries=0, timeout=120)
    ids = [m.id for m in c.models.list()]
    check(a.model in ids, f"models {ids}")
    t0 = time.monotonic()
    v = c.videos.create_and_poll(model=a.model, prompt=PROMPT, seconds="5", size="832x480", poll_interval_ms=1000)
    wall = time.monotonic() - t0
    check(v.status == "completed", f"{v.status} {v.error}")
    t1 = time.monotonic()
    body = c.videos.download_content(v.id).read()
    dl = time.monotonic() - t1
    tmp = pathlib.Path(a.out) / "tmp" / "openai.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(body)
    p = ffprobe(tmp)
    tmp.unlink()
    return {"openai": openai.__version__, "models": ids, "video_id": v.id, "create_and_poll_s": round(wall, 2),
            "download_s": round(dl, 2), "size": v.size, "seconds": v.seconds, "mp4": p}


# --------------------------------------------------------------- native

def t_native():
    h = {"Authorization": f"Bearer {a.key}"}
    caps = requests.get(f"{a.base}/fv/v1/capabilities", headers=h, timeout=30)
    check(caps.status_code == 200, caps.text[:300])
    cj = caps.json()
    t0 = time.monotonic()
    r = requests.post(f"{a.base}/fv/v1/jobs", headers=h, timeout=60, json={
        "model": a.model, "prompt": PROMPT, "aspect_ratio": "16:9", "short_edge": int(RES[:-1]), "seconds": 5, "seed": 7})
    check(r.status_code in (200, 201, 202), f"{r.status_code} {r.text[:300]}")
    jid = r.json()["id"]
    while True:
        s = requests.get(f"{a.base}/fv/v1/jobs/{jid}", headers=h, timeout=30).json()
        if s["status"] in ("succeeded", "failed", "cancelled"):
            break
        check(time.monotonic() - t0 < 900, s)
        time.sleep(1)
    wall = time.monotonic() - t0
    check(s["status"] == "succeeded", s)
    c = requests.get(f"{a.base}/fv/v1/jobs/{jid}/content", headers=h, allow_redirects=False, timeout=30)
    loc = c.headers.get("location")
    probe = download(loc, "native") if loc else None
    return {"models": [m.get("id") for m in cj.get("models", [])], "job": jid, "wall_s": round(wall, 2),
            "content_status": c.status_code, "timings": s.get("timings") or s.get("phases"), "mp4": probe,
            "resolved": {k: s.get(k) for k in ("width", "height", "num_frames", "fps")}}


def t_health():
    out = {}
    for p in ("/health", "/healthz", "/ping", "/metrics", "/"):
        r = requests.get(a.base + p, timeout=30)
        out[p] = r.status_code
    check(all(v == 200 for v in out.values()), out)
    return out


def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default=os.environ.get("FV_KEY"))
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--app", default="minimax/h3-turbo")
    ap.add_argument("--model", default="fasth3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    a.only = [x for x in a.only.split(",") if x]
    rec("health", t_health)
    rec("fal.t2v.480p", t_fal_t2v_480)
    rec("fal.upload", t_fal_upload)
    rec("fal.i2v.480p", t_fal_i2v)
    rec("fal.r2v", t_fal_r2v)
    rec("fal.cancel", t_fal_cancel)
    rec("fal.sync", t_fal_sync)
    rec("fal.webhook", t_fal_webhook)
    rec("fal.errors", t_fal_errors)
    rec("fal.unserved-app", t_fal_unserved)
    rec("minimax", t_minimax)
    rec("openai", t_openai)
    rec("native", t_native)
    rec("fal.t2v.768p", t_fal_t2v_768)
    out = pathlib.Path(a.out) / "batch.json"
    old = json.loads(out.read_text()) if out.exists() else []
    names = {r["check"] for r in RESULTS}
    merged = [r for r in old if r["check"] not in names] + RESULTS
    out.write_text(json.dumps(merged, indent=1) + "\n")
    log(f"wrote {out}: {sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(r["ok"] for r in RESULTS) else 1)


if __name__ == "__main__":
    main()
