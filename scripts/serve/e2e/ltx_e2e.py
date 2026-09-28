#!/usr/bin/env python3
"""WP-18 GPU E2E cases for an LTX-2.5 fv-serve (docs/serve/e2e/ltx.md).

    ltx_e2e.py <base url> <out dir> <case> [<case> ...]     (key: FV_KEY)

Each case runs real generations through one API and appends one JSON line to
<out dir>/results.jsonl: pass/fail, HTTP codes, client wall time, the job's
started/completed times from the native job list (queue wait vs run), the
server's stage durations where the API exposes them, and ffprobe facts of the
MP4 (frames, r_frame_rate, size, audio). MP4s go to <out dir>/mp4/.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import requests

BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
KEY = os.environ["FV_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
PROMPT = (
    "A red fox trots through fresh snow at dawn in a pine forest, its breath visible in the cold air, "
    "soft golden light, the camera tracks alongside at ground level, cinematic, shallow depth of field"
)
os.makedirs(f"{OUT}/mp4", exist_ok=True)


def ffprobe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
             "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels,duration",
             "-show_entries", "format=duration,size", "-of", "json", path],
            capture_output=True, text=True, timeout=300, check=True).stdout
        j = json.loads(out)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return {
        "codec": v.get("codec_name"), "width": v.get("width"), "height": v.get("height"),
        "frames": int(v.get("nb_read_frames", 0)), "r_frame_rate": v.get("r_frame_rate"),
        "duration_s": float(j["format"].get("duration", 0)), "bytes": int(j["format"].get("size", 0)),
        "audio": a and {"codec": a.get("codec_name"), "rate": int(a.get("sample_rate", 0)), "channels": a.get("channels")},
    }


def save(name, content):
    p = f"{OUT}/mp4/{name}.mp4"
    with open(p, "wb") as f:
        f.write(content)
    return ffprobe(p)


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def native_times(model=None):
    """started/completed of the newest native-listed job (every protocol shares the store)."""
    try:
        r = requests.get(f"{BASE}/fv/v1/jobs", headers=H, params={"limit": 1}, timeout=30).json()
        j = r["data"][0] if "data" in r else r[0]
        c, s, d = iso(j.get("created_at")), iso(j.get("started_at")), iso(j.get("completed_at"))
        return {"job": j.get("id"), "num_frames": j.get("num_frames"), "fps": j.get("fps"),
                "width": j.get("width"), "height": j.get("height"),
                "queue_s": s and c and round(s - c, 2), "run_s": d and s and round(d - s, 2)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def record(case, **kw):
    kw = {"case": case, "at": datetime.utcnow().isoformat() + "Z", **kw}
    with open(f"{OUT}/results.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def expect(want_frames=None, want_rate=None, audio=None):
    def check(p):
        ok = p.get("frames", 0) > 0
        if want_frames is not None:
            ok &= p.get("frames") == want_frames
        if want_rate is not None:
            ok &= p.get("r_frame_rate") == f"{want_rate}/1"
        if audio is not None:
            ok &= bool(p.get("audio")) == audio
        return bool(ok)
    return check


def ltx_body(seconds=6, fps=24, res="1920x1080", model="ltx-2-5-fast", **kw):
    return {"prompt": PROMPT, "model": model, "duration": seconds, "resolution": res, "fps": fps, **kw}


def ltx_v1(case, body, want):
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/text-to-video", json=body, headers=H, timeout=900)
    wall = time.time() - t0
    info = {"http": r.status_code, "x_request_id": r.headers.get("x-request-id"), "wall_s": round(wall, 2), "request": body}
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("video/mp4"):
        return record(case, api="ltx v1 sync", ok=False, body=r.text[:400], **info)
    p = save(case, r.content)
    record(case, api="ltx v1 sync", ok=want(p), mp4=p, server=native_times(), **info)


def ltx_v2(case, body, want, endpoint="text-to-video"):
    t0 = time.time()
    r = requests.post(f"{BASE}/v2/{endpoint}", json=body, headers=H, timeout=60)
    if r.status_code != 202:
        return record(case, api=f"ltx v2 {endpoint}", ok=False, http=r.status_code, body=r.text[:400], request=body)
    jid, seen = r.json()["id"], []
    while True:
        s = requests.get(f"{BASE}/v2/{endpoint}/{jid}", headers=H, timeout=30).json()
        if not seen or seen[-1] != s["status"]:
            seen.append(s["status"])
        if s["status"] in ("completed", "failed"):
            break
        if time.time() - t0 > 1500:
            return record(case, api=f"ltx v2 {endpoint}", ok=False, error="timeout", statuses=seen)
        time.sleep(2)
    done = time.time() - t0
    if s["status"] != "completed":
        return record(case, api=f"ltx v2 {endpoint}", ok=False, status=s, statuses=seen, wall_s=round(done, 2), request=body)
    v = requests.get(s["result"]["video_url"], timeout=300)  # signed URL, no key
    p = save(case, v.content)
    record(case, api=f"ltx v2 {endpoint}", ok=want(p) and v.status_code == 200, statuses=seen,
           submit_to_completed_s=round(done, 2), download_http=v.status_code,
           video_host=s["result"]["video_url"].split("/")[2], mp4=p, server=native_times(), request=body)


def ltx_error(case, method, path, body, want_http, want_type, headers=None):
    r = requests.request(method, f"{BASE}{path}", json=body, headers=headers or H, timeout=120)
    try:
        b = r.json()
    except ValueError:
        b = {"raw": r.text[:300]}
    ok = r.status_code == want_http and b.get("type") == "error" and b.get("error", {}).get("type") == want_type
    record(case, api=f"ltx {path}", ok=ok, http=r.status_code, body=b, expected=[want_http, want_type])


def ltx_upload_i2v(case):
    up = requests.post(f"{BASE}/v1/upload", headers={"Authorization": f"Bearer {KEY}"}, timeout=30)
    u = up.json()
    img = open(os.path.join(os.path.dirname(__file__), "../../gpu/fixtures/ti2v-beach-832x480.jpg"), "rb").read()
    put = requests.put(u["upload_url"], data=img, headers={"Content-Type": "image/jpeg", **u.get("required_headers", {})}, timeout=60)
    body = ltx_body(image_uri=u["storage_uri"], prompt="waves roll onto the beach")
    body["prompt"] = "waves roll onto the beach, gentle camera push in"
    v1 = requests.post(f"{BASE}/v1/image-to-video", json=body, headers=H, timeout=900)
    v2 = requests.post(f"{BASE}/v2/image-to-video", json=body, headers=H, timeout=60)
    def summ(r):
        if r.headers.get("content-type", "").startswith("video/mp4"):
            return {"http": r.status_code, "mp4": save(case, r.content)}
        return {"http": r.status_code, "body": r.text[:300]}
    record(case, api="ltx /v1/upload + PUT + image-to-video(ltx://)", upload_http=up.status_code,
           storage_uri_ok=u.get("storage_uri", "").startswith("ltx://uploads/"), put_http=put.status_code,
           v1=summ(v1), v2=summ(v2),
           ok=up.status_code == 200 and put.status_code in (200, 201, 204) and v1.status_code in (200, 400) and v2.status_code in (202, 400))


def openai_videos(case, body, want):
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/videos", json=body, headers=H, timeout=60)
    if r.status_code not in (200, 201, 202):
        return record(case, api="/v1/videos", ok=False, http=r.status_code, body=r.text[:400], request=body)
    vid = r.json()["id"]
    while True:
        s = requests.get(f"{BASE}/v1/videos/{vid}", headers=H, timeout=30).json()
        if s.get("status") in ("completed", "failed"):
            break
        if time.time() - t0 > 1500:
            return record(case, api="/v1/videos", ok=False, error="timeout")
        time.sleep(2)
    done = time.time() - t0
    c = requests.get(f"{BASE}/v1/videos/{vid}/content", headers=H, timeout=300)
    hdr = {k: v for k, v in c.headers.items() if k.lower().startswith("x-")}
    p = save(case, c.content) if c.status_code == 200 else None
    record(case, api="/v1/videos", ok=bool(p) and want(p) and s["status"] == "completed", status=s.get("status"),
           error=s.get("error"), submit_to_completed_s=round(done, 2), content_http=c.status_code, headers=hdr, mp4=p,
           server=native_times(), request=body)


def native(case, body, want):
    t0 = time.time()
    r = requests.post(f"{BASE}/fv/v1/jobs", json=body, headers=H, timeout=60)
    if r.status_code not in (200, 201, 202):
        return record(case, api="/fv/v1/jobs", ok=False, http=r.status_code, body=r.text[:400], request=body)
    jid = r.json()["id"]
    while True:
        s = requests.get(f"{BASE}/fv/v1/jobs/{jid}", headers=H, timeout=30).json()
        if s.get("status") in ("succeeded", "failed", "cancelled"):
            break
        if time.time() - t0 > 1800:
            return record(case, api="/fv/v1/jobs", ok=False, error="timeout")
        time.sleep(3)
    done = time.time() - t0
    p = None
    if s["status"] == "succeeded":
        v = requests.get(s["output"]["url"], timeout=600)
        p = save(case, v.content)
    run = iso(s.get("completed_at")) - iso(s.get("started_at")) if s.get("started_at") and s.get("completed_at") else None
    record(case, api="/fv/v1/jobs", ok=bool(p) and want(p), status=s["status"], error=s.get("error"),
           submit_to_completed_s=round(done, 2), run_s=run and round(run, 2),
           job={k: s.get(k) for k in ("id", "model", "resolved_model", "tier", "recipe", "width", "height", "num_frames", "fps")},
           output={k: v for k, v in (s.get("output") or {}).items() if k != "url"}, mp4=p, request=body)


def fal_queue(case, app, body, want):
    fh = {"Authorization": f"Key {KEY}", "Content-Type": "application/json"}
    t0 = time.time()
    r = requests.post(f"{BASE}/{app}/text-to-video", json=body, headers=fh, timeout=60)
    if r.status_code != 200:
        return record(case, api=f"fal {app}", ok=False, http=r.status_code, body=r.text[:400])
    rid = r.json()["request_id"]
    while True:
        s = requests.get(f"{BASE}/{app}/requests/{rid}/status", headers=fh, timeout=30).json()
        if s.get("status") == "COMPLETED":
            break
        if time.time() - t0 > 1500:
            return record(case, api=f"fal {app}", ok=False, error="timeout", status=s)
        time.sleep(2)
    done = time.time() - t0
    res = requests.get(f"{BASE}/{app}/requests/{rid}", headers=fh, timeout=60)
    j = res.json()
    url = (j.get("video") or {}).get("url")
    p = save(case, requests.get(url, timeout=300).content) if url else None
    record(case, api=f"fal queue {app}", ok=bool(p) and want(p), submit_to_completed_s=round(done, 2),
           status_body={k: v for k, v in s.items() if k not in ("logs",)}, result_http=res.status_code,
           result_keys=sorted(j), mp4=p, server=native_times(), request=body)


def probe(case):
    hz = requests.get(f"{BASE}/healthz", timeout=30)
    caps = requests.get(f"{BASE}/fv/v1/capabilities", headers=H, timeout=30).json()
    fal = requests.get(f"{BASE}/fal/schema", headers=H, timeout=30)
    models = [{"id": m["caps"].get("id"), "tasks": m["caps"].get("tasks"), "fps": m["caps"].get("fps"),
               "audio": m["caps"].get("audio"), "recipe": m.get("recipe")} for m in caps.get("models", [])]
    record(case, api="probe", ok=hz.status_code == 200 and bool(models), tiers=caps.get("tiers"), healthz=hz.json() if hz.ok else hz.text[:200],
           models=models, fal_schema=fal.json() if fal.ok else fal.status_code)


CASES = {
    "probe": lambda: probe("probe"),
    # warm-up + v2 at 720p
    "v2-720p-24": lambda: ltx_v2("v2-720p-24", ltx_body(res="1280x720"), expect(145, 24, True)),
    "v1-1080p-24": lambda: ltx_v1("v1-1080p-24", ltx_body(), expect(145, 24, True)),
    "v2-1080p-25": lambda: ltx_v2("v2-1080p-25", ltx_body(fps=25), expect(153, 25, True)),
    "v2-1080p-48": lambda: ltx_v2("v2-1080p-48", ltx_body(fps=48), expect(289, 48, True)),
    "v2-1080p-50": lambda: ltx_v2("v2-1080p-50", ltx_body(fps=50), expect(305, 50, True)),
    "v2-1080p-24-silent": lambda: ltx_v2("v2-1080p-24-silent", ltx_body(generate_audio=False), expect(145, 24, False)),
    "v2-1080p-20s": lambda: ltx_v2("v2-1080p-20s", ltx_body(seconds=20), expect(481, 24, True)),
    "i2v-upload": lambda: ltx_upload_i2v("i2v-upload"),
    "errors": lambda: [
        ltx_error("err-pro-unserved", "POST", "/v2/text-to-video", ltx_body(model="ltx-2-5-pro"), 403, "permission_error"),
        ltx_error("err-5s", "POST", "/v2/text-to-video", ltx_body(seconds=5), 400, "invalid_request_error"),
        ltx_error("err-50fps-20s", "POST", "/v2/text-to-video", ltx_body(seconds=20, fps=50), 400, "invalid_request_error"),
        ltx_error("err-retake", "POST", "/v2/retake", {}, 403, "permission_error"),
        ltx_error("err-auth", "POST", "/v2/text-to-video", ltx_body(), 401, "authentication_error",
                  headers={"Authorization": "Bearer wrong", "Content-Type": "application/json"}),
    ],
    "openai-1080p-5s": lambda: openai_videos("openai-1080p-5s", {"model": "ltx-turbo", "prompt": PROMPT, "seconds": 5, "size": "1920x1080"}, expect(121, 24, True)),
    "native-1080p-5s-50": lambda: native("native-1080p-5s-50", {"model": "ltx-turbo", "prompt": PROMPT, "seconds": 5, "fps": 50, "size": "1920x1080", "seed": 7}, expect(257, 50, True)),
    "fal-turbo": lambda: fal_queue("fal-turbo", "fastvideo/ltx-turbo", {"prompt": PROMPT, "seed": 3}, expect(None, None, None)),
    "fal-turbo-1080p": lambda: fal_queue("fal-turbo-1080p", "fastvideo/ltx-turbo", {"prompt": PROMPT, "seed": 3, "resolution": "1080P"}, expect(None, 24, True)),
    # ltx-pro pod
    "pro-v2-1080p-24": lambda: ltx_v2("pro-v2-1080p-24", ltx_body(model="ltx-2-5-pro"), expect(145, 24, True)),
    "pro-native-1080p-20s": lambda: native("pro-native-1080p-20s", {"model": "ltx-pro", "prompt": PROMPT, "seconds": 20, "size": "1920x1080", "seed": 7}, expect(481, 24, True)),
    "pro-fal-1080p": lambda: fal_queue("pro-fal-1080p", "fastvideo/ltx-pro", {"prompt": PROMPT, "seed": 3, "resolution": "1080P"}, expect(121, 24, True)),
    "pro-err-20s": lambda: ltx_error("pro-err-20s", "POST", "/v2/text-to-video", ltx_body(model="ltx-2-5-pro", seconds=20), 400, "invalid_request_error"),
}

for c in sys.argv[3:]:
    try:
        CASES[c]()
    except Exception as e:  # noqa: BLE001
        record(c, ok=False, exception=repr(e))
