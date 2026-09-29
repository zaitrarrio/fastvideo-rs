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
    """started/completed of the newest job in the native listing. That listing holds
    native-protocol jobs only, so this describes a native job, never an LTX, fal or
    /v1/videos one (an earlier run attached it to those and reported another job's
    dims); only native() cases use it now."""
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
    record(case, api="ltx v1 sync", ok=want(p), mp4=p, **info)


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
           video_host=s["result"]["video_url"].split("/")[2], mp4=p, request=body)


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
    # v2 first, then v1 sync, then v2 again: the same upload serves all three.
    v2a = requests.post(f"{BASE}/v2/image-to-video", json=body, headers=H, timeout=60)
    v1 = requests.post(f"{BASE}/v1/image-to-video", json=body, headers=H, timeout=900)
    v2b = requests.post(f"{BASE}/v2/image-to-video", json=body, headers=H, timeout=60)
    def summ(r):
        hdr = {k: v for k, v in r.headers.items() if k.lower() in ("x-request-id", "server", "content-type", "cf-ray")}
        if r.headers.get("content-type", "").startswith("video/mp4"):
            return {"http": r.status_code, "headers": hdr, "mp4": save(case, r.content)}
        return {"http": r.status_code, "headers": hdr, "body": r.text[:300]}
    record(case, api="ltx /v1/upload + PUT + image-to-video(ltx://)", upload_http=up.status_code,
           storage_uri_ok=u.get("storage_uri", "").startswith("ltx://uploads/"), put_http=put.status_code,
           v2_first=summ(v2a), v1=summ(v1), v2_after_v1=summ(v2b),
           ok=up.status_code == 200 and put.status_code in (200, 201, 204) and v1.status_code == 200
           and v2a.status_code == 202 and v2b.status_code == 202)


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
           request=body)


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


def fal_queue(case, app, body, want, sub="text-to-video"):
    fh = {"Authorization": f"Key {KEY}", "Content-Type": "application/json"}
    t0 = time.time()
    r = requests.post(f"{BASE}/{app}/{sub}", json=body, headers=fh, timeout=60)
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
           result_keys=sorted(j), mp4=p, request=body)


def probe(case):
    hz = requests.get(f"{BASE}/healthz", timeout=30)
    caps = requests.get(f"{BASE}/fv/v1/capabilities", headers=H, timeout=30).json()
    fal = requests.get(f"{BASE}/fal/schema", headers=H, timeout=30)
    models = [{"id": m["caps"].get("id"), "tasks": m["caps"].get("tasks"), "fps": m["caps"].get("fps"),
               "audio": m["caps"].get("audio"), "recipe": m.get("recipe")} for m in caps.get("models", [])]
    record(case, api="probe", ok=hz.status_code == 200 and bool(models), tiers=caps.get("tiers"), healthz=hz.json() if hz.ok else hz.text[:200],
           models=models, fal_schema=fal.json() if fal.ok else fal.status_code)


FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../gpu/fixtures")
BEACH = os.path.join(FIXTURES, "ti2v-beach-832x480.jpg")
BEACH_ZOOM = os.path.join(FIXTURES, "ti2v-beach-zoom-832x480.jpg")
I2V_PROMPT = ("Aerial drone shot of a tropical beach: turquoise sea waves roll in and break into white foam "
              "on the sand, the camera glides slowly forward along the shoreline, bright sunny day.")


def data_uri(path):
    import base64
    mime = {"png": "image/png", "flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg", "mp4": "video/mp4"}.get(
        path.rsplit(".", 1)[-1], "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


# Audio-to-video (docs/oracle.md "LTX-2.5 audio-to-video"): the oracle's 7 s speech
# clip (44.1 kHz stereo FLAC). The clip is the longest 8k+1 that fits in the
# audio: 161 frames at 24 fps (6.71 s); the output carries the input audio.
SPEECH = os.path.join(FIXTURES, "speech-flite-44k.flac")
A2V_PROMPT = ("A close-up of a woman with short dark hair talking directly to the camera in a bright living "
              "room, natural light, she speaks clearly and calmly, her lips moving with every word.")


def audio_passthrough(case):
    """The output's audio against the input: rate, and the first 6 s decoded at 16 kHz mono
    (AAC round trip) correlated with the input's."""
    p = f"{OUT}/mp4/{case}.mp4"
    if not os.path.exists(p):
        return
    import numpy as np

    def pcm(path):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-t", "6",
                              "-f", "f32le", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype="<f4")
    a, b = pcm(SPEECH), pcm(p)
    n = min(len(a), len(b))
    lags = range(-2000, 2001, 16)
    best = max(lags, key=lambda k: float(np.dot(a[max(0, k):n + min(0, k)], b[max(0, -k):n - max(0, k)])))
    x, y = a[max(0, best):n + min(0, best)], b[max(0, -best):n - max(0, best)]
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 100 else None
    record(case + "-audio", api="output audio vs input (ffmpeg)", ok=bool(r and r > 0.9), corr=r,
           lag_ms=best / 16, input_s=round(len(a) / 16000, 3), output_s=round(len(b) / 16000, 3))


# Reference-to-video (docs/ports/ltx-ref2v.md): the oracle's reference sheet and prompt.
SHEET = os.path.join(FIXTURES, "ltx-ref-sheet-768x448.png")
REF_PROMPT = (
    "Reference sheet: Top Row Left (Setting): a rocky coastline at golden hour, dark boulders in the surf and "
    "green hills behind a sandy beach. Top Row Right (Setting): a closer view of the same boulders with waves "
    "breaking around them. Bottom Row Left (Prop): a red and white striped beach umbrella, shown twice. Bottom Row "
    "Right (Character): a cartoon orange crab with big claws and eyes on stalks, shown twice. Generated video: A "
    "bright 3D animated shot on the rocky beach at golden hour. The cheerful orange cartoon crab scuttles sideways "
    "across the wet sand in front of the dark boulders, waving its big claws, next to the red and white striped "
    "beach umbrella planted in the sand, while waves roll in and break into white foam behind it."
)


def ref_first_frame_vs_sheet(case):
    """How much of the sheet shows up: SSIM of the first frame against the sheet (both at
    768x448). Low by design (the video is a new shot, not the sheet); recorded as context."""
    p = f"{OUT}/mp4/{case}.mp4"
    if not os.path.exists(p):
        return
    lav = ("[0:v]select=eq(n\\,0),setpts=N/TB,scale=768:448,format=yuv420p[a];"
           "[1:v]format=yuv420p[b];[a][b]ssim")
    r = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", p, "-i", SHEET, "-lavfi", lav,
                        "-frames:v", "1", "-f", "null", "-"], capture_output=True, text=True)
    line = [x for x in r.stderr.splitlines() if "Parsed_ssim" in x]
    record(case + "-sheet", api="frame 0 vs sheet (ffmpeg)", ok=True,
           ssim=line[-1].split("] ", 1)[-1] if line else r.stderr[-200:])


def frame_fidelity(mp4, pins, width, height):
    """SSIM / PSNR of the pinned frames of `mp4` against their images, prepared as the
    engine prepares them: cover + center crop to the generation canvas (the output
    rounded up to a multiple of 64, LTX pad-and-crop), then the output's center crop.
    pins: [(frame, image)]."""
    gw, gh = -(-width // 64) * 64, -(-height // 64) * 64
    out = {}
    for idx, img in pins:
        for m in ("ssim", "psnr"):
            lav = (f"[0:v]select=eq(n\\,{idx}),setpts=N/TB,format=yuv420p[a];"
                   f"[1:v]scale={gw}:{gh}:force_original_aspect_ratio=increase:flags=bilinear,"
                   f"crop={gw}:{gh},crop={width}:{height},format=yuv420p[b];[a][b]{m}")
            r = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", mp4, "-i", img, "-lavfi", lav,
                                "-frames:v", "1", "-f", "null", "-"], capture_output=True, text=True)
            line = [x for x in r.stderr.splitlines() if f"Parsed_{m}" in x]
            out[f"frame{idx}_{m}"] = line[-1].split("] ", 1)[-1] if line else r.stderr[-200:]
    return out


def ltx_i2v_v2(case, res, last=False, seconds=6, fps=24):
    w, h = (int(x) for x in res.split("x"))
    body = ltx_body(seconds=seconds, fps=fps, res=res, image_uri=data_uri(BEACH))
    body["prompt"] = I2V_PROMPT
    if last:
        body["last_frame_uri"] = data_uri(BEACH_ZOOM)
    frames = seconds * fps + 1
    ltx_v2(case, body, expect(frames, fps, True), endpoint="image-to-video")
    p = f"{OUT}/mp4/{case}.mp4"
    if os.path.exists(p):
        pins = [(0, BEACH)] + ([(frames - 1, BEACH_ZOOM)] if last else [])
        record(case + "-fidelity", api="frame fidelity (ffmpeg)", ok=True, **frame_fidelity(p, pins, w, h))


def native_i2v(case, size, last=False, seconds=5):
    w, h = (int(x) for x in size.split("x"))
    body = {"model": "ltx-turbo", "prompt": I2V_PROMPT, "seconds": seconds, "size": size, "seed": 11,
            "image_url": data_uri(BEACH)}
    if last:
        body["last_image_url"] = data_uri(BEACH_ZOOM)
    frames = seconds * 24 + 1
    native(case, body, expect(frames, 24, True))
    p = f"{OUT}/mp4/{case}.mp4"
    if os.path.exists(p):
        pins = [(0, BEACH)] + ([(frames - 1, BEACH_ZOOM)] if last else [])
        record(case + "-fidelity", api="frame fidelity (ffmpeg)", ok=True, **frame_fidelity(p, pins, w, h))


# Retake / extend (docs/oracle.md "LTX-2.5 retake and extend"): the oracle's beach
# push-in (768x512, 121 frames at 24 fps, the speech clip as its soundtrack).
SOURCE = os.path.join(FIXTURES, "beach-push-768x512-24fps.mp4")
RETAKE_PROMPT = ("A huge wave crashes over the dark rocks at golden hour, white spray bursting high into the air, "
                 "a narrator speaks calmly.")
EXTEND_PROMPT = ("The camera keeps pushing in slowly over the rocky beach at golden hour, waves rolling onto the "
                 "sand, a narrator speaks calmly.")


def kept_fidelity(case, spans, shift=0):
    """SSIM / PSNR of the output's kept frames against the source's: `spans` are
    [start, end) source frame ranges, found in the output `shift` frames later."""
    p = f"{OUT}/mp4/{case}.mp4"
    if not os.path.exists(p):
        return
    sel = "+".join(f"between(n\\,{a}\\,{b - 1})" for a, b in spans)
    osel = "+".join(f"between(n\\,{a + shift}\\,{b - 1 + shift})" for a, b in spans)
    out = {}
    for m in ("ssim", "psnr"):
        lav = (f"[0:v]select='{osel}',setpts=N/24/TB,format=yuv420p[a];"
               f"[1:v]select='{sel}',setpts=N/24/TB,format=yuv420p[b];[a][b]{m}")
        r = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", p, "-i", SOURCE, "-lavfi", lav, "-f", "null", "-"],
                           capture_output=True, text=True)
        line = [x for x in r.stderr.splitlines() if f"Parsed_{m}" in x]
        out[m] = line[-1].split("] ", 1)[-1] if line else r.stderr[-200:]
    ok = False
    try:
        ok = float(out["ssim"].split("All:")[1].split()[0]) > 0.8
    except Exception:  # noqa: BLE001
        pass
    record(case + "-kept", api="kept frames vs source (ffmpeg)", ok=ok, spans=spans, shift=shift, **out)


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
    # E5 / E9 image conditioning, and the 1440p tier.
    "i2v-v2-1080p": lambda: ltx_i2v_v2("i2v-v2-1080p", "1920x1080"),
    "kf-v2-1080p": lambda: ltx_i2v_v2("kf-v2-1080p", "1920x1080", last=True),
    "i2v-v2-720p": lambda: ltx_i2v_v2("i2v-v2-720p", "1280x720"),
    "native-kf-720p": lambda: native_i2v("native-kf-720p", "1280x720", last=True),
    "v2-1440p-24": lambda: ltx_v2("v2-1440p-24", ltx_body(res="2560x1440"), expect(145, 24, True)),
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
    # Reference-to-video pod (configs/serve/runpod-ltx-ref2v.toml): fal's `ingredient`
    # endpoint on ltx-pro -> the IC-LoRA companion, 1536x896x121, and the native API.
    "ref2v-fal-ingredient": lambda: (
        fal_queue("ref2v-fal-ingredient", "fal-ai/ltx-2.3-quality",
                  {"prompt": REF_PROMPT, "image_url": data_uri(SHEET), "seed": 1024},
                  expect(121, 24, True), sub="ingredient"),
        ref_first_frame_vs_sheet("ref2v-fal-ingredient")),
    "ref2v-native": lambda: (
        native("ref2v-native", {"model": "ltx-pro", "prompt": REF_PROMPT, "size": "1536x896", "num_frames": 121,
                                "seed": 1024, "reference_urls": [data_uri(SHEET)]}, expect(121, 24, True)),
        ref_first_frame_vs_sheet("ref2v-native")),
    # Audio-to-video (ltx-turbo pod): LTX v2, native with an image, fal fast.
    "a2v-v2-1080p": lambda: (
        ltx_v2("a2v-v2-1080p", {"audio_uri": data_uri(SPEECH), "prompt": A2V_PROMPT, "model": "ltx-2-5-fast"},
               expect(161, 24, True), endpoint="audio-to-video"),
        audio_passthrough("a2v-v2-1080p")),
    "a2v-native-i2v-720p": lambda: (
        native("a2v-native-i2v-720p", {"model": "ltx-turbo", "prompt": I2V_PROMPT + " A narrator speaks.",
                                       "size": "1280x720", "seed": 11, "image_url": data_uri(BEACH),
                                       "audio_url": data_uri(SPEECH)}, expect(161, 24, True)),
        audio_passthrough("a2v-native-i2v-720p"),
        os.path.exists(f"{OUT}/mp4/a2v-native-i2v-720p.mp4") and record(
            "a2v-native-i2v-720p-fidelity", api="frame fidelity (ffmpeg)", ok=True,
            **frame_fidelity(f"{OUT}/mp4/a2v-native-i2v-720p.mp4", [(0, BEACH)], 1280, 720))),
    "a2v-fal-fast": lambda: (
        fal_queue("a2v-fal-fast", "lightricks/ltx-2.5", {"audio_url": data_uri(SPEECH), "prompt": A2V_PROMPT, "seed": 5},
                  expect(161, 24, True), sub="audio-to-video/fast"),
        audio_passthrough("a2v-fal-fast")),
    "a2v-errors": lambda: [
        ltx_error("a2v-err-no-prompt", "POST", "/v2/audio-to-video", {"audio_uri": data_uri(SPEECH)}, 400, "invalid_request_error"),
        ltx_error("a2v-err-pro-unserved", "POST", "/v2/audio-to-video", {"audio_uri": data_uri(SPEECH), "prompt": "p", "model": "ltx-2-5-pro"}, 403, "permission_error"),
        ltx_error("a2v-err-image-as-audio", "POST", "/v2/audio-to-video", {"audio_uri": data_uri(BEACH), "prompt": "p", "model": "ltx-2-5-fast"}, 400, "invalid_request_error"),
    ],
    # Retake [1.5, 3.5) s of both streams on the LTX API: the whole clip comes back
    # (121 frames at 24 fps, 768x512, with audio); latent frames 5..11 (pixel frames
    # 33..88) are regenerated, the rest are VAE round trips of the source.
    "retake-v2": lambda: (
        ltx_v2("retake-v2", {"video_uri": data_uri(SOURCE), "start_time": 1.5, "duration": 2, "prompt": RETAKE_PROMPT},
               expect(121, 24, audio=True), endpoint="retake"),
        kept_fidelity("retake-v2", [(0, 33), (89, 121)])),
    # Retake the audio only (the video frozen) on fal: every frame is kept.
    "retake-fal-audio": lambda: (
        fal_queue("retake-fal-audio", "fal-ai/ltx-2.3", {"video_url": data_uri(SOURCE), "prompt": RETAKE_PROMPT,
                  "start_time": 1.5, "duration": 2, "retake_mode": "replace_audio", "seed": 7},
                  expect(121, 24, audio=True), sub="retake-video"),
        kept_fidelity("retake-fal-audio", [(0, 121)])),
    # Extend 2 s after the end on fal: 169 frames, the source's 0..104 kept.
    "extend-fal": lambda: (
        fal_queue("extend-fal", "fal-ai/ltx-2.3", {"video_url": data_uri(SOURCE), "prompt": EXTEND_PROMPT,
                  "duration": 2, "seed": 7}, expect(169, 24, audio=True), sub="extend-video"),
        kept_fidelity("extend-fal", [(0, 105)])),
    # Extend 2 s before the start with 2 s of context (native): the model sees the
    # first 41 source frames; the other 80 are copied after the generated clip
    # (48 + 41 generated, + 80 stitched = 169 frames).
    "extend-native-start": lambda: (
        native("extend-native-start", {"model": "ltx-pro", "prompt": EXTEND_PROMPT, "video_url": data_uri(SOURCE),
               "extend_s": 2, "extend_at": "start", "context_s": 2, "seed": 7}, expect(169, 24, audio=True)),
        kept_fidelity("extend-native-start", [(41, 121)], shift=48)),
    "edit-errors": lambda: [
        ltx_error("retake-err-past-end", "POST", "/v2/retake", {"video_uri": data_uri(SOURCE), "start_time": 6, "duration": 2}, 400, "invalid_request_error"),
        ltx_error("extend-err-image", "POST", "/v2/extend", {"video_uri": data_uri(BEACH), "duration": 2}, 400, "invalid_request_error"),
        ltx_error("extend-err-too-long", "POST", "/v2/extend", {"video_uri": data_uri(SOURCE), "duration": 25}, 400, "invalid_request_error"),
    ],
    # Guided audio-to-video pod (configs/serve/runpod-ltx-a2v.toml): ltx-pro's A2V
    # on the dev transformer (A2VidPipelineTwoStage). fal pro with its
    # guidance_scale, the native API at 720p with the default guidance, and
    # the fast endpoint (not served on this pod).
    "a2v-fal-pro": lambda: (
        fal_queue("a2v-fal-pro", "lightricks/ltx-2.5",
                  {"audio_url": data_uri(SPEECH), "prompt": A2V_PROMPT, "seed": 5, "guidance_scale": 3},
                  expect(161, 24, True), sub="audio-to-video/pro"),
        audio_passthrough("a2v-fal-pro")),
    "a2v-native-pro-720p": lambda: (
        native("a2v-native-pro-720p", {"model": "ltx-pro", "prompt": A2V_PROMPT, "size": "1280x720", "seed": 11,
                                       "audio_url": data_uri(SPEECH)}, expect(161, 24, True)),
        audio_passthrough("a2v-native-pro-720p")),
    "a2v-guided-errors": lambda: [
        ltx_error("a2v-err-fast-unserved", "POST", "/v2/audio-to-video",
                  {"audio_uri": data_uri(SPEECH), "prompt": "p", "model": "ltx-2-5-fast"}, 403, "permission_error"),
    ],
    "pro-err-20s": lambda: ltx_error("pro-err-20s", "POST", "/v2/text-to-video", ltx_body(model="ltx-2-5-pro", seconds=20), 400, "invalid_request_error"),
}

for c in sys.argv[3:]:
    try:
        CASES[c]()
    except Exception as e:  # noqa: BLE001
        record(c, ok=False, exception=repr(e))
