#!/usr/bin/env python3
"""H3 reference-to-video E2E against one fv-serve pod (docs/ports/h3-ref2v.md).

    ref2v.py --base https://<pod>-8000.proxy.runpod.net --out artifacts/serve/e2e/h3-ref2v \
        [--only caps,fal.turbo,fal.two,minimax,refuse,fal.max]

The API key comes from FV_KEY. Checks: capabilities (Ref2V models, limits),
fal `minimax/h3-turbo/reference-to-video` with one uploaded image (768P) and
with two images (480P), MiniMax V2 `reference_image` content, refusals (10
images, 1080P, no references), and the max tier (`minimax/h3-max`, swapped
in). Writes <out>/ref2v.json and keeps the MP4s under <out>/samples/.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import time
import traceback

import requests

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "scripts/gpu/fixtures/ti2v-beach-832x480.jpg"
PROMPT = ("The camera glides slowly forward along the shoreline of the beach in Image 1, turquoise waves "
          "rolling in and breaking into white foam, bright sunny day, the sound of the surf and a light wind.")
a = None
OUT = {}


def fal_h():
    return {"Authorization": f"Key {a.key}"}


def check(cond, what):
    if not cond:
        raise AssertionError(str(what)[:600])


def ffprobe(path):
    exe = shutil.which("ffprobe") or os.environ.get("FFPROBE", "ffprobe")
    r = subprocess.run([exe, "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,nb_frames,sample_rate,channels:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    au = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return {"width": v["width"], "height": v["height"], "frames": int(v.get("nb_frames") or 0),
            "duration_s": float(j["format"]["duration"]),
            "audio": au and {"codec": au["codec_name"], "rate": int(au["sample_rate"]), "channels": au["channels"]}}


def download(url, name):
    d = pathlib.Path(a.out) / "samples"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.mp4"
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    p.write_bytes(r.content)
    out = ffprobe(p)
    out["bytes"] = len(r.content)
    return out


def upload():
    if "file_url" in OUT:
        return OUT["file_url"]
    r = requests.post(f"{a.base}/storage/upload/initiate", headers=fal_h(), params={"storage_type": "fal-cdn-v3"},
                      json={"content_type": "image/jpeg", "file_name": "beach.jpg"}, timeout=30)
    check(r.status_code == 200, f"initiate {r.status_code} {r.text[:300]}")
    j = r.json()
    p = requests.put(j["upload_url"], data=FIXTURE.read_bytes(), headers={"content-type": "image/jpeg"}, timeout=60)
    check(p.status_code in (200, 201, 204), f"PUT {p.status_code}")
    OUT["file_url"] = j["file_url"]
    return j["file_url"]


def fal_job(app, body, name, timeout=3600):
    t0 = time.monotonic()
    r = requests.post(f"{a.base}/{app}/reference-to-video", headers=fal_h(), json=body, timeout=60)
    check(r.status_code == 200, f"submit {r.status_code} {r.text[:400]}")
    rid = r.json()["request_id"]
    seen = []
    while True:
        s = requests.get(f"{a.base}/{app}/requests/{rid}/status", headers=fal_h(), timeout=30).json()
        if not seen or seen[-1] != s["status"]:
            seen.append(s["status"])
        if s["status"] == "COMPLETED":
            break
        check(time.monotonic() - t0 < timeout, f"timeout {seen}")
        time.sleep(2)
    wall = time.monotonic() - t0
    res = requests.get(f"{a.base}/{app}/requests/{rid}", headers=fal_h(), timeout=60)
    check(res.status_code == 200, f"result {res.status_code} {res.text[:400]}")
    j = res.json()
    return {"request_id": rid, "wall_s": round(wall, 1), "statuses": seen, "timings": j.get("timings"),
            "seed": j.get("seed"), "mp4": download(j["video"]["url"], name)}


def t_caps():
    r = requests.get(f"{a.base}/fv/v1/capabilities", headers={"Authorization": f"Bearer {a.key}"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    ms = [m["caps"] for m in j["models"]]
    ref = [m for m in ms if "ref2v" in m.get("tasks", [])]
    check(ref, "no ref2v model")
    return {"models": [{k: m.get(k) for k in ("id", "tier", "tasks", "refs", "recipe", "resident")} for m in ms],
            "ref2v_models": [m.get("id") for m in ref], "tiers": j.get("tiers"), "aliases": j.get("aliases")}


def t_fal_turbo():
    d = fal_job("minimax/h3-turbo", {"prompt": PROMPT, "reference_image_urls": [upload()], "duration": 5,
                                     "resolution": "768P", "seed": 7}, "fal-r2v-turbo-768p")
    check(min(d["mp4"]["width"], d["mp4"]["height"]) == 768, d["mp4"])
    check(d["mp4"]["audio"] and d["mp4"]["audio"]["channels"] == 2, d["mp4"])
    check(d["seed"] == 7, d)
    return d


def t_fal_two():
    img = upload()
    d = fal_job("minimax/h3-turbo", {"prompt": "Image 1 at golden hour, then the same shoreline from Image 2 under storm clouds",
                                     "reference_image_urls": [img, img], "duration": 5, "resolution": "480P",
                                     "aspect_ratio": "16:9"}, "fal-r2v-turbo-480p-two")
    check(min(d["mp4"]["width"], d["mp4"]["height"]) == 480, d["mp4"])
    return d


def t_minimax():
    h = {"Authorization": f"Bearer {a.key}", "Content-Type": "application/json"}
    t0 = time.monotonic()
    r = requests.post(f"{a.base}/v2/video_generation", headers=h, timeout=60, json={
        "model": "MiniMax-H3-Turbo", "resolution": "768P", "duration": 5,
        "content": [{"type": "text", "text": PROMPT.replace("Image 1", "Picture 1")},
                    {"type": "image_url", "image_url": {"url": upload()}, "role": "reference_image"}]})
    check(r.status_code == 200 and "task_id" in r.json(), f"create {r.status_code} {r.text[:300]}")
    tid = r.json()["task_id"]
    while True:
        q = requests.get(f"{a.base}/v2/query/video_generation/{tid}", headers=h, timeout=30).json()
        q = q.get("task", q)
        if str(q.get("status", "")).lower() in ("success", "succeeded", "fail", "failed"):
            break
        check(time.monotonic() - t0 < 3600, q)
        time.sleep(3)
    check(str(q["status"]).lower() in ("success", "succeeded"), q)
    return {"task_id": tid, "wall_s": round(time.monotonic() - t0, 1), "usage": q.get("usage"),
            "mp4": download(q["content"]["url"], "minimax-r2v-768p")}


def t_refuse():
    img = upload()
    out = {}
    for name, body in [
        ("ten_images", {"prompt": "x", "reference_image_urls": [img] * 10}),
        ("res_1080p", {"prompt": "x", "reference_image_urls": [img], "resolution": "1080P"}),
        ("no_refs", {"prompt": "x"}),
    ]:
        r = requests.post(f"{a.base}/minimax/h3-turbo/reference-to-video", headers=fal_h(), json=body, timeout=60)
        out[name] = {"code": r.status_code, "body": r.text[:300]}
        check(400 <= r.status_code < 500, out[name])
    return out


def t_fal_max():
    d = fal_job("minimax/h3-max", {"prompt": PROMPT, "reference_image_urls": [upload()], "duration": 5,
                                   "resolution": "768P", "seed": 7}, "fal-r2v-max-768p")
    check(min(d["mp4"]["width"], d["mp4"]["height"]) == 768, d["mp4"])
    return d


CHECKS = {"caps": t_caps, "fal.turbo": t_fal_turbo, "fal.two": t_fal_two, "minimax": t_minimax,
          "refuse": t_refuse, "fal.max": t_fal_max}


def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default=os.environ.get("FV_KEY"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(CHECKS))
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    results = {}
    for name in a.only.split(","):
        t0 = time.monotonic()
        try:
            results[name] = {"ok": True, "data": CHECKS[name](), "s": round(time.monotonic() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:],
                             "s": round(time.monotonic() - t0, 1)}
        print(name, "ok" if results[name]["ok"] else "FAIL " + results[name]["error"][:300], flush=True)
        (pathlib.Path(a.out) / "ref2v.json").write_text(json.dumps(results, indent=1))
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
