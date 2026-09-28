#!/usr/bin/env python3
"""GPU E2E, Wan batch (WP-18, design §7.6): real clients against a pod
serving FastWan (wan-turbo) through the Runpod proxy.

- FastWan Video API: GET /health, GET /, POST /generate -> GET /status/{id}
  -> GET /video/{id}, a rejection, DELETE /video/{id};
- FastVideo /v1/videos through `openai` 3.6.0: models.list, videos.create ->
  retrieve -> download_content, create_and_poll, delete;
- native /fv/v1/jobs: submit -> poll -> /content (redirect to the artifact);
- fal queue (raw HTTP, the documented flow) when an app serves the model;
- /fv/v1/capabilities.

Every clip is saved under --out and probed with ffprobe; one JSON result
per check goes to results.json (endpoint, ok, timings, facts).

    python wan_batch.py --base https://<pod>-8000.proxy.runpod.net --key fvk-... --out DIR
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import requests
from openai import OpenAI

MODEL = "fastwan21-1.3b"
W, H, FPS = 832, 480, 16
PROMPTS = [
    "A red fox trotting through fresh snow in a pine forest, soft morning light, cinematic",
    "A paper boat drifting on a rain puddle in a city street at night, neon reflections",
    "A hot air balloon rising over green hills at sunrise, gentle wind, wide shot",
    "Waves crashing against black volcanic rocks, slow motion spray, overcast sky",
]


def ffprobe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,profile,width,height,r_frame_rate,nb_frames,sample_rate,channels:format=duration,size",
             "-of", "json", path], capture_output=True, text=True, timeout=60, check=False)
        j = json.loads(out.stdout or "{}")
        return {"streams": j.get("streams", []), "format": j.get("format", {})}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def faststart(path):
    """True when the moov atom comes before mdat."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    m, d = head.find(b"moov"), head.find(b"mdat")
    return m != -1 and (d == -1 or m < d)


class Run:
    def __init__(self, out):
        self.out = out
        self.results = []

    def rec(self, endpoint, fn):
        t0 = time.monotonic()
        r = {"endpoint": endpoint}
        try:
            r.update(fn() or {})
            r["ok"] = r.get("ok", True)
        except Exception as e:  # noqa: BLE001 - record and continue
            traceback.print_exc()
            r["ok"] = False
            r["error"] = f"{type(e).__name__}: {e}"[:500]
        r["wall_s"] = round(time.monotonic() - t0, 3)
        self.results.append(r)
        print(json.dumps(r), flush=True)
        with open(os.path.join(self.out, "results.json"), "w") as f:
            json.dump(self.results, f, indent=1)
        return r

    def save(self, name, data):
        p = os.path.join(self.out, name)
        with open(p, "wb") as f:
            f.write(data)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fal-app", default="fastvideo/fastwan21-1.3b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base, key = a.base.rstrip("/"), a.key
    auth = {"Authorization": f"Bearer {key}"}
    s = requests.Session()
    run = Run(a.out)

    def health():
        h = s.get(f"{base}/health", timeout=30)
        root = s.get(f"{base}/", headers=auth, timeout=30)
        caps = s.get(f"{base}/fv/v1/capabilities", headers=auth, timeout=30).json()
        with open(os.path.join(a.out, "capabilities.json"), "w") as f:
            json.dump(caps, f, indent=1)
        m = [c for c in caps.get("models", []) if (c.get("caps") or {}).get("id") == MODEL]
        return {"ok": h.status_code == 200 and h.json().get("model_loaded") is True and bool(m),
                "health": h.json(), "root": root.json() if root.ok else root.status_code,
                "models": [(c.get("caps") or {}).get("id") for c in caps.get("models", [])],
                "recipe": ((m[0].get("recipe") or {}).get("name") if m else None)}
    run.rec("GET /health + / + /fv/v1/capabilities", health)

    # ---- FastWan Video API ------------------------------------------------
    def fastwan(frames, prompt, seed, name):
        def go():
            t0 = time.monotonic()
            r = s.post(f"{base}/generate", headers=auth, timeout=60,
                       json={"prompt": prompt, "width": W, "height": H, "num_frames": frames, "fps": FPS, "seed": seed})
            r.raise_for_status()
            job = r.json()
            jid, statuses = job["prompt_id"], [job.get("status")]
            t_proc = None
            while job.get("status") not in ("completed", "failed"):
                time.sleep(0.25)
                job = s.get(f"{base}/status/{jid}", headers=auth, timeout=30).json()
                if job.get("status") == "processing" and t_proc is None:
                    t_proc = time.monotonic()
                if statuses[-1] != job.get("status"):
                    statuses.append(job.get("status"))
            t_done = time.monotonic()
            if job["status"] != "completed":
                return {"ok": False, "job": job, "statuses": statuses}
            v = s.get(f"{base}/video/{jid}", headers=auth, timeout=120)
            t_dl = time.monotonic()
            p = run.save(name, v.content)
            pr = ffprobe(p)
            vs = [x for x in pr.get("streams", []) if x.get("codec_type") == "video"]
            ok = v.status_code == 200 and v.content[4:8] == b"ftyp" and vs and int(vs[0].get("nb_frames", 0)) == frames
            return {"ok": bool(ok), "id": jid, "frames": frames, "statuses": statuses,
                    "submit_to_completed_s": round(t_done - t0, 3),
                    "queued_s": round((t_proc or t_done) - t0, 3),
                    "download_s": round(t_dl - t_done, 3), "bytes": len(v.content),
                    "faststart": faststart(p), "ffprobe": pr}
        return go

    first = run.rec("FastWan POST /generate -> /status -> /video (81 f, warm-up)",
                    fastwan(81, PROMPTS[0], 1, "fastwan-81f-a.mp4"))
    run.rec("FastWan /generate (81 f, warm)", fastwan(81, PROMPTS[1], 2, "fastwan-81f-b.mp4"))
    run.rec("FastWan /generate (121 f)", fastwan(121, PROMPTS[2], 3, "fastwan-121f.mp4"))

    def fastwan_reject():
        bad = s.post(f"{base}/generate", headers=auth, timeout=30,
                     json={"prompt": "p", "width": W, "height": H, "num_frames": 50, "fps": FPS, "seed": 1})
        return {"ok": bad.status_code in (400, 413, 415, 422) and isinstance(bad.json().get("detail"), str),
                "status": bad.status_code, "body": bad.text[:200]}
    run.rec("FastWan /generate off-grid rejection", fastwan_reject)

    def fastwan_delete():
        jid = first.get("id")
        d = s.delete(f"{base}/video/{jid}", headers=auth, timeout=30)
        g = s.get(f"{base}/status/{jid}", headers=auth, timeout=30)
        return {"ok": d.status_code == 200 and g.status_code == 404, "delete": d.status_code, "status_after": g.status_code}
    run.rec("FastWan DELETE /video/{id}", fastwan_delete)

    # ---- FastVideo /v1/videos (openai 3.6.0) -----------------------------
    c = OpenAI(base_url=f"{base}/v1", api_key=key, max_retries=0, timeout=120)

    def oa_models():
        ids = [m.id for m in c.models.list()]
        return {"ok": MODEL in ids, "ids": ids}
    run.rec("openai models.list", oa_models)

    def oa_create():
        t0 = time.monotonic()
        v = c.videos.create(model=MODEL, prompt=PROMPTS[3], size=f"{W}x{H}", seconds="5")
        created = {"status": v.status, "model": v.model, "size": v.size, "seconds": v.seconds}
        while v.status in ("queued", "in_progress"):
            time.sleep(0.25)
            v = c.videos.retrieve(v.id)
        t_done = time.monotonic()
        if v.status != "completed":
            return {"ok": False, "video": v.model_dump()}
        body = c.videos.download_content(v.id).read()
        p = run.save("openai-5s.mp4", body)
        pr = ffprobe(p)
        d = c.videos.delete(v.id)
        return {"ok": body[4:8] == b"ftyp" and d.deleted, "id": v.id, "created": created,
                "submit_to_completed_s": round(t_done - t0, 3), "bytes": len(body), "ffprobe": pr,
                "faststart": faststart(p)}
    run.rec("openai videos.create -> retrieve -> download_content -> delete", oa_create)

    def oa_poll():
        t0 = time.monotonic()
        v = c.videos.create_and_poll(model=MODEL, prompt=PROMPTS[0], size=f"{W}x{H}", seconds="5", poll_interval_ms=250)
        return {"ok": v.status == "completed", "status": v.status, "wall": round(time.monotonic() - t0, 3)}
    run.rec("openai videos.create_and_poll", oa_poll)

    # ---- native /fv/v1/jobs -------------------------------------------------
    def native():
        t0 = time.monotonic()
        r = s.post(f"{base}/fv/v1/jobs", headers=auth, timeout=60,
                   json={"model": MODEL, "prompt": PROMPTS[2], "seed": 7, "width": W, "height": H, "num_frames": 81})
        job = r.json()
        jid = job.get("id")
        if not jid:
            return {"ok": False, "submit": r.status_code, "body": r.text[:300]}
        while job.get("status") not in ("succeeded", "failed", "cancelled"):
            time.sleep(0.25)
            job = s.get(f"{base}/fv/v1/jobs/{jid}", headers=auth, timeout=30).json()
        t_done = time.monotonic()
        cr = s.get(f"{base}/fv/v1/jobs/{jid}/content", headers=auth, timeout=120, allow_redirects=False)
        loc = cr.headers.get("location")
        body = s.get(loc, timeout=120).content if loc else cr.content
        p = run.save("native-81f.mp4", body)
        host = loc.split("/")[2] if loc and "://" in loc else None
        return {"ok": job.get("status") == "succeeded" and body[4:8] == b"ftyp", "id": jid, "status": job.get("status"),
                "content_status": cr.status_code, "media_host": host, "submit_to_succeeded_s": round(t_done - t0, 3),
                "bytes": len(body), "job_timings": job.get("timings") or job.get("metrics"), "ffprobe": ffprobe(p)}
    run.rec("native POST /fv/v1/jobs -> GET -> /content", native)

    # ---- fal queue (documented HTTP flow) ----------------------------------
    def fal():
        fk = {"Authorization": f"Key {key}"}
        app = a.fal_app
        t0 = time.monotonic()
        r = s.post(f"{base}/{app}/text-to-video", headers=fk, timeout=60,
                   json={"prompt": PROMPTS[1], "duration": 5, "resolution": "480P", "aspect_ratio": "16:9", "seed": 11})
        if r.status_code != 200:
            return {"ok": False, "submit": r.status_code, "body": r.text[:400]}
        sub = r.json()
        st = {}
        while st.get("status") != "COMPLETED":
            time.sleep(0.5)
            st = s.get(sub["status_url"].replace("https://queue.fal.run", base), headers=fk, params={"logs": 1}, timeout=30).json()
        t_done = time.monotonic()
        res = s.get(sub["response_url"], headers=fk, timeout=60)
        out = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
        url = (out.get("video") or {}).get("url")
        body = s.get(url, timeout=120).content if url else b""
        p = run.save("fal-5s.mp4", body) if body else None
        return {"ok": res.status_code == 200 and body[4:8] == b"ftyp", "submit": sub, "status": st.get("status"),
                "submit_to_completed_s": round(t_done - t0, 3), "response_status": res.status_code,
                "output": {k: v for k, v in out.items() if k != "video"}, "file": (out.get("video") or {}).get("file_name"),
                "ffprobe": ffprobe(p) if p else None}
    run.rec(f"fal queue {a.fal_app}/text-to-video", fal)

    ok = all(r["ok"] for r in run.results)
    print(json.dumps({"ok": ok, "passed": sum(r["ok"] for r in run.results), "total": len(run.results)}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
