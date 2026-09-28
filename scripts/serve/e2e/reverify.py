#!/usr/bin/env python3
"""On-pod checks for the E2E re-verification (docs/serve/e2e/reverify.md).
Runs next to fv-serve (127.0.0.1:8000) from the /e2e venv; the API key is
FV_KEY (never printed). Every subcommand prints one JSON line.

  reverify.py fal <app> <json body> <name>   fal queue job: submit, 0.25 s status
                                             polls, result, download + ffprobe
  reverify.py sync <model> <WxH> <name>      /v1/videos/sync (302 + X-* headers)
  reverify.py sfwan <seconds> <out dir>      /fv/v1/streams WHIP to MediaMTX,
                                             RTSP recording, set_prompt at 60/120 s
"""

import json
import os
import subprocess
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"
KEY = os.environ["FV_KEY"]
OUT = "/e2e/out"
os.makedirs(OUT, exist_ok=True)


def ffprobe(path):
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,profile,width,height,nb_read_frames,r_frame_rate,sample_rate,channels:format=duration,size",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout)
    return {"streams": j["streams"], "format": j["format"]}


def fal(app, body, name):
    h = {"Authorization": f"Key {KEY}"}
    task = "text-to-video"
    t0 = time.time()
    r = requests.post(f"{BASE}/{app}/{task}", headers=h, json=body, timeout=60)
    out = {"app": app, "body": body, "submit_http": r.status_code, "t_submit": t0}
    if r.status_code != 200:
        out["submit_body"] = r.text[:600]
        return out
    rid = r.json()["request_id"]
    seen, first_prog = [], None
    while True:
        s = requests.get(f"{BASE}/{app}/requests/{rid}/status", headers=h, timeout=30).json()
        if not seen or seen[-1] != s["status"]:
            seen.append(s["status"])
            if s["status"] == "IN_PROGRESS" and first_prog is None:
                first_prog = time.time() - t0
        if s["status"] == "COMPLETED":
            break
        if time.time() - t0 > 900:
            out["timeout"] = True
            break
        time.sleep(0.25)
    out.update(statuses=seen, in_queue_s=round(first_prog, 2) if first_prog else None,
               completed_s=round(time.time() - t0, 2), status=s)
    res = requests.get(f"{BASE}/{app}/requests/{rid}", headers=h, timeout=60)
    out["result_http"] = res.status_code
    if res.status_code == 200:
        j = res.json()
        out["result_keys"] = sorted(j)
        out["timings"] = j.get("timings")
        mp4 = f"{OUT}/{name}.mp4"
        with open(mp4, "wb") as f:
            f.write(requests.get(j["video"]["url"], timeout=300).content)
        out["mp4"] = ffprobe(mp4)
    else:
        out["result_body"] = res.text[:600]
    return out


def sync(model, size, name):
    h = {"Authorization": f"Bearer {KEY}"}
    body = {"model": model, "prompt": "A red fox trotting through fresh snow in a birch forest, soft morning light, "
            "shallow depth of field, cinematic", "size": size, "seconds": "5", "seed": 7}
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/videos/sync", headers=h, json=body, timeout=600, allow_redirects=False)
    out = {"model": model, "size": size, "http": r.status_code, "wall_s": round(time.time() - t0, 2),
           "headers": {k: v for k, v in r.headers.items() if k.lower().startswith("x-") and "request" not in k.lower()}}
    mp4 = f"{OUT}/{name}.mp4"
    if r.status_code in (301, 302, 303, 307):
        data = requests.get(r.headers["location"], timeout=300).content
    elif r.status_code == 200:
        data = r.content
    else:
        out["body"] = r.text[:600]
        return out
    with open(mp4, "wb") as f:
        f.write(data)
    out["mp4"] = ffprobe(mp4)
    for i in (0, 30, 60, 90, 120):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vf", f"select=eq(n\\,{i})", "-frames:v", "1",
                        "-q:v", "3", f"{OUT}/{name}-f{i:03d}.jpg"], check=False)
    return out


P = [
    "A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed",
    "A drone shot gliding over snowy mountain peaks at dawn, pink sky, slow steady forward camera motion, highly detailed",
    "A drone shot flying low over a turquoise tropical lagoon with white sand, bright midday sun, slow steady forward camera motion, highly detailed",
]


def sfwan(seconds, out_dir):
    seconds = int(seconds)
    os.makedirs(out_dir, exist_ok=True)
    h = {"Authorization": f"Bearer {KEY}"}
    t_post = time.time()
    c = requests.post(f"{BASE}/fv/v1/streams", headers=h, timeout=120, json={
        "model": "sf-wan", "whip_url": "http://127.0.0.1:8889/sfwan/whip", "prompt": P[0],
        "width": 832, "height": 480, "max_seconds": seconds + 30}).json()
    sid = c["id"]
    st = {}
    while time.time() - t_post < 120:
        st = requests.get(f"{BASE}/fv/v1/streams/{sid}", headers=h, timeout=10).json()
        if st.get("status", {}).get("state") in ("streaming", "closed"):
            break
        time.sleep(0.1)
    t_stream = time.time()
    time.sleep(1)
    rec = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i",
                            "rtsp://127.0.0.1:8554/sfwan", "-t", str(seconds), "-c", "copy", "-y", f"{out_dir}/rtsp.mkv"])
    t_rec = time.time()
    status_log = open(f"{out_dir}/status.jsonl", "w")
    switches = []
    n = 0
    while time.time() - t_rec < seconds:
        el = time.time() - t_rec
        want = int(el // 60)
        if want > n and want <= 2:
            n = want
            before = requests.get(f"{BASE}/fv/v1/streams/{sid}", headers=h, timeout=10).json()
            ts = time.time()
            rep = requests.post(f"{BASE}/fv/v1/streams/{sid}/commands", headers=h, timeout=10,
                                json={"type": "set_prompt", "data": {"prompt": P[n]}}).json()
            te = time.time()
            after = requests.get(f"{BASE}/fv/v1/streams/{sid}", headers=h, timeout=10).json()
            switches.append({"n": n, "t": ts, "rec_t": ts - t_rec, "reply_ms": round((te - ts) * 1000, 1), "reply": rep,
                             "before": {"pacer": before.get("pacer"), "block": before.get("session", {}).get("block_index")},
                             "after": {"pacer": after.get("pacer"), "block": after.get("session", {}).get("block_index")}})
        s = requests.get(f"{BASE}/fv/v1/streams/{sid}", headers=h, timeout=10).json()
        status_log.write(json.dumps({"t": time.time(), "s": s}) + "\n")
        status_log.flush()
        time.sleep(1)
    rec.wait(timeout=60)
    end = requests.get(f"{BASE}/fv/v1/streams/{sid}", headers=h, timeout=10).json()
    requests.delete(f"{BASE}/fv/v1/streams/{sid}", headers=h, timeout=30)
    pk = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                         "-of", "csv=p=0", f"{out_dir}/rtsp.mkv"], capture_output=True, text=True).stdout.split()
    rows = [x.split(",") for x in pk if x.split(",")[0] not in ("", "N/A")]
    pts = [float(r[0]) for r in rows]
    keys = [float(r[0]) for r in rows if len(r) > 1 and "K" in r[1]]
    dur = (max(pts) - min(pts)) if pts else 0
    gaps = sorted(b - a for a, b in zip(sorted(pts), sorted(pts)[1:]))
    for sw in switches:  # frames around each switch (recording time ~= wall time since the recorder started)
        for d in (0, 2, 4, 6, 8, 10):
            t = sw["rec_t"] + d
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", f"{out_dir}/rtsp.mkv", "-frames:v", "1",
                            "-q:v", "4", "-vf", "scale=416:240", f"{out_dir}/sw{sw['n']}-{d:02d}s.jpg"], check=False)
    json.dump(switches, open(f"{out_dir}/switches.json", "w"))
    return {"stream": sid, "create": {k: c.get(k) for k in ("id", "mode", "model")}, "t_post_to_streaming_s": round(t_stream - t_post, 2),
            "end": end, "switches": switches,
            "rtsp": {"packets": len(pts), "duration_s": round(dur, 2), "fps": round((len(pts) - 1) / dur, 3) if dur else None,
                     "keyframes": len(keys), "idr_per_min": round(len(keys) / dur * 60, 2) if dur else None,
                     "max_gap_s": round(gaps[-1], 3) if gaps else None, "p99_gap_s": round(gaps[int(len(gaps) * 0.99)], 3) if gaps else None}}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fal":
        r = fal(sys.argv[2], json.loads(sys.argv[3]), sys.argv[4])
    elif cmd == "sync":
        r = sync(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "sfwan":
        r = sfwan(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
    print(json.dumps(r))
