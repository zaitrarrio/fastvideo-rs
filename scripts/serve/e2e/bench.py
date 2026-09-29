#!/usr/bin/env python3
"""Single-pod GPU benchmark through the native API (docs/serve/bench/).

    bench.py --base https://<pod>-8000.proxy.runpod.net --key-file <state.json> \
        --out artifacts/serve/bench/b200/runs.jsonl --label h3t-480 --reps 3 \
        --body '{"model":"fasth3","aspect_ratio":"16:9","short_edge":480,"seconds":5}'
    bench.py ... --concurrent 5      # submit N at once, record queue times

Each job is submitted to `POST /fv/v1/jobs`, polled every 0.5 s, and its
job object (with the engine's `metrics`: inference_s, stage_durations,
peak_memory_mb, queue_s, run_s) is appended as one JSON line with the
client wall time. `@file` in the image_url / audio_url fields becomes a data
URI. The API key is read from the state file and never printed.
"""

import argparse
import base64
import concurrent.futures as cf
import json
import mimetypes
import pathlib
import sys
import time

import urllib.request

PROMPT = ("A red fox trots through fresh snow at dawn, its breath visible in the "
          "cold air, pine trees behind it, cinematic lighting")


def req(method, url, key, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {key}")
    r.add_header("User-Agent", "curl/8.5.0")  # the Runpod proxy (Cloudflare) refuses Python-urllib (1010)
    if data is not None:
        r.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:800]}


def data_uri(path):
    p = pathlib.Path(path)
    mt = mimetypes.guess_type(p.name)[0] or ("audio/flac" if p.suffix == ".flac" else "application/octet-stream")
    return f"data:{mt};base64,{base64.b64encode(p.read_bytes()).decode()}"


def run_one(a, key, body, label, i, t_submit_gate=None):
    t0 = time.monotonic()
    code, job = req("POST", f"{a.base}/fv/v1/jobs", key, body)
    if code >= 300 or "id" not in job:
        return {"label": label, "i": i, "ok": False, "submit_code": code, "error": job}
    jid = job["id"]
    while True:
        time.sleep(0.5)
        code, st = req("GET", f"{a.base}/fv/v1/jobs/{jid}", key)
        if code == 200 and st.get("status") in ("succeeded", "failed", "cancelled"):
            break
        if time.monotonic() - t0 > a.timeout:
            return {"label": label, "i": i, "ok": False, "error": "client timeout", "id": jid}
    wall = time.monotonic() - t0
    keep = {k: st.get(k) for k in ("id", "model", "status", "width", "height", "num_frames", "fps", "error", "metrics",
                                   "tier", "recipe", "created_at", "started_at", "completed_at")}
    return {"label": label, "i": i, "ok": st.get("status") == "succeeded", "wall_s": round(wall, 2), **keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--concurrent", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    key = json.loads(pathlib.Path(a.key_file).read_text())["key"]
    body = json.loads(a.body)
    body.setdefault("prompt", PROMPT)
    body.setdefault("seed", 7)
    for f in ("image_url", "audio_url", "last_image_url"):
        if isinstance(body.get(f), str) and body[f].startswith("@"):
            body[f] = data_uri(body[f][1:])
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    if a.concurrent:
        with cf.ThreadPoolExecutor(a.concurrent) as ex:
            futs = [ex.submit(run_one, a, key, dict(body, seed=body["seed"] + i), a.label, i) for i in range(a.concurrent)]
            recs = [f.result() for f in futs]
    else:
        for i in range(a.reps):
            recs.append(run_one(a, key, body, a.label, i))
            r = recs[-1]
            m = r.get("metrics") or {}
            print(f"{a.label}#{i} ok={r['ok']} wall={r.get('wall_s')} run={m.get('run_s')} inf={m.get('inference_s')} "
                  f"peak={m.get('peak_memory_mb')} stages={json.dumps(m.get('stage_durations'))} {r.get('width')}x{r.get('height')}x{r.get('num_frames')}"
                  + ("" if r["ok"] else f" err={json.dumps(r.get('error'))[:400]}"), file=sys.stderr, flush=True)
    with out.open("a") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    if a.concurrent:
        for r in recs:
            m = r.get("metrics") or {}
            print(f"{a.label}#{r['i']} ok={r['ok']} wall={r.get('wall_s')} queue={m.get('queue_s')} run={m.get('run_s')}", file=sys.stderr)
    sys.exit(0 if all(r["ok"] for r in recs) else 1)


if __name__ == "__main__":
    main()
