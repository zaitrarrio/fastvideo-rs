#!/usr/bin/env python3
# Used by docs/perf/sol-bench.md runs: python3 -I scripts/gpu/sol_bench_mirror.py b1 <pod id> <out dir>
"""mirror.py <set> <pod id> <out dir>: copy each finished cell (summary.json present)
from the pod's port-8000 file server; frames PNGs, cold/warmup dirs and caches skipped."""
import re, subprocess, sys, time, os
s, pod, out = sys.argv[1:4]
base = f"https://{pod}-8000.proxy.runpod.net/sol-bench-{s}/"
skip = re.compile(r"\.png$|/cold/|/warmup/|\.cache$|text-cache|\.wav$")
def get(url, dest=None):
    args = ["curl", "-sS", "--max-time", "600", "--fail", url]
    if dest:
        args += ["-o", dest]
    r = subprocess.run(args, capture_output=True, text=dest is None)
    return r.stdout if r.returncode == 0 and dest is None else (r.returncode == 0)
def tree(rel, dest):
    os.makedirs(dest, exist_ok=True)
    listing = get(base + rel) or ""
    for e in re.findall(r'href="([^"]+)"', listing):
        if e.startswith(("/", "?", "..")) or skip.search(rel + e):
            continue
        if e.endswith("/"):
            tree(rel + e, os.path.join(dest, e[:-1]))
        else:
            get(base + rel + e, os.path.join(dest, e))
done = set()
while True:
    root = get(base) or ""
    tags = re.findall(r'href="([^"/]+)/"', root)
    if tags:
        tag = tags[0]
        listing = get(base + tag + "/") or ""
        for f in ("live.log", "sysinfo.txt", "box.txt"):
            get(base + tag + "/" + f, os.path.join(out, f)) if os.makedirs(out, exist_ok=True) is None else None
        for cell in re.findall(r'href="([^"/]+)/"', listing):
            if cell in done:
                continue
            if get(base + f"{tag}/{cell}/summary.json"):
                tree(f"{tag}/{cell}/", os.path.join(out, cell))
                done.add(cell)
                print(f"mirrored {s}/{cell}", flush=True)
        if get(base + tag + "/DONE"):
            print(f"{s} DONE", flush=True)
            break
    time.sleep(60)
