#!/usr/bin/env python3
"""Summaries of a plug.sh work dir: per cell the median denoise / total and
per-prompt denoise (benchmark.json); per compare-clips report the metrics."""
import json, sys, glob, os
wk = sys.argv[1] if len(sys.argv) > 1 else "/root/plug"
out = {"cells": {}, "compare": {}}
for b in sorted(glob.glob(f"{wk}/*/benchmark.json")):
    c = os.path.basename(os.path.dirname(b))
    d = json.load(open(b))
    pp = d.get("aggregate", {}).get("per_prompt", [])
    out["cells"][c] = {
        "denoise_s": d.get("denoise_s"), "total_s": d.get("e2e_seconds", d.get("total_s")),
        "load_s": d.get("load_s"), "peak_mib": d.get("max_device_memory_used_mib"),
        "per_prompt": {p["name"]: round(p["denoise_s"], 2) for p in pp},
    }
for f in sorted(glob.glob(f"{wk}/compare/*.json")):
    d = json.load(open(f))
    ch = {c.get("name"): c.get("values") or {} for c in d.get("checks", [])}
    pm, lp = ch.get("pixel_metrics", {}), ch.get("lpips", {})
    r = lambda x, n=3: round(x, n) if isinstance(x, (int, float)) else x
    out["compare"][os.path.basename(f)[len("compare-clips-"):-5]] = {
        "psnr": r(pm.get("psnr_mean"), 2), "lpips": r(lp.get("mean")), "lpips_max": r(lp.get("max")),
        "sharp": r(pm.get("sharpness_ratio_mean")), "jitter": r(pm.get("temporal_jitter_ratio_mean")),
        "frames": (ch.get("frame_counts") or {}).get("pairs"),
    }
json.dump(out, sys.stdout, indent=1)
