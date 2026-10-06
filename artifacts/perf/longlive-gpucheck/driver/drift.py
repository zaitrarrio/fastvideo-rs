#!/usr/bin/env python3
"""warp_err / lum_flicker (scripts/gpu/hd_upscaler_metrics.py's definitions) on
`wan stream ... dump=1` frames: per run, overall, per 5 s segment, and in a
+-1 s window around each prompt switch (15/30/45 s at 16 fps).
    drift.py <frames root with one dir per run>  > drift.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/e2e/ll")
import hd_upscaler_metrics as m

FPS = 16
root = Path(sys.argv[1]); out = {}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    fs = m.frames(d)
    ys = [m.luma(p) for p in fs]
    fl = m.flows(ys)
    e, ehf = m.warp_errors(ys, fl)
    means = np.array([y.mean() for y in ys])
    dm = np.diff(means)
    seg = []
    for s in range(0, len(e), 5 * FPS):
        seg.append({"t0_s": s / FPS, "warp_err": float(np.mean(e[s:s + 5 * FPS])), "lum_flicker": float(np.std(dm[s:s + 5 * FPS]))})
    sw = {}
    for t in (15, 30, 45):
        a, b = (t - 1) * FPS, (t + 1) * FPS
        if b <= len(e):
            sw[str(t)] = {"warp_err": float(np.mean(e[a:b])), "warp_err_max": float(np.max(e[a:b])),
                          "lum_jump_max": float(np.max(np.abs(dm[a:b]))), "lum_flicker": float(np.std(dm[a:b]))}
    out[d.name] = {"frames": len(ys), "warp_err": float(np.mean(e)), "warp_err_hf": float(np.mean(ehf)),
                   "warp_err_max_over_median": float(np.max(e) / np.median(e)), "warp_err_argmax": int(np.argmax(e)) + 1,
                   "lum_flicker": float(np.std(dm)), "segments": seg, "switch_windows": sw}
    print(d.name, out[d.name]["warp_err"], out[d.name]["lum_flicker"], file=sys.stderr, flush=True)
json.dump(out, sys.stdout, indent=1)
