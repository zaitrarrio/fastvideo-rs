#!/usr/bin/env python3
"""Absolute per-clip statistics (no reference): luma Laplacian variance
(sharpness, mean over frames), mean |frame diff| (motion), std of the
frame-mean luma steps (flicker). clipstats.py <workdir> > clipstats.json"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
wk = Path(sys.argv[1]); out = {}
for cell in sorted(p for p in wk.iterdir() if (p / "frames").is_dir()):
    for pd in sorted(q for q in (cell / "frames").iterdir() if q.is_dir() and q.name not in ("cold", "warmup")):
        fs = sorted(pd.glob("frame-*.png"))[::2]
        if not fs:
            continue
        ys = [np.asarray(Image.open(f).convert("L"), dtype=np.float32) for f in fs]
        lap = [float(np.var(y[1:-1, 1:-1] * 4 - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:])) for y in ys]
        motion = [float(np.mean(np.abs(b - a))) for a, b in zip(ys, ys[1:])]
        means = np.array([y.mean() for y in ys])
        out.setdefault(cell.name, {})[pd.name] = {
            "frames": len(sorted(pd.glob("frame-*.png"))), "sharpness": round(float(np.mean(lap)), 1),
            "motion": round(float(np.mean(motion)), 2), "flicker": round(float(np.std(np.diff(means))), 3),
            "luma": round(float(means.mean()), 1)}
json.dump(out, sys.stdout, indent=1)
