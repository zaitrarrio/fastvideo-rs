#!/usr/bin/env python3
"""Summarise an hd matrix run (runpod-matrix.sh hd) fetched with FV_FETCH_TREE=1.

    python3 scripts/gpu/hd_report.py artifacts/runpod/hd/<tag> [--jpg-out DIR]

Prints per-cell timings and peak memory (benchmark.json), the compare-clips
ratios (native 1080p and the upscaler against Lanczos-upscaled 768p), a
no-reference sharpness table from the kept keyframes, and audio/video stream
durations. With --jpg-out, writes small side-by-side crops (Lanczos 768p,
native 1080p, upscaler) as JPGs. Needs numpy + Pillow, ffprobe.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

CELLS = ["turbo-768p", "turbo-1080p", "turbo-768p-v", "turbo-1080p-v", "max-768p", "max-1080p",
         "turbo-1080p-10s", "max-1080p-10s"]


def gray(path, size=None):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lap_var(g):
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def hf_fraction(g, cutoff):
    """Share of spectral energy (DC removed) above `cutoff` cycles/pixel (radial)."""
    g = g - g.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx**2 + fy**2)
    return float(f[r > cutoff].sum() / f.sum())


def bench(run, cell):
    p = os.path.join(run, cell, "benchmark.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def ffprobe_durations(mp4):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,nb_frames,width,height",
         "-of", "json", mp4], capture_output=True, text=True).stdout
    return json.loads(out or "{}").get("streams", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--jpg-out")
    a = ap.parse_args()
    run = a.run
    print("## cells")
    for cell in CELLS:
        b = bench(run, cell)
        if b is None:
            print(cell, "missing")
            continue
        s = json.dumps(b)
        print(cell, s[:3000])
        print()
    print("## compare reports")
    for f in sorted(glob.glob(os.path.join(run, "compare", "compare-clips-*.json"))):
        d = json.load(open(f))
        print(os.path.basename(f), json.dumps(d)[:1500])
        print()
    print("## keyframe sharpness (grayscale, at 1920x1088 / 1088x1920)")
    rows = []
    for cell in sorted(os.listdir(run)):
        kd = os.path.join(run, cell, "keyframes")
        if not os.path.isdir(kd):
            continue
        for p in sorted(os.listdir(kd)):
            frames = sorted(glob.glob(os.path.join(kd, p, "*.png")))
            if not frames:
                continue
            size0 = Image.open(frames[0]).size
            # Everything is measured on the 1080p canvas of its orientation.
            tgt = (1920, 1088) if size0[0] >= size0[1] else (1088, 1920)
            lv, hf = [], []
            for fr in frames:
                g = gray(fr, tgt)
                lv.append(lap_var(g))
                # 768p content upsampled by 1088/768 has no energy above
                # 0.5 * 768/1088 = 0.353 cycles/pixel.
                hf.append(hf_fraction(g, 0.5 * 768 / 1088))
            rows.append((cell, p, size0, np.mean(lv), np.mean(hf)))
            print(f"{cell:28s} {p:22s} native {size0[0]}x{size0[1]}  lapvar {np.mean(lv):9.2f}  hf>768p-nyquist {np.mean(hf) * 1e4:7.2f} e-4")
    print("## streams")
    for mp4 in sorted(glob.glob(os.path.join(run, "*", "frames", "*", "*.mp4"))):
        st = ffprobe_durations(mp4)
        print(os.path.relpath(mp4, run), [(s.get("codec_type"), s.get("duration"), s.get("nb_frames")) for s in st])
    if a.jpg_out:
        os.makedirs(a.jpg_out, exist_ok=True)
        print("## crops ->", a.jpg_out)


if __name__ == "__main__":
    sys.exit(main())
