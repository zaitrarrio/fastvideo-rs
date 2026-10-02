#!/usr/bin/env python3
"""Calibrated H3 gate (docs/perf/sage-attention.md section 8.2), written
before the runs. Reads compare-clips JSONs and the cells' benchmark.json,
prints the per-prompt table and the verdict per recipe.

    calib.py <results dir> [max dense]

<results dir>/compare/compare-clips-<recipe>-<a>--<b>-<prompt>-s<seed>.json
<results dir>/<recipe>-<arm>/benchmark.json
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

PROMPTS = ["h3-demo", "ltx-multishot", "ltx-newsbroadcast", "ltx-frogyoga", "spark-mountain-lake"]
METRICS = ["lpips", "psnr", "sharp", "jitter"]
# Rule 1 margins (PSNR: dB below P5) and rule 2 additive floors (PSNR: dB below min).
MARGIN = {"lpips": 0.03, "psnr": 1.0, "sharp": 0.02, "jitter": 0.03}
OUTLIER_ADD = {"lpips": 0.05, "psnr": 3.0, "sharp": 0.02, "jitter": 0.03}
MIN_SPEEDUP = 1.10


def load_pair(root, recipe, a, b):
    """{(prompt, seed): {metric: deviation, sharp_raw, jitter_raw, lpips_max}}"""
    out = {}
    pat = f"{root}/compare/compare-clips-{recipe}-{a}--{b}-*.json"
    for f in sorted(glob.glob(pat)):
        clip = f.split(f"{recipe}-{a}--{b}-", 1)[1][: -len(".json")]
        m = re.fullmatch(r"(.*)-s(\d+)", clip)
        if not m:
            continue
        d = json.load(open(f))
        c = {x["name"]: x["values"] for x in d.get("checks", [])}
        pm, lp = c.get("pixel_metrics", {}), c.get("lpips", {})
        sh, ji = pm.get("sharpness_ratio_mean"), pm.get("temporal_jitter_ratio_mean")
        out[(m.group(1), int(m.group(2)))] = {
            "lpips": lp.get("mean"),
            "lpips_max": lp.get("max"),
            "psnr": pm.get("psnr_mean"),
            "sharp": abs(math.log(sh)) if sh else None,
            "jitter": abs(math.log(ji)) if ji else None,
            "sharp_raw": sh,
            "jitter_raw": ji,
        }
    return out


def per_clip_denoise(root, recipe, arm):
    b = json.load(open(f"{root}/{recipe}-{arm}/benchmark.json"))
    got = {name: r.get("denoise_s") for name, r in (b.get("prompts") or {}).items()}
    vals = [v for v in got.values() if v]
    return got, (float(np.median(vals)) if vals else b.get("denoise_s"))


def judge(recipe, root):
    ctl = load_pair(root, recipe, "cud", "fw2")
    sage = load_pair(root, recipe, "cud", "sage")
    alt = load_pair(root, recipe, "fw2", "sage")
    print(f"\n## {recipe}: {len(ctl)} control pairs, {len(sage)} Sage pairs, {len(alt)} fw2-vs-Sage pairs")
    passes, outliers, lines = 0, [], []
    for p in PROMPTS:
        seeds = sorted(s for (q, s) in ctl if q == p)
        C = {m: np.array([ctl[(p, s)][m] for s in seeds], float) for m in METRICS}
        sseeds = sorted(s for (q, s) in sage if q == p)
        S = {m: np.array([sage[(p, s)][m] for s in sseeds], float) for m in METRICS}
        if len(seeds) == 0 or len(sseeds) == 0:
            lines.append(f"| {p} | missing | | | | | |")
            continue
        ok1, why, cells = True, [], []
        for m in METRICS:
            med = float(np.median(S[m]))
            if m == "psnr":
                band = float(np.percentile(C[m], 5)) - MARGIN[m]
                inside = med >= band
                bound = float(C[m].min()) - OUTLIER_ADD[m]
                out = [float(x) for x in S[m] if x < bound]
                cells.append(f"{med:.2f} / {band:.2f}")
            else:
                band = float(np.percentile(C[m], 95)) + MARGIN[m]
                inside = med <= band
                cmax = float(C[m].max())
                bound = max(1.5 * cmax, cmax + OUTLIER_ADD[m])
                out = [float(x) for x in S[m] if x > bound]
                cells.append(f"{med:.3f} / {band:.3f}")
            if not inside:
                ok1 = False
                why.append(f"{m} median outside band")
            if out:
                outliers.append(f"{p} {m} {out} beyond {bound:.3f}")
                why.append(f"{m} outlier")
        passes += ok1
        sr = np.median([sage[(p, s)]["sharp_raw"] for s in sseeds])
        jr = np.median([sage[(p, s)]["jitter_raw"] for s in sseeds])
        csr = np.median([ctl[(p, s)]["sharp_raw"] for s in seeds])
        cjr = np.median([ctl[(p, s)]["jitter_raw"] for s in seeds])
        verdict = "pass" if ok1 and not any(o.startswith(p + " ") for o in outliers) else "FAIL: " + ", ".join(why)
        lines.append(f"| {p} | {' | '.join(cells)} | {sr:.3f} ({csr:.3f}) | {jr:.3f} ({cjr:.3f}) | {verdict} |")
    print("| prompt | LPIPS med S / band | PSNR med S / band | ln-sharp med S / band | ln-jitter med S / band | sharp ratio med S (ctl) | jitter ratio med S (ctl) | prompt |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    # Raw rows for the record.
    print("\nraw (clip: ctl | sage | fw2-vs-sage), lpips psnr sharp jitter:")
    for k in sorted(ctl):
        def f(d):
            if k not in d:
                return "-"
            r = d[k]
            return f"{r['lpips']:.3f} {r['psnr']:.2f} {r['sharp_raw']:.3f} {r['jitter_raw']:.3f}"
        print(f"  {k[0]}-s{k[1]}: {f(ctl)} | {f(sage)} | {f(alt)}")
    try:
        dc, mc = per_clip_denoise(root, recipe, "cud")
        ds, ms = per_clip_denoise(root, recipe, "sage")
        df, mf = per_clip_denoise(root, recipe, "fw2")
        speed = mc / ms if mc and ms else float("nan")
        print(f"\ndenoise median s: cud {mc} fw2 {mf} sage {ms} -> sage speedup {speed:.3f}x")
    except (OSError, ValueError, TypeError) as e:
        speed = float("nan")
        print(f"\ndenoise: unavailable ({e})")
    ok = passes >= 4 and not outliers and speed >= MIN_SPEEDUP
    print(f"\nrule 1: {passes}/5 prompts inside the band; rule 2 outliers: {outliers or 'none'}; "
          f"speed {speed:.3f}x (>= {MIN_SPEEDUP}) -> VERDICT {recipe}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    root = sys.argv[1]
    for r in sys.argv[2:] or ["max", "dense"]:
        if os.path.isdir(f"{root}/{r}-cud"):
            judge(r, root)


if __name__ == "__main__":
    main()
