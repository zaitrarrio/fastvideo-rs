#!/usr/bin/env python3
"""The calibrated SageAttention2 gate (docs/perf/sage-attention.md sections 8.2
and 9) over a fetched fv-lab `sage-wan` run (WO-2b of ADR-0024 step 2,
docs/plan/2026-11-24-adr24-step2-wo.md). Reads the cell's compare-clips
reports and the arms' benchmark.json, prints the per-prompt table, the raw
rows and the verdict. Written before the first run; the rule is the H3 one
(artifacts/perf/sage-calibrated/driver/calib.py) with the three-prompt
threshold of section 9 (2 of 3 prompts in band).

    sage-calib.py <fetched cell dir> [--prompts a,b,c] [--min-pass N]
                  [--control cud,fw2] [--sage cud,sage]

<cell dir>/compare/compare-clips-<a>--<b>-<prompt>-s<seed>.json (or the
same files directly under <cell dir>), <cell dir>/<arm>/benchmark.json.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

PROMPTS = ["ltx-multishot", "ltx-newsbroadcast", "ltx-frogyoga"]
METRICS = ["lpips", "psnr", "sharp", "jitter"]
MARGIN = {"lpips": 0.03, "psnr": 1.0, "sharp": 0.02, "jitter": 0.03}
OUTLIER_ADD = {"lpips": 0.05, "psnr": 3.0, "sharp": 0.02, "jitter": 0.03}
MIN_SPEEDUP = 1.10
MIN_PASS = 2


def compare_files(root, a, b):
    pats = [f"{root}/compare/compare-clips-{a}--{b}-*.json", f"{root}/compare-clips-{a}--{b}-*.json"]
    out = []
    for p in pats:
        out += glob.glob(p)
    return sorted(set(out))


def load_pair(root, a, b):
    """{(prompt, seed): {metric: deviation, sharp_raw, jitter_raw, lpips_max}}"""
    out = {}
    for f in compare_files(root, a, b):
        clip = os.path.basename(f).split(f"{a}--{b}-", 1)[1][: -len(".json")]
        m = re.fullmatch(r"(.*)-s(\d+)", clip)
        if not m:
            continue
        d = json.load(open(f))
        if d.get("status") == "skipped":
            continue
        c = {x["name"]: x.get("values", {}) for x in d.get("checks", [])}
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


def denoise_median(root, arm):
    """Median per-clip denoise seconds of an arm's benchmark.json (`prompts`
    rows with denoise_s, else the top-level denoise_s), or None."""
    cands = glob.glob(f"{root}/{arm}/benchmark.json") + glob.glob(f"{root}/{arm}/**/benchmark.json", recursive=True)
    if not cands:
        return None
    b = json.load(open(cands[0]))
    rows = b.get("prompts") or {}
    vals = [r.get("denoise_s") for r in rows.values() if isinstance(r, dict) and r.get("denoise_s")]
    if vals:
        return float(np.median(vals))
    return b.get("denoise_s")


def judge(root, control, sage_pair):
    ctl = load_pair(root, *control)
    sage = load_pair(root, *sage_pair)
    alt = load_pair(root, control[1], sage_pair[1])
    print(f"## {os.path.basename(root.rstrip('/'))}: {len(ctl)} control pairs ({'--'.join(control)}), "
          f"{len(sage)} Sage pairs ({'--'.join(sage_pair)}), {len(alt)} {control[1]}-vs-{sage_pair[1]} pairs")
    passes, outliers, lines = 0, [], []
    for p in PROMPTS:
        seeds = sorted(s for (q, s) in ctl if q == p)
        sseeds = sorted(s for (q, s) in sage if q == p)
        if not seeds or not sseeds:
            lines.append(f"| {p} | missing | | | | | | |")
            continue
        C = {m: np.array([ctl[(p, s)][m] for s in seeds], float) for m in METRICS}
        S = {m: np.array([sage[(p, s)][m] for s in sseeds], float) for m in METRICS}
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
    print(f"\nraw (clip: {'--'.join(control)} | {'--'.join(sage_pair)} | {control[1]}--{sage_pair[1]}), lpips psnr sharp jitter:")
    for k in sorted(set(ctl) | set(sage)):
        def f(d):
            if k not in d:
                return "-"
            r = d[k]
            return f"{r['lpips']:.3f} {r['psnr']:.2f} {r['sharp_raw']:.3f} {r['jitter_raw']:.3f}"
        print(f"  {k[0]}-s{k[1]}: {f(ctl)} | {f(sage)} | {f(alt)}")
    mc, mf, ms = (denoise_median(root, a) for a in (control[0], control[1], sage_pair[1]))
    speed = (mc / ms) if mc and ms else float("nan")
    print(f"\ndenoise median s: {control[0]} {mc} {control[1]} {mf} {sage_pair[1]} {ms} -> sage speedup {speed:.3f}x")
    ok = passes >= MIN_PASS and not outliers and speed >= MIN_SPEEDUP
    print(f"\nrule 1: {passes}/{len(PROMPTS)} prompts inside the band (need {MIN_PASS}); rule 2 outliers: {outliers or 'none'}; "
          f"speed {speed:.3f}x (>= {MIN_SPEEDUP}) -> VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    global PROMPTS, MIN_PASS
    args = sys.argv[1:]
    control, sage_pair = ("cud", "fw2"), ("cud", "sage")
    while len(args) > 1 and args[0].startswith("--"):
        if args[0] == "--prompts":
            PROMPTS = args[1].split(",")
        elif args[0] == "--min-pass":
            MIN_PASS = int(args[1])
        elif args[0] == "--control":
            control = tuple(args[1].split(","))
        elif args[0] == "--sage":
            sage_pair = tuple(args[1].split(","))
        args = args[2:]
    if not args:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if judge(args[0], control, sage_pair) else 1)


if __name__ == "__main__":
    main()
