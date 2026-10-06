#!/usr/bin/env python3
"""Five-prompt H3 check against base (docs/serve/research-longlive.md 12.6),
written before the runs. Reads compare-clips JSONs and benchmark.json files.

    plugcmp.py <results dir>

<dir>/compare/compare-clips-base--<arm>-<prompt>-s<seed>.json   arm in plug max turbo
<dir>/compare/compare-clips-plug--plugfw2-<prompt>-s<seed>.json   the control
<dir>/<arm>/benchmark.json
"""
import glob
import json
import math
import re
import sys

import numpy as np

PROMPTS = ["h3-demo", "ltx-multishot", "ltx-newsbroadcast", "ltx-frogyoga", "spark-mountain-lake"]
OWN_SEED = {"h3-demo": 0}
ARMS = ["plug", "max", "turbo"]
D = {"sharp": 0.02, "jitter": 0.03}


def load(root, a, b):
    out = {}
    for f in sorted(glob.glob(f"{root}/compare/compare-clips-{a}--{b}-*.json")):
        clip = f.split(f"{a}--{b}-", 1)[1][: -len(".json")]
        m = re.fullmatch(r"(.*)-s(\d+)", clip)
        if not m:
            continue
        c = {x["name"]: x["values"] for x in json.load(open(f)).get("checks", [])}
        pm, lp = c.get("pixel_metrics", {}), c.get("lpips", {})
        sh, ji = pm.get("sharpness_ratio_mean"), pm.get("temporal_jitter_ratio_mean")
        out[(m.group(1), int(m.group(2)))] = {"lpips": lp.get("mean"), "psnr": pm.get("psnr_mean"), "sharp": sh, "jitter": ji}
    return out


def times(root, arm):
    try:
        b = json.load(open(f"{root}/{arm}/benchmark.json"))
    except OSError:
        return None, None
    rows = (b.get("prompts") or {}).values()
    med = lambda k: float(np.median([r[k] for r in rows if r.get(k)])) if any(r.get(k) for r in rows) else None
    return med("denoise_s"), med("total_s")


def main():
    root = sys.argv[1]
    ctl = load(root, "plug", "plugfw2")
    cmax = {m: max(abs(math.log(r[m])) for r in ctl.values()) for m in ("sharp", "jitter")} if ctl else {}
    bound = {m: max(1.5 * cmax[m], cmax[m] + D[m]) for m in cmax}
    print(f"control plug vs plugfw2: {len(ctl)} clips; LPIPS {min(r['lpips'] for r in ctl.values()):.3f}-{max(r['lpips'] for r in ctl.values()):.3f}, "
          f"PSNR {min(r['psnr'] for r in ctl.values()):.2f}-{max(r['psnr'] for r in ctl.values()):.2f}, "
          f"Cmax |ln sharp| {cmax['sharp']:.3f} |ln jitter| {cmax['jitter']:.3f} -> bound {bound['sharp']:.3f} / {bound['jitter']:.3f} "
          f"(sharpness {math.exp(-bound['sharp']):.3f}-{math.exp(bound['sharp']):.3f}, jitter {math.exp(-bound['jitter']):.3f}-{math.exp(bound['jitter']):.3f})")
    data = {a: load(root, "base", a) for a in ARMS}
    med = {}
    print("\n| prompt | arm | LPIPS med (same seed) | PSNR med | sharpness ratio med | jitter ratio med | rule 2 |")
    print("|---|---|---|---|---|---|---|")
    r2 = {a: [] for a in ARMS}
    for p in PROMPTS:
        own = OWN_SEED.get(p, 42)
        for a in ARMS:
            rows = [v for (q, s), v in data[a].items() if q == p]
            if not rows:
                print(f"| {p} | {a} | missing | | | | |")
                continue
            m = {k: float(np.median([r[k] for r in rows])) for k in ("lpips", "psnr")}
            # medians of |ln| for rule 2; signed medians of the ratio for the table
            m["lsharp"] = float(np.median([abs(math.log(r["sharp"])) for r in rows]))
            m["ljit"] = float(np.median([abs(math.log(r["jitter"])) for r in rows]))
            m["sharp"] = float(np.median([r["sharp"] for r in rows]))
            m["jitter"] = float(np.median([r["jitter"] for r in rows]))
            same = data[a].get((p, own), {}).get("lpips")
            med[(p, a)] = m
            bad = []
            if bound and m["lsharp"] > bound["sharp"]:
                bad.append(f"sharpness |ln| {m['lsharp']:.3f}")
            if bound and m["ljit"] > bound["jitter"]:
                bad.append(f"jitter |ln| {m['ljit']:.3f}")
            if bad:
                r2[a].append(f"{p}: " + ", ".join(bad))
            print(f"| {p} | {a} | {m['lpips']:.3f} ({same if same is None else f'{same:.3f}'}) | {m['psnr']:.2f} | {m['sharp']:.3f} | {m['jitter']:.3f} | {'out: ' + ', '.join(bad) if bad else 'in'} |")
    closer = [p for p in PROMPTS if (p, "plug") in med and (p, "max") in med and med[(p, "plug")]["lpips"] < med[(p, "max")]["lpips"]]
    print(f"\nrule 1: plug closer than max (lower median LPIPS) on {len(closer)}/5: {closer}")
    for a in ARMS:
        print(f"rule 2 {a}: {r2[a] or 'no outliers'}")
    print("\nraw (clip: lpips psnr sharp jitter) per arm vs base:")
    for a in ARMS + ["control"]:
        d = ctl if a == "control" else data[a]
        for k in sorted(d):
            r = d[k]
            print(f"  {a} {k[0]}-s{k[1]}: {r['lpips']:.3f} {r['psnr']:.2f} {r['sharp']:.3f} {r['jitter']:.3f}")
    print()
    for a in ["base"] + ARMS + ["plugfw2"]:
        dn, tot = times(root, a)
        print(f"time {a}: denoise median {dn} s, total median {tot} s")
    ok = len(closer) >= 4 and not r2["plug"]
    print(f"\nVERDICT h3-plug-4step vs h3-max (rules 1+2): {'PASS: plug is the better h3-max candidate' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
