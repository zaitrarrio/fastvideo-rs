#!/usr/bin/env python3
"""Plug-fast check against base H3 (docs/serve/research-longlive.md 12.8),
written before the runs. Reads compare-clips JSONs and benchmark.json files.

    pfcmp.py <results dir>

<dir>/compare/compare-clips-base--<arm>-<prompt>-s<seed>.json   arm in pfast max pdense
<dir>/compare/compare-clips-pdense--pfast-<prompt>-s<seed>.json  informational
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
ARMS = ["pfast", "max", "pdense"]
# 12.7's control band (plug vs plugfw2, 15 clips), fixed in 12.8.
BOUND = {"sharp": 0.152, "jitter": 0.253}
COST_X = 1.10


def load(root, a, b):
    out = {}
    for f in sorted(glob.glob(f"{root}/compare/compare-clips-{a}--{b}-*.json")):
        clip = f.split(f"{a}--{b}-", 1)[1][: -len(".json")]
        m = re.fullmatch(r"(.*)-s(\d+)", clip)
        if not m:
            continue
        c = {x["name"]: x["values"] for x in json.load(open(f)).get("checks", [])}
        pm, lp = c.get("pixel_metrics", {}), c.get("lpips", {})
        out[(m.group(1), int(m.group(2)))] = {
            "lpips": lp.get("mean"), "psnr": pm.get("psnr_mean"),
            "sharp": pm.get("sharpness_ratio_mean"), "jitter": pm.get("temporal_jitter_ratio_mean"),
        }
    return out


def times(root, arm):
    try:
        b = json.load(open(f"{root}/{arm}/benchmark.json"))
    except OSError:
        return None, None, 0
    rows = list((b.get("prompts") or {}).values())
    med = lambda k: float(np.median([r[k] for r in rows if r.get(k)])) if any(r.get(k) for r in rows) else None
    return med("denoise_s"), med("total_s"), len(rows)


def per_prompt(rows):
    m = {k: float(np.median([r[k] for r in rows])) for k in ("lpips", "psnr", "sharp", "jitter")}
    m["lsharp"] = float(np.median([abs(math.log(r["sharp"])) for r in rows]))
    m["ljit"] = float(np.median([abs(math.log(r["jitter"])) for r in rows]))
    m["n"] = len(rows)
    return m


def main():
    root = sys.argv[1]
    data = {a: load(root, "base", a) for a in ARMS}
    med = {}
    print(f"rule 2 band (12.7 control, fixed): |ln sharp| <= {BOUND['sharp']} ({math.exp(-BOUND['sharp']):.3f}-{math.exp(BOUND['sharp']):.3f}), "
          f"|ln jitter| <= {BOUND['jitter']} ({math.exp(-BOUND['jitter']):.3f}-{math.exp(BOUND['jitter']):.3f})")
    print("\n| prompt | arm | n | LPIPS med (same seed) | PSNR med | sharpness ratio med | jitter ratio med | rule 2 |")
    print("|---|---|---|---|---|---|---|---|")
    r2 = {a: [] for a in ARMS}
    for p in PROMPTS:
        own = OWN_SEED.get(p, 42)
        for a in ARMS:
            rows = [v for (q, s), v in data[a].items() if q == p]
            if not rows:
                print(f"| {p} | {a} | 0 | missing | | | | |")
                continue
            m = per_prompt(rows)
            med[(p, a)] = m
            same = data[a].get((p, own), {}).get("lpips")
            bad = []
            if m["lsharp"] > BOUND["sharp"]:
                bad.append(f"sharpness |ln| {m['lsharp']:.3f}")
            if m["ljit"] > BOUND["jitter"]:
                bad.append(f"jitter |ln| {m['ljit']:.3f}")
            if bad:
                r2[a].append(f"{p}: " + ", ".join(bad))
            print(f"| {p} | {a} | {m['n']} | {m['lpips']:.3f} ({same if same is None else f'{same:.3f}'}) | {m['psnr']:.2f} | "
                  f"{m['sharp']:.3f} | {m['jitter']:.3f} | {'out: ' + ', '.join(bad) if bad else 'in'} |")
    have = [p for p in PROMPTS if (p, "pfast") in med and (p, "max") in med]
    closer = [p for p in have if med[(p, "pfast")]["lpips"] < med[(p, "max")]["lpips"]]
    nowider = [p for p in have if med[(p, "pfast")]["lsharp"] <= med[(p, "max")]["lsharp"] and med[(p, "pfast")]["ljit"] <= med[(p, "max")]["ljit"]]
    rule1 = len(have) == 5 and len(closer) >= 4
    rule2 = len(have) == 5 and not r2["pfast"]
    print(f"\nrule 1: pfast closer than max (lower median LPIPS) on {len(closer)}/{len(have)}: {closer}")
    for a in ARMS:
        print(f"rule 2 {a}: {(r2[a] or 'no outliers') if data[a] else 'no data'}")
    print(f"R2 alt: pfast |ln sharp| and |ln jitter| both <= max's on {len(nowider)}/{len(have)}: {nowider}")

    print("\nraw (clip: lpips psnr sharp jitter) vs base:")
    for a in ARMS:
        for k in sorted(data[a]):
            r = data[a][k]
            print(f"  {a} {k[0]}-s{k[1]}: {r['lpips']:.3f} {r['psnr']:.2f} {r['sharp']:.3f} {r['jitter']:.3f}")
    pd = load(root, "pdense", "pfast")
    if pd:
        print("\npdense -> pfast (what the sparse route + MXFP8 move; informational):")
        for k in sorted(pd):
            r = pd[k]
            print(f"  {k[0]}-s{k[1]}: {r['lpips']:.3f} {r['psnr']:.2f} {r['sharp']:.3f} {r['jitter']:.3f}")
    print()
    t = {}
    for a in ["base"] + ARMS:
        dn, tot, n = times(root, a)
        t[a] = dn
        print(f"time {a}: denoise median {dn} s, total median {tot} s ({n} timed clips)")
    if t.get("pfast") and t.get("max"):
        print(f"pfast / max denoise: {t['pfast'] / t['max']:.3f}x ({t['pfast'] - t['max']:+.2f} s)")
    if t.get("pfast") and t.get("pdense"):
        print(f"pfast / pdense denoise: {t['pfast'] / t['pdense']:.3f}x ({t['pfast'] - t['pdense']:+.2f} s)")
    r3 = bool(t.get("pfast") and t.get("max") and t["pfast"] <= COST_X * t["max"])
    print(f"\nVERDICT plug-fast vs h3-max (rules 1+2): {'PASS' if rule1 and rule2 else 'FAIL'} (rule 1 {'holds' if rule1 else 'fails'}, rule 2 {'holds' if rule2 else 'fails'})")
    rec = rule1 and (rule2 or len(nowider) >= 4) and r3
    print(f"RECOMMENDATION (R1 {rule1}, R2 {rule2 or len(nowider) >= 4}, R3 {r3}): "
          f"{'switch h3-max to plug-fast (owner decides)' if rec else 'keep Sol-H3 as h3-max'}")


if __name__ == "__main__":
    main()
