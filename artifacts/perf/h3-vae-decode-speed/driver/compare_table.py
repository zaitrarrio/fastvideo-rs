#!/usr/bin/env python3
"""One row per compare-clips report (LPIPS, PSNR, sharpness, jitter) plus the
median per pair prefix. Usage: compare_table.py <results dir> [prefix...]"""
import glob
import json
import os
import statistics
import sys

root = sys.argv[1]
prefixes = sys.argv[2:]


def row(path):
    doc = json.load(open(path))
    out = {}
    for c in doc.get("checks", []):
        v = c.get("values", {})
        if c["name"] == "lpips":
            out["lpips_mean"], out["lpips_max"] = v.get("mean"), v.get("max")
        if c["name"] == "pixel_metrics":
            for k in ("psnr_mean", "psnr_min", "sharpness_ratio_mean", "temporal_jitter_ratio_mean"):
                out[k] = v.get(k)
    return out


keys = ["lpips_mean", "lpips_max", "psnr_mean", "psnr_min", "sharpness_ratio_mean", "temporal_jitter_ratio_mean"]
print("| pair | clip | " + " | ".join(keys) + " |")
print("|---|---|" + "---|" * len(keys))
groups = {}
for p in sorted(glob.glob(os.path.join(root, "compare", "compare-clips-*.json"))):
    name = os.path.basename(p)[len("compare-clips-"):-5]
    pair, _, clip = name.partition("-h3-") if "-h3-" in name else name.rpartition("-")
    for pre in prefixes or [None]:
        if pre and not name.startswith(pre + "-"):
            continue
        pair = pre or name
        clip = name[len(pre) + 1:] if pre else ""
        r = row(p)
        groups.setdefault(pair, []).append(r)
        fmt = lambda x: "—" if x is None else f"{x:.4g}"
        print(f"| {pair} | {clip} | " + " | ".join(fmt(r.get(k)) for k in keys) + " |")
for pair, rows in groups.items():
    med = []
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        med.append(f"{statistics.median(vals):.4g}" if vals else "—")
    mins = [r["sharpness_ratio_mean"] for r in rows if r.get("sharpness_ratio_mean") is not None]
    print(f"| **{pair}** | median of {len(rows)} | " + " | ".join(med) + f" | (sharpness min {min(mins):.4g})" if mins else "")
