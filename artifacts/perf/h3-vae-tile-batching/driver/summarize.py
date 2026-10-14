#!/usr/bin/env python3
"""Per-cell table from the unpacked results: per clip and median of denoise,
video VAE (DecodeSplit vae_s), video decode (whole decode + mp4), generate.
Usage: summarize.py <results dir> <cell>..."""
import json
import os
import statistics
import sys

root = sys.argv[1]


def clips(doc):
    ps = doc.get("prompts")
    if isinstance(ps, dict):
        return list(ps.items())
    if isinstance(ps, list):
        return [(p.get("workload", {}).get("prompt_name", str(i)), p) for i, p in enumerate(ps)]
    return [(doc.get("workload", {}).get("prompt_name", "?"), doc)]


print("| cell | clip | denoise s | video VAE s | video decode s | generate s | VAE share |")
print("|---|---|---|---|---|---|---|")
for cell in sys.argv[2:]:
    path = os.path.join(root, cell, "benchmark.json")
    if not os.path.exists(path):
        print(f"| {cell} | (no benchmark.json) | | | | | |")
        continue
    doc = json.load(open(path))
    rows = []
    for name, c in clips(doc):
        st = c.get("stage_seconds", {})
        r = (c.get("denoise_s"), st.get("video_vae"), st.get("video_decode"), c.get("total_s"))
        rows.append(r)
        print(f"| {cell} | {name} | {r[0]:.2f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.2f} | {r[1] / r[3]:.1%} |")
    med = [statistics.median(x) for x in zip(*rows)]
    print(f"| **{cell}** | median of {len(rows)} | {med[0]:.2f} | **{med[1]:.3f}** | {med[2]:.3f} | {med[3]:.2f} | {med[1] / med[3]:.1%} |")
