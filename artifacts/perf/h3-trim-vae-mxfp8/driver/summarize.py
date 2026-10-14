#!/usr/bin/env python3
"""Tables from the unpacked results.tgz of pod.sh: per resolution, the video VAE time of the MXFP8 decode and of the
bf16 re-decode of the same latents (the "h3 vae a/b" log lines, timed clips only), peak memory (benchmark.json's
video_decode stage and nvidia-smi) of the A/B cell and of the =0 cell, and the compare-clips metrics bf16 -> MXFP8.
Usage: summarize.py <results dir>"""
import glob
import json
import os
import re
import statistics
import sys

root = os.path.join(sys.argv[1], "runs")
med = statistics.median


def ab_times(cell):
    """(mxfp8 s, bf16 s) per timed clip: the warm-up clip's line comes first and is dropped."""
    lines = open(os.path.join(root, cell, "log.txt"), errors="replace").read().splitlines()
    pairs = []
    for line in lines:
        m = re.search(r"h3 vae a/b: video vae mxfp8 ([\d.]+)s, bf16 ([\d.]+)s \(\d+ frames to (\S+)\)", line)
        if m and "/warmup/" not in m.group(3) and "/cold/" not in m.group(3):
            pairs.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return pairs


def bench(cell):
    p = os.path.join(root, cell, "benchmark.json")
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    ps = d.get("prompts")
    if isinstance(ps, dict):
        return list(ps.values())
    return ps if isinstance(ps, list) else [d]


def decode_peak(clip):
    for s in clip.get("memory", {}).get("stages", []):
        if s["stage"] == "video_decode":
            return s["peak_allocated_gib"], s["peak_reserved_gib"]
    return None, None


def smi_peak(cell):
    p = os.path.join(root, cell, "smi.csv")
    vals = [int(x) for x in open(p).read().split() if x.isdigit()] if os.path.exists(p) else []
    return max(vals) if vals else None


print("## Video VAE time (same latents, same process; timed clips)\n")
print("| res | clips | MXFP8 median s | bf16 median s | change | MXFP8 range | bf16 range |")
print("|---|---|---|---|---|---|---|")
for r in (480, 768):
    t = ab_times(f"ab{r}")
    if not t:
        print(f"| {r}p | 0 | | | | | |")
        continue
    mx, bf = [a for a, _, _ in t], [b for _, b, _ in t]
    print(f"| {r}p | {len(t)} | **{med(mx):.3f}** | {med(bf):.3f} | **{med(mx) / med(bf) - 1:+.1%}** | "
          f"{min(mx):.3f}-{max(mx):.3f} | {min(bf):.3f}-{max(bf):.3f} |")

print("\n## Peak memory\n")
print("| cell | decoder linears | video_decode stage peak allocated / reserved GiB (median over clips) | whole-run peak allocated GiB | nvidia-smi peak MiB |")
print("|---|---|---|---|---|")
for cell, what in (("ab480", "MXFP8 (+ bf16 kept)"), ("v480", "bf16 (=0)"), ("ab768", "MXFP8 (+ bf16 kept)"), ("v768", "bf16 (=0)")):
    cs = bench(cell)
    dp = [decode_peak(c) for c in cs]
    dp = [x for x in dp if x[0] is not None]
    alloc = [c.get("peak_allocated_gib") for c in cs if c.get("peak_allocated_gib") is not None]
    a = f"{med([x[0] for x in dp]):.2f} / {med([x[1] for x in dp]):.2f}" if dp else "-"
    print(f"| {cell} | {what} | {a} | {max(alloc):.2f} | {smi_peak(cell)} |" if alloc else f"| {cell} | {what} | {a} | - | {smi_peak(cell)} |")

print("\n## Quality: MXFP8 decode against the bf16 decode of the same latents (compare-clips)\n")
keys = ["lpips_mean", "lpips_max", "psnr_mean", "psnr_min", "sharpness_ratio_mean", "temporal_jitter_ratio_mean"]
for r in (480, 768):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "compare", f"compare-clips-bf16-{r}--mx-{r}-*.json"))):
        doc = json.load(open(p))
        out = {"clip": os.path.basename(p)[len(f"compare-clips-bf16-{r}--mx-{r}-"):-5], "status": doc.get("status")}
        for c in doc.get("checks", []):
            v = c.get("values", {})
            if c["name"] == "lpips":
                out["lpips_mean"], out["lpips_max"] = v.get("mean"), v.get("max")
            if c["name"] == "pixel_metrics":
                for k in keys[2:]:
                    out[k] = v.get(k)
        rows.append(out)
    print(f"### {r}p ({len(rows)} clips)\n")
    print("| clip | status | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for o in rows:
        print(f"| {o['clip']} | {o['status']} | " + " | ".join("-" if o.get(k) is None else f"{o[k]:.4g}" for k in keys) + " |")
    if rows:
        def col(k):
            return [o[k] for o in rows if o.get(k) is not None]
        print(f"| **median** | | " + " | ".join(f"**{med(col(k)):.4g}**" if col(k) else "-" for k in keys) + " |")
        print(f"| worst | | {max(col('lpips_mean')):.4g} | {max(col('lpips_max')):.4g} | {min(col('psnr_mean')):.4g} | "
              f"{min(col('psnr_min')):.4g} | {min(col('sharpness_ratio_mean')):.4g}-{max(col('sharpness_ratio_mean')):.4g} | "
              f"{min(col('temporal_jitter_ratio_mean')):.4g}-{max(col('temporal_jitter_ratio_mean')):.4g} |")
    print()

print("## Gate (quality checks)\n")
for r in (480, 768):
    p = os.path.join(root, "gate", f"gate-{r}.out")
    if not os.path.exists(p):
        continue
    lines = open(p).read().splitlines()
    q = [l for l in lines if "/quality/" in l]
    hard = [l for l in q if l.startswith(("[PASS]", "[FAIL]"))]
    print(f"- {r}p: quality checks {sum(l.startswith('[PASS]') for l in hard)} / {len(hard)} pass "
          f"({sum(l.startswith('[INFO]') for l in q)} telemetry rows); failed: "
          + (", ".join(l.split()[1] for l in hard if l.startswith("[FAIL]")) or "none"))
