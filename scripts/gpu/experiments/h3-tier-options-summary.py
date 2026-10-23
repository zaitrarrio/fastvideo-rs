#!/usr/bin/env python3
"""Tables from an H3 NVFP4 tier-options run (`h3-tier-options-validate.sh fetch <pod> DIR`).

    h3-tier-options-summary.py DIR [--json OUT]

Prints markdown: per cell (arm x canvas) the warm medians over the prompt set (denoise, VAE decode, end to end,
peak device memory), the per-pair quality medians and worst cases (PSNR, LPIPS, sharpness and temporal-jitter
ratios), the gate verdicts, and the NVFP4 VAE check lines.
"""
import json, os, re, statistics, sys

root = sys.argv[1]
out = {"cells": {}, "pairs": {}, "gates": {}, "check": {}}


def load(p):
    with open(p) as f:
        return json.load(f)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


cells = os.path.join(root, "cells")
print("## Cells (warm; medians over the five prompts)\n")
print("| cell | denoise s | s / forward | VAE decode s | video decode s | end to end s (text cached) | peak GiB (in-process) | peak GiB (nvidia-smi) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for cell in sorted(os.listdir(cells)) if os.path.isdir(cells) else []:
    p = os.path.join(cells, cell, "benchmark.json")
    if not os.path.exists(p):
        continue
    b = load(p)
    prompts = b.get("prompts") or {}
    rows = list(prompts.values()) or [b]
    vae = med([(r.get("stage_seconds") or {}).get("video_vae") for r in rows])
    vdec = med([(r.get("stage_seconds") or {}).get("video_decode") for r in rows])
    den = med([r.get("denoise_s") for r in rows])
    # End to end with the text cached: the first cell streamed the encoder for four prompts.
    tot = med([r.get("total_s") - (r.get("text_s") or 0) for r in rows if r.get("total_s") is not None])
    steps = [s for r in rows for s in (r.get("step_seconds") or [])]
    fwd = med(steps)
    peak_in = max([r.get("max_device_memory_used_mib") or 0 for r in rows]) / 1024 or None
    smi = None
    sp = os.path.join(cells, cell, "nvsmi.csv")
    if os.path.exists(sp):
        vals = []
        for line in open(sp, errors="replace"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
        smi = max(vals) / 1024 if vals else None
    out["cells"][cell] = dict(denoise_s=den, forward_s=fwd, vae_s=vae, video_decode_s=vdec, total_s=tot,
                              peak_gib=peak_in, nvsmi_peak_gib=smi,
                              per_prompt={n: dict(denoise_s=r.get("denoise_s"), total_s=r.get("total_s"),
                                                  vae_s=(r.get("stage_seconds") or {}).get("video_vae"),
                                                  peak_mib=r.get("max_device_memory_used_mib"))
                                          for n, r in prompts.items()})
    print(f"| {cell} | {f(den)} | {f(fwd, 3)} | {f(vae)} | {f(vdec)} | {f(tot)} | {f(peak_in, 1)} | {f(smi, 1)} |")

cdir = os.path.join(root, "compare", "clips")
print("\n## Quality per pair (baseline -> candidate; median over prompts, [min..max])\n")
print("| pair | prompts | PSNR dB | LPIPS mean | sharpness ratio | jitter ratio |")
print("|---|---:|---|---|---|---|")
pairs = {}
for fn in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
    m = re.match(r"compare-clips-(.+?-\d{3})--(.+?-\d{3})-(.+)\.json$", fn)
    if not m:
        continue
    c = load(os.path.join(cdir, fn)).get("context", {})
    px = c.get("pixel_metrics", {})
    pairs.setdefault(f"{m[1]} -> {m[2]}", {})[m[3]] = dict(
        psnr=px.get("psnr_mean"), lpips=c.get("lpips_mean"), sharpness=px.get("sharpness_ratio_mean"),
        jitter=px.get("temporal_jitter_ratio_mean"))


def span(vals, d):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{statistics.median(vals):.{d}f} [{min(vals):.{d}f}..{max(vals):.{d}f}]"


for pair, per in pairs.items():
    out["pairs"][pair] = per
    print(f"| {pair} | {len(per)} | {span([r['psnr'] for r in per.values()], 2)} | "
          f"{span([r['lpips'] for r in per.values()], 3)} | {span([r['sharpness'] for r in per.values()], 3)} | "
          f"{span([r['jitter'] for r in per.values()], 3)} |")

print("\n## Gate (scripts/gpu/gate-policy.toml, kind lossy)\n")
print("| baseline -> candidate | verdict | denoise speed-up | total speed-up | quality | failed hard checks |")
print("|---|---|---:|---:|---|---|")
gdir = os.path.join(root, "compare", "gate")
for fn in sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []:
    m = re.match(r"gate-(.+?-\d{3})--(.+?-\d{3})\.json$", fn)
    if not m:
        continue
    v = load(os.path.join(gdir, fn)).get("context", {}).get("verdict", {})
    gates = v.get("gates", {})
    perf = gates.get("performance", {}).get("detail", {})
    q = gates.get("quantitative_quality", {})
    fails = [f"{c['check']} {c.get('value', '')!s:.6} ({c.get('report', '').split('--')[-1].replace('.json', '')})"
             for c in q.get("detail", {}).get("checks", []) if c.get("hard") and not c.get("ok")]
    out["gates"][f"{m[1]} -> {m[2]}"] = dict(verdict=v.get("verdict"), speedup=perf.get("speedup"),
                                             secondary=perf.get("secondary_speedup"), quality=q.get("status"),
                                             fails=fails)
    print(f"| {m[1]} -> {m[2]} | {v.get('verdict')} | {f(perf.get('speedup'), 3)} | "
          f"{f(perf.get('secondary_speedup'), 3)} | {q.get('status')} | {'; '.join(fails) or '-'} |")

print("\n## NVFP4 VAE check (FASTVIDEO_H3_VAE_CHECK, first tile batch, h3-base bf16 decoder as reference)\n")
for act in ("static", "dynamic"):
    p = os.path.join(root, "check", act, "vae-lines.txt")
    if os.path.exists(p):
        lines = [l.strip() for l in open(p, errors="replace") if "check" in l or "NVFP4" in l]
        out["check"][act] = lines
        print(f"- **{act}**:")
        for l in lines:
            print(f"  - `{l[-900:]}`")

if "--json" in sys.argv:
    with open(sys.argv[sys.argv.index("--json") + 1], "w") as fh:
        json.dump(out, fh, indent=1)
