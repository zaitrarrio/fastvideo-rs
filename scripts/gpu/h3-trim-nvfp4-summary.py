#!/usr/bin/env python3
"""Tables from a FastH3 Trim NVFP4 validation run (`h3-trim-nvfp4-validate.sh fetch <pod> DIR`).

    h3-trim-nvfp4-summary.py DIR [--json OUT]

Prints markdown: the speed / memory cells (warm medians over the prompt set), the per-prompt quality of each pair
(PSNR, LPIPS, sharpness and temporal-jitter ratios) and the gate verdicts, plus the parity run's per-step latents.
"""
import json, os, re, sys

root = sys.argv[1]
runs = os.path.join(root, "runs")
out = {"cells": {}, "pairs": {}, "gates": {}, "parity": {}}


def load(p):
    with open(p) as f:
        return json.load(f)


print("## Cells (warm; medians over the prompt set)\n")
print("| cell | denoise s | s / forward | total s | VAE decode s | peak MiB (nvidia-smi) | DiT resident GiB |")
print("|---|---:|---:|---:|---:|---:|---:|")
for res in ("480", "768"):
    for arm in ("bf16", "mxfp8", "nvfp4"):
        cell = f"{arm}-{res}"
        p = os.path.join(runs, cell, "benchmark.json")
        if not os.path.exists(p):
            continue
        b = load(p)
        steps = b.get("step_seconds") or []
        fwd = sorted(steps)[len(steps) // 2] if steps else float("nan")
        dit = None
        log = os.path.join(runs, cell, "stderr.log")
        if os.path.exists(log):
            m = re.findall(r"dit_nonlinear ([0-9.]+)", open(log, errors="replace").read())
            dit = float(m[-1]) if m else None
        # Per-prompt device peaks of the measured runs (the cold / text-encoding run excluded).
        peaks = [v.get("max_device_memory_used_mib") or v.get("peak_memory_mb") for v in (b.get("prompts") or {}).values()]
        peaks = [x for x in peaks if x]
        peak = min(peaks) if peaks else b.get("max_device_memory_used_mib")
        row = dict(denoise_s=b["denoise_s"], forward_s=fwd, total_s=b["total_s"], decode_s=b["decode_s"],
                   peak_mib=peak, peak_mib_max=max(peaks) if peaks else None, dit_gib=dit)
        out["cells"][cell] = row
        print(f"| {cell} | {b['denoise_s']:.2f} | {fwd:.2f} | {b['total_s']:.2f} | {b['decode_s']:.2f} | "
              f"{peak:.0f} | {dit if dit is not None else '?'} |")

print("\n## Quality per prompt (baseline -> candidate)\n")
print("| pair | prompt | PSNR dB | LPIPS mean | sharpness ratio | jitter ratio |")
print("|---|---|---:|---:|---:|---:|")
cdir = os.path.join(runs, "compare")
for f in sorted(os.listdir(cdir)):
    m = re.match(r"compare-clips-(.+?-\d+)--(.+?-\d+)-(.+)\.json$", f)
    if not m:
        continue
    c = load(os.path.join(cdir, f)).get("context", {})
    px = c.get("pixel_metrics", {})
    pair, prompt = f"{m[1]} -> {m[2]}", m[3]
    row = dict(psnr=px.get("psnr_mean"), lpips=c.get("lpips_mean"), sharpness=px.get("sharpness_ratio_mean"),
               jitter=px.get("temporal_jitter_ratio_mean"))
    out["pairs"].setdefault(pair, {})[prompt] = row
    print(f"| {pair} | {prompt} | {row['psnr']:.2f} | {row['lpips']:.3f} | {row['sharpness']:.3f} | {row['jitter']:.3f} |")

print("\n## Gate (scripts/gpu/gate-policy.toml, kind lossy)\n")
print("| baseline -> candidate | verdict | denoise speed-up | failed hard checks |")
print("|---|---|---:|---|")
gdir = os.path.join(runs, "gate")
for f in sorted(os.listdir(gdir)):
    m = re.match(r"gate-(.+?-\d+)--(.+?-\d+)\.json$", f)
    if not m:
        continue
    v = load(os.path.join(gdir, f)).get("context", {}).get("verdict", {})
    gates = v.get("gates", {})
    perf = gates.get("performance", {}).get("detail", {})
    fails = [f"{c['check']} ({c['report'].split('--')[-1]})"
             for c in gates.get("quantitative_quality", {}).get("detail", {}).get("checks", []) if c.get("hard") and not c.get("ok")]
    out["gates"][f"{m[1]} -> {m[2]}"] = dict(verdict=v.get("verdict"), speedup=perf.get("speedup"), fails=fails)
    print(f"| {m[1]} -> {m[2]} | {v.get('verdict')} | {perf.get('speedup', float('nan')):.3f} | {'; '.join(fails) or '-'} |")

print("\n## Parity (832x480, one prompt, seed 1024, the same cached text; rel-L2 against bf16)\n")
names = ["video_step01", "video_step02", "video_step04", "video_step08", "video_vel_step01", "audio_step01",
         "audio_step08", "step00_block_0", "step00_block_20", "step00_block_41"]
print("| tensor | " + " | ".join(names) + " |")
print("|---|" + "---:|" * len(names))
for arm in ("mxfp8", "nvfp4"):
    p = os.path.join(root, f"compare-dumps-bf16-{arm}.txt")
    vals = {}
    for line in open(p, errors="replace"):
        m = re.match(r"compare-dumps (\S+)\s+rel_l2 (\S+)", line)
        if m:
            vals[m[1]] = float(m[2])
    out["parity"][arm] = {n: vals.get(n) for n in names}
    print(f"| bf16 -> {arm} | " + " | ".join(f"{vals.get(n, float('nan')):.2e}" for n in names) + " |")
for arm in ("mxfp8", "nvfp4"):
    p = os.path.join(root, "parity", "compare", f"compare-clips-bf16--{arm}.json")
    if os.path.exists(p):
        c = load(p).get("context", {})
        px = c.get("pixel_metrics", {})
        out["parity"][arm + "_clip"] = dict(psnr=px.get("psnr_mean"), lpips=c.get("lpips_mean"))
        print(f"\nbf16 -> {arm} decoded clip: PSNR {px.get('psnr_mean'):.2f} dB, LPIPS {c.get('lpips_mean'):.3f}")

if "--json" in sys.argv:
    with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
        json.dump(out, f, indent=1)
