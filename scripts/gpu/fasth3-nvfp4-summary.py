#!/usr/bin/env python3
"""Tables from a FastH3 NVFP4 apples-to-apples run (`fasth3-nvfp4-validate.sh fetch <pod> DIR`).

    fasth3-nvfp4-summary.py DIR [--json OUT]

Prints markdown: upstream FastVideo's runs (wall, denoise, s / forward, decode, device peak from the pod's
nvidia-smi trace over each run's window), ours per arm and prompt (warm, in-process), the parity of ours on the
reference's noise and text (per-step latents, clip PSNR / LPIPS, the lossy gate's hard quality checks), and the
per-block phase profile.
"""
import json, os, re, statistics, sys

root = sys.argv[1]
out = {"upstream": {}, "ours": {}, "parity": {}, "profile": {}}
FORWARDS = {"v2": 8, "4s": 4}
# scripts/gpu/gate-policy.toml, kind lossy: the hard quality windows.
SHARP = (0.95, 1.08)
JITTER = (0.85, 1.20)


def load(p):
    with open(p) as f:
        return json.load(f)


def nvsmi_peak(t0, t1):
    p = os.path.join(root, "nvsmi.csv")
    if not os.path.exists(p):
        return None
    peak = None
    for line in open(p, errors="replace"):
        parts = [x.strip() for x in line.split(",")]
        try:
            t, mib = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        if t0 - 1 <= t <= t1 + 1:
            peak = mib if peak is None else max(peak, mib)
    return peak


def stage(stages, *keys):
    tot = 0.0
    hit = False
    for k, v in (stages or {}).items():
        if v and any(s in k.lower() for s in keys):
            tot += v
            hit = True
    return tot if hit else None


print("## Upstream FastVideo (d5253fd, the RTX PRO 6000 page's recipe)\n")
print("| run | ok | wall s | denoise s | s / forward | decode s | device peak MiB |")
print("|---|---|---:|---:|---:|---:|---:|")
for ck in ("v2", "4s"):
    p = os.path.join(root, f"up-{ck}", "result.json")
    if not os.path.exists(p):
        continue
    r = load(p)
    out["upstream"][ck] = {"load_s": r.get("load_s"), "runs": []}
    # A request that failed after denoising (FastVideo d5253fd's compiled VAE asserts at a second shape) has no
    # stages in its result: its denoise time is the stage log's "completed in" line, in request order.
    logged = []
    lp = os.path.join(root, f"up-{ck}.log")
    if os.path.exists(lp):
        logged = [float(m[1]) / 1000 for m in re.finditer(r"\[denoising_stage\|[^\]]*\] Execution completed in ([0-9.]+) ms",
                                                          open(lp, errors="replace").read())]
    for i, run in enumerate(r.get("runs", [])):
        st = run.get("stages_s") or {}
        den = stage(st, "denois")
        if den is None and i < len(logged) and run.get("steps") != 8:
            den = logged[i]
        dec = stage(st, "decod")
        peak = nvsmi_peak(run.get("t_start", 0), run.get("t_end", 0))
        row = dict(tag=run["tag"], ok=run.get("ok"), wall_s=run.get("wall_s"), denoise_s=den,
                   forward_s=den / FORWARDS[ck] if den else None, decode_s=dec, peak_mib=peak,
                   error=run.get("error"), stages=st)
        out["upstream"][ck]["runs"].append(row)
        f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"
        err = (run.get("error") or "").splitlines()
        why = "refused (grid points)" if run.get("steps") == 8 else ("decode failed (upstream VAE compile)" if den else (err[0] if err else ""))
        print(f"| {run['tag']} | {'yes' if run.get('ok') else 'no: ' + why[:90]} | {f(row['wall_s'])} | "
              f"{f(den)} | {f(row['forward_s'])} | {f(dec)} | {f(peak, 0)} |")
    print(f"\nload {r.get('load_s', 0):.1f} s ({ck})\n")

print("\n## Ours (fv-gpucheck h3 gen, one process per arm, warm after the 480p h3-demo generation)\n")
print("| checkpoint | arm | prompt | denoise s | s / forward (median) | decode s | total s | device peak MiB |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for ck in ("v2", "4s"):
    for arm in ("nvfp4", "nvfp4ffn-mxfp8", "dense-mxfp8"):
        p = os.path.join(root, f"ours-{ck}", arm, "benchmark.json")
        if not os.path.exists(p):
            continue
        b = load(p)
        rows = {}
        for name, v in (b.get("prompts") or {}).items():
            steps = v.get("step_seconds") or []
            row = dict(denoise_s=v.get("denoise_s"), forward_s=statistics.median(steps) if steps else None,
                       decode_s=v.get("decode_s"), total_s=v.get("total_s"),
                       peak_mib=v.get("max_device_memory_used_mib") or v.get("peak_memory_mb"))
            rows[name] = row
            print(f"| {ck} | {arm} | {name} | {row['denoise_s']:.2f} | {row['forward_s']:.3f} | {row['decode_s']:.2f} | "
                  f"{row['total_s']:.2f} | {row['peak_mib'] or 0:.0f} |")
        out["ours"].setdefault(ck, {})[arm] = {"prompts": rows, "warmup": b.get("warmup_requests"),
                                               "load_s": b.get("load_s")}

print("\n## Parity: ours (NVFP4 attention + FFN, VSA) on the reference's noise and text, 832x480x124\n")
for ck in ("v2", "4s"):
    p = os.path.join(root, f"compare-dumps-{ck}.txt")
    if not os.path.exists(p):
        continue
    want = re.compile(r"^compare-dumps (\S+)\s+rel_l2 (\S+)\s+cosine (\S+)")
    rel = {}
    for line in open(p, errors="replace"):
        m = want.match(line)
        if m:
            rel[m[1]] = float(m[2])
    keys = [k for k in rel if re.fullmatch(r"(video|audio)_step\d\d|text_hidden|text_refined|step00_block_(0|1|12|24|36|49)|video_vel_step01", k)]
    print(f"**{ck}**: " + ", ".join(f"{k} {rel[k]:.2e}" for k in sorted(keys)) + "\n")
    out["parity"][ck] = {"rel_l2": rel}
    a = os.path.join(root, f"av-psnr-{ck}.json")
    if os.path.exists(a):
        out["parity"][ck]["av_psnr"] = load(a)
        print(f"clip PSNR: {json.dumps(load(a))[:400]}\n")
    cdir = os.path.join(root, f"ours-{ck}", "compare")
    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if not f.endswith(".json"):
                continue
            c = load(os.path.join(cdir, f)).get("context", {})
            px = c.get("pixel_metrics", {})
            sh, ji = px.get("sharpness_ratio_mean"), px.get("temporal_jitter_ratio_mean")
            ok = sh is not None and ji is not None and SHARP[0] <= sh <= SHARP[1] and JITTER[0] <= ji <= JITTER[1]
            out["parity"][ck]["clip"] = dict(psnr=px.get("psnr_mean"), lpips=c.get("lpips_mean"), sharpness=sh,
                                             jitter=ji, gate_hard_quality="pass" if ok else "fail")
            print(f"clip vs upstream: PSNR {px.get('psnr_mean')}, LPIPS {c.get('lpips_mean')}, sharpness {sh}, "
                  f"jitter {ji} -> lossy gate hard quality checks: {'pass' if ok else 'FAIL'}\n")

print("\n## Per-block profile (ours, V2, 480p, FASTVIDEO_PROFILE=1 phases: synchronizing, so slower than timed)\n")
for ck in ("v2", "4s"):
    p = os.path.join(root, f"ours-{ck}", "profile", "phases", "..", "benchmark.json")
    log = os.path.join(root, f"ours-{ck}", "profile", "phases.log")
    if not os.path.exists(log):
        continue
    phases = {}
    for line in open(log, errors="replace"):
        m = re.search(r"\b(h3_[0-9a-z_]+)\b.*?calls[= ](\d+).*?(?:seconds|s)[= ]([0-9.]+)", line)
        if m:
            phases[m[1]] = (int(m[2]), float(m[3]))
    out["profile"][ck] = phases
    if phases:
        print("| phase | calls | total s | ms per call |")
        print("|---|---:|---:|---:|")
        for k, (c, s) in sorted(phases.items(), key=lambda kv: -kv[1][1]):
            print(f"| {k} | {c} | {s:.2f} | {1000 * s / max(c, 1):.2f} |")

if "--json" in sys.argv:
    with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
        json.dump(out, f, indent=1, default=str)
