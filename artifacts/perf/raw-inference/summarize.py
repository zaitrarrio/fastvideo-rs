#!/usr/bin/env python3
"""Summarize /root/work/res: per job/arm, the 5 timed runs (median/min/max)."""
import glob, json, os, re, statistics, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "/root/work/res"


def stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    return [round(statistics.median(xs), 3), round(min(xs), 3), round(max(xs), 3)]


def smi_peak(job):
    try:
        vals = [int(l.split(",")[0]) for l in open(os.path.join(RES, job, "smi.csv")) if l.strip() and l.split(",")[0].strip().isdigit()]
        return max(vals) if vals else None
    except OSError:
        return None


out = {}
for path in sorted(glob.glob(f"{RES}/**/benchmark.json", recursive=True)):
    rel = os.path.relpath(os.path.dirname(path), RES)
    job = rel.split(os.sep)[0]
    doc = json.load(open(path))
    runs = doc.get("prompts") or {"r1": doc}
    names = sorted(runs)
    gen, tot, den, dec, txt, peak, ref = [], [], [], [], [], [], []
    for n in names:
        d = runs[n]
        t = d.get("total_s")
        ts = d.get("text_s") or 0.0
        if t is not None:
            tot.append(t)
            gen.append(t - ts)
        txt.append(ts)
        den.append(d.get("denoise_s"))
        dec.append(d.get("decode_s"))
        ss = d.get("stage_seconds") or {}
        ref.append(ss.get("refine") or ss.get("upsample"))
        peak.append(d.get("max_device_memory_used_mib") or d.get("peak_memory_mb"))
    out[rel] = {
        "n": len(names),
        "gen_minus_text_s": stats(gen),
        "total_s": stats(tot),
        "text_s": stats(txt),
        "denoise_s": stats(den),
        "decode_s": stats(dec),
        "peak_mib_runs2_5": max([p for p in peak[1:] if p] or [0]) or None,
        "peak_mib_all": max([p for p in peak if p] or [0]) or None,
        "smi_peak_mib": smi_peak(job),
        "load_s": doc.get("load_s"),
        "decoder": doc.get("decoder"),
        "text_cache": [runs[n].get("text_cache") for n in names],
        "workload": {k: (doc.get("workload") or {}).get(k) for k in ("height", "width", "num_frames")},
    }

# SF-Wan stream reports: steady fps per run.
for path in sorted(glob.glob(f"{RES}/sf-*/out/*.json")):
    job = os.path.relpath(path, RES).split(os.sep)[0]
    try:
        doc = json.load(open(path))
    except Exception:
        continue
    runs = {}
    def walk(v, key=""):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{key}/{k}" if key else k)
        elif isinstance(v, (int, float)) and re.search(r"(fps|frames_per_s|block.*p50|ttff|first_frame|wall|seconds|mib)", key, re.I):
            runs[key] = v
    walk(doc)
    out[f"{job}:{os.path.basename(path)}"] = {k: v for k, v in runs.items() if not k.startswith("checks")}

json.dump(out, sys.stdout, indent=1)
print()
