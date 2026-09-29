#!/usr/bin/env python3
"""Per-kernel GPU profile of one job from an Nsight Systems SQLite export.

    nsys export --type sqlite -o job.sqlite job.nsys-rep
    nsys_profile.py job.sqlite --label h3t-768 --out-dir prof/h3t-768 [--tail-s 10]

Writes (small, committable):
  summary.json   window vs GPU busy, idle-gap attribution (host sync / CUDA API /
                 host CPU / launch latency), gap histogram, category split,
                 CUDA API totals, cuBLAS / cuDNN host ranges, memcpy totals,
                 timeline segments
  kernels.csv    every kernel name: category, count, total ms, share, mean us,
                 most common grid/block, registers, shared memory
  timeline.csv   GPU busy fraction and per-category ms in 0.5 s bins
  kbins.csv      per-kernel ms in 0.25 s bins (stage re-analysis without the trace)
  gaps.csv       every idle gap >= 20 us with its attributed cause

The window is first-to-last device activity (kernels, memcpy, memset) of the
collection, or its last --tail-s seconds. docs/perf/datacenter-profile.md
explains the attribution; stdlib only (python3 + sqlite3).
"""

import argparse
import bisect
import collections
import csv
import json
import os
import re
import sqlite3
import sys

GAP_BUCKETS = [("<5us", 5e3), ("5-50us", 5e4), ("50-500us", 5e5), ("0.5-5ms", 5e6),
               ("5-50ms", 5e7), (">50ms", float("inf"))]

# Host calls that block until device work finishes.
SYNC_RE = re.compile(r"(Synchronize|cuMemcpyDtoH(_v2)?$|cuMemcpy(_v2)?$|cudaMemcpy$|cuMemcpyDtoD(_v2)?$|"
                     r"cuMemcpyHtoD(_v2)?$|cuMemcpy2D(_v2)?$|cuMemcpy3D(_v2)?$|cudaMemcpy2D$|"
                     r"cuMemFree(_v2)?$|cudaFree$)")
LAUNCH_RE = re.compile(r"(LaunchKernel|cuLaunch|cudaLaunch|GraphLaunch)")


def category(name):
    """Coarse bucket for a kernel (extends crates/.../wan/gpu_trace.rs `category`).

    `vae_conv` is convolution work (cuDNN / CUTLASS implicit-GEMM fprop, their
    layout transposes, group norm, upsample); the ViT-style H3 VAE decoder's
    GEMMs and attention land in `gemm` / `attention`, so VAE time is split by
    stage window in docs/perf/datacenter-profile.md, not by category.
    """
    n = name.lower().replace("bcast", "bcst")
    has = lambda *ks: any(k in n for k in ks)  # noqa: E731
    if n.startswith(("memcpy", "memset")):
        return "copies_memset"
    if has("attn", "flash", "fmha", "sdpa", "vsa_", "sol_", "pisa", "softmax", "attention", "fa_dc"):
        return "attention"
    if has("fprop", "convolve", "winograd", "dgrad", "conv2d", "conv3d", "conv_", "fft2d", "fft3d",
           "implicit_gemm", "nchwtonhwc", "nhwctonchw", "cudnn", "group_norm", "upsample", "temporal_unfold",
           "d2s", "rms_norm_channels", "snake", "ltxv_"):
        return "vae_conv"
    if has("gemm", "gemv", "cutlass", "xmma", "nvjet", "cublas", "matmul", "splitk", "wgmma", "tensorop",
           "s16816", "sm80_", "sm90_", "sm100_", "sm120_", "ampere_", "hopper", "blackwell"):
        return "gemm"
    if has("cast", "convert", "quantiz", "quant_", "dequant", "e4m3", "fp8", "mxfp8", "nvfp4", "w8a8",
           "_bf16_f32", "_f32_bf16", "amax"):
        return "cast_quant"
    if has("copy", "gather", "scatter", "split_heads", "merge_heads", "qkv_heads", "pad_axis", "repeat",
           "permute", "transpose", "index_", "unfold", "concat", "fill", "reduce", "pack_rgb"):
        return "layout_copy"
    if has("norm", "mod", "adaln", "rope", "rotary", "gate", "swiglu", "gelu", "silu", "elem", "bcst",
           "binary", "unary", "add", "mul", "sub", "lincomb", "scalar", "residual", "bias", "sigmoid",
           "tanh", "clamp", "abs", "act", "scale", "topk", "_ln", "ln_"):
        return "norm_elementwise"
    return "other"


def short(name, cap=140):
    s = name[5:] if name.startswith("void ") else name
    return s if len(s) <= cap else s[:cap] + "…"


def load(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    strings = dict(cur.execute("select id, value from StringIds"))
    acts = []  # (start, end, kind, name, corr, extra)
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        for r in cur.execute("select start, end, demangledName, correlationId, gridX, gridY, gridZ, blockX, blockY,"
                             " blockZ, registersPerThread, staticSharedMemory + dynamicSharedMemory, graphNodeId,"
                             " streamId from CUPTI_ACTIVITY_KIND_KERNEL"):
            acts.append((r[0], r[1], "kernel", strings.get(r[2], "?"), r[3],
                         {"grid": (r[4], r[5], r[6]), "block": (r[7], r[8], r[9]), "regs": r[10], "smem": r[11],
                          "graph": r[12] is not None, "stream": r[13]}))
    kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP", 3: "HtoA", 4: "AtoH", 11: "HtoH"}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for r in cur.execute("select start, end, copyKind, bytes, correlationId, streamId"
                             " from CUPTI_ACTIVITY_KIND_MEMCPY"):
            acts.append((r[0], r[1], "memcpy", "memcpy " + kinds.get(r[2], str(r[2])), r[4],
                         {"bytes": r[3], "stream": r[5]}))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for r in cur.execute("select start, end, bytes, correlationId, streamId from CUPTI_ACTIVITY_KIND_MEMSET"):
            acts.append((r[0], r[1], "memset", "memset", r[3], {"bytes": r[2], "stream": r[4]}))
    apis = []  # (start, end, name, corr, tid)
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        for r in cur.execute("select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
            apis.append((r[0], r[1], strings.get(r[2], "?"), r[3], r[4]))
    lib = []  # (start, end, lib, name)
    for t in ("CUBLAS_EVENTS", "CUDNN_EVENTS"):
        if t in tables:
            for r in cur.execute(f"select start, end, nameId from {t}"):
                lib.append((r[0], r[1], t.split("_")[0].lower(), strings.get(r[2], "?")))
    con.close()
    acts.sort(key=lambda a: (a[0], a[1]))
    apis.sort()
    return acts, apis, lib


def union(acts):
    out = []  # [start, end, first_idx, last_idx]
    for i, a in enumerate(acts):
        s, e = a[0], max(a[0], a[1])
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
                out[-1][3] = i
        else:
            out.append([s, e, i, i])
    return out


def analyze(acts, apis, lib, tail_s=None):
    if not acts:
        return {"error": "no device activity"}, [], []
    if tail_s:
        t_end = max(a[1] for a in acts)
        acts = [a for a in acts if a[0] >= t_end - tail_s * 1e9]
    first = acts[0][0]
    last = max(a[1] for a in acts)
    window = last - first
    busy = union(acts)
    busy_ns = sum(b[1] - b[0] for b in busy)
    by_corr = {}
    api_start = [a[0] for a in apis]
    for a in apis:
        if a[3] is not None:
            by_corr.setdefault(a[3], a)
    syncs = [a for a in apis if SYNC_RE.search(a[2]) and first <= a[1] <= last + 1]
    sync_end = sorted(a[1] for a in syncs)
    sync_by_end = sorted(syncs, key=lambda a: a[1])

    hist = [[0, 0] for _ in GAP_BUCKETS]
    attrib = collections.Counter()   # class -> ns of idle
    attrib_n = collections.Counter()
    gaps = []
    for b0, b1 in zip(busy, busy[1:]):
        g0, g1 = b0[1], b1[0]
        g = g1 - g0
        if g <= 0:
            continue
        k = next(i for i, (_, hi) in enumerate(GAP_BUCKETS) if g < hi)
        hist[k][0] += 1
        hist[k][1] += g
        nxt = acts[b1[2]]
        api = by_corr.get(nxt[4])
        if api is None or api[0] <= g0:
            # Work was already queued when the GPU went idle: launch / dependency latency.
            cls, host_ns = "launch_latency", 0
        else:
            host_ns = min(api[0], g1) - g0
            # Did a blocking host call return inside the idle stretch (or just before it)?
            j = bisect.bisect_left(sync_end, g0 - 50_000)
            s_hit = None
            while j < len(sync_end) and sync_end[j] <= api[0]:
                s_hit = sync_by_end[j]
                j += 1
            if s_hit is not None:
                cls = "host_sync:" + s_hit[2]
            else:
                # Largest CUDA API call overlapping [g0, api.start).
                lo = bisect.bisect_left(api_start, g0 - 5_000_000_000)
                ov = collections.Counter()
                hi_t = api[0]
                i = lo
                while i < len(apis) and apis[i][0] < hi_t:
                    a = apis[i]
                    o = min(a[1], hi_t) - max(a[0], g0)
                    if o > 0 and a is not api:
                        ov[a[2]] += o
                    i += 1
                if ov and ov.most_common(1)[0][1] > 0.5 * host_ns:
                    cls = "host_api:" + ov.most_common(1)[0][0]
                else:
                    cls = "host_cpu"
            attrib["launch_latency"] += g - host_ns
        attrib[cls] += host_ns if cls != "launch_latency" else g
        attrib_n[cls] += 1
        gaps.append((g, g0 - first, cls, acts[b0[3]][3], nxt[3]))

    # Category / kernel stats.
    kern = {}
    cat = collections.Counter()
    cat_n = collections.Counter()
    mem = collections.Counter()
    mem_b = collections.Counter()
    mem_n = collections.Counter()
    graph_ns = 0
    for a in acts:
        d = a[1] - a[0]
        if a[2] == "kernel":
            c = category(a[3])
            e = kern.setdefault(a[3], {"n": 0, "ns": 0, "cfg": collections.Counter(), "regs": a[5]["regs"],
                                       "smem": a[5]["smem"], "cat": c, "min": d, "max": d})
            e["n"] += 1
            e["ns"] += d
            e["min"] = min(e["min"], d)
            e["max"] = max(e["max"], d)
            e["cfg"][(a[5]["grid"], a[5]["block"])] += 1
            cat[c] += d
            cat_n[c] += 1
            if a[5]["graph"]:
                graph_ns += d
        else:
            key = a[3]
            mem[key] += d
            mem_b[key] += a[5]["bytes"]
            mem_n[key] += 1
            cat["copies_memset"] += d
            cat_n["copies_memset"] += 1
    dev_ns = sum(cat.values())

    rows = []
    for name, e in sorted(kern.items(), key=lambda kv: -kv[1]["ns"]):
        (grid, block), ncfg = e["cfg"].most_common(1)[0]
        rows.append({"name": short(name, 300), "category": e["cat"], "count": e["n"],
                     "total_ms": round(e["ns"] / 1e6, 3), "pct_device": round(100 * e["ns"] / dev_ns, 2),
                     "mean_us": round(e["ns"] / e["n"] / 1e3, 2), "min_us": round(e["min"] / 1e3, 2),
                     "max_us": round(e["max"] / 1e3, 2), "grid": "x".join(map(str, grid)),
                     "block": "x".join(map(str, block)), "cfg_share": round(ncfg / e["n"], 2),
                     "n_cfgs": len(e["cfg"]), "regs": e["regs"], "smem": e["smem"]})

    # Segments: device activity split at idle gaps > 20 ms.
    segs = []
    cur = None
    for b in busy:
        if cur is None or b[0] - cur["end"] > 20_000_000:
            if cur:
                segs.append(cur)
            cur = {"start": b[0], "end": b[1], "busy": 0, "first": b[2], "last": b[3]}
        cur["end"] = max(cur["end"], b[1])
        cur["busy"] += b[1] - b[0]
        cur["last"] = b[3]
    segs.append(cur)
    seg_out = []
    for s in segs:
        c = collections.Counter()
        top = collections.Counter()
        n = 0
        for a in acts[s["first"]:s["last"] + 1]:
            d = a[1] - a[0]
            c[category(a[3])] += d
            top[a[3]] += d
            n += 1
        tot = sum(c.values()) or 1
        seg_out.append({"t0_s": round((s["start"] - first) / 1e9, 3), "dur_s": round((s["end"] - s["start"]) / 1e9, 3),
                        "busy_pct": round(100 * s["busy"] / max(1, s["end"] - s["start"]), 1), "n": n,
                        "mix": {k: round(100 * v / tot, 1) for k, v in c.most_common(4)},
                        "top": short(top.most_common(1)[0][0], 80)})
    if len(seg_out) > 60:
        seg_out = seg_out[:30] + [{"note": f"{len(seg_out) - 60} segments omitted"}] + seg_out[-30:]

    # Timeline bins.
    bin_ns = 500_000_000
    nb = int(window // bin_ns) + 1
    tl = [collections.Counter() for _ in range(nb)]
    for a in acts:
        c = category(a[3])
        s, e = a[0] - first, a[1] - first
        while s < e:
            k = int(s // bin_ns)
            edge = min(e, (k + 1) * bin_ns)
            tl[k][c] += edge - s
            s = edge
    busy_bins = [0] * nb
    for b in busy:
        s, e = b[0] - first, b[1] - first
        while s < e:
            k = int(s // bin_ns)
            edge = min(e, (k + 1) * bin_ns)
            busy_bins[k] += edge - s
            s = edge
    cats = ["gemm", "attention", "norm_elementwise", "cast_quant", "layout_copy", "vae_conv", "copies_memset", "other"]
    timeline = [{"t_s": round(k * bin_ns / 1e9, 1), "busy_pct": round(100 * busy_bins[k] / bin_ns, 1),
                 **{c: round(tl[k][c] / 1e6, 1) for c in cats}} for k in range(nb)]

    # CUDA API totals in the window.
    api_tot = collections.Counter()
    api_n = collections.Counter()
    launches = 0
    for a in apis:
        if first - 1_000_000_000 <= a[0] <= last:
            api_tot[a[2]] += a[1] - a[0]
            api_n[a[2]] += 1
            if LAUNCH_RE.search(a[2]):
                launches += 1
    lib_tot = collections.Counter()
    lib_n = collections.Counter()
    for s0, e0, lb, nm in lib:
        if first <= s0 <= last:
            lib_tot[f"{lb}:{nm}"] += e0 - s0
            lib_n[f"{lb}:{nm}"] += 1

    small = sum(1 for a in acts if a[2] == "kernel" and a[1] - a[0] < 10_000)
    nk = sum(1 for a in acts if a[2] == "kernel")
    summary = {
        "window_s": round(window / 1e9, 4),
        "gpu_busy_s": round(busy_ns / 1e9, 4),
        "gpu_busy_pct": round(100 * busy_ns / window, 2),
        "idle_s": round((window - busy_ns) / 1e9, 4),
        "device_time_s": round(dev_ns / 1e9, 4),
        "kernels": nk,
        "kernels_under_10us": small,
        "kernel_time_in_graphs_pct": round(100 * graph_ns / max(1, dev_ns), 1),
        "host_launch_calls": launches,
        "idle_attribution_s": {k: round(v / 1e9, 4) for k, v in attrib.most_common()},
        "idle_attribution_n": dict(attrib_n.most_common()),
        "gap_histogram": {GAP_BUCKETS[i][0]: {"n": h[0], "s": round(h[1] / 1e9, 4)} for i, h in enumerate(hist)},
        "largest_gaps": [{"gap_ms": round(g[0] / 1e6, 3), "at_s": round(g[1] / 1e9, 3), "cause": g[2],
                          "before": short(g[3], 90), "after": short(g[4], 90)}
                         for g in sorted(gaps, key=lambda g: -g[0])[:15]],
        "category_s": {k: round(v / 1e9, 4) for k, v in cat.most_common()},
        "category_pct": {k: round(100 * v / dev_ns, 2) for k, v in cat.most_common()},
        "memcpy": {k: {"n": mem_n[k], "ms": round(mem[k] / 1e6, 3), "MB": round(mem_b[k] / 1e6, 1),
                       "GBps": round(mem_b[k] / max(1, mem[k]), 2)} for k in mem},
        "cuda_api_top": [{"api": k, "n": api_n[k], "s": round(v / 1e9, 4)} for k, v in api_tot.most_common(15)],
        "library_host_ranges": [{"range": k, "n": lib_n[k], "s": round(v / 1e9, 4)} for k, v in lib_tot.most_common(12)],
        "segments": seg_out,
    }
    # Per-kernel time in 0.25 s bins and the gap list, for stage re-analysis.
    kb = collections.Counter()
    kn = collections.Counter()
    for a in acts:
        k = int((a[0] - first) // 250_000_000)
        kb[(k, a[3])] += a[1] - a[0]
        kn[(k, a[3])] += 1
    summary["_kbins"] = [{"t_s": k * 0.25, "name": short(n, 160), "category": category(n),
                          "ms": round(v / 1e6, 4), "n": kn[(k, n)]} for (k, n), v in sorted(kb.items())]
    summary["_gaps"] = [{"t_s": round(g[1] / 1e9, 4), "gap_us": round(g[0] / 1e3, 1), "cause": g[2],
                         "before": short(g[3], 80), "after": short(g[4], 80)} for g in gaps if g[0] >= 20_000]
    return summary, rows, timeline


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db")
    p.add_argument("--label", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tail-s", type=float, default=None, help="analyze only the last N seconds of device activity")
    p.add_argument("--meta", default="{}", help="extra JSON merged into summary.json (gpu, job metrics, ...)")
    a = p.parse_args()
    acts, apis, lib = load(a.db)
    summary, rows, timeline = analyze(acts, apis, lib, a.tail_s)
    summary = {"label": a.label, **json.loads(a.meta), **summary}
    os.makedirs(a.out_dir, exist_ok=True)
    write_csv(os.path.join(a.out_dir, "kbins.csv"), summary.pop("_kbins", []))
    write_csv(os.path.join(a.out_dir, "gaps.csv"), summary.pop("_gaps", []))
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    write_csv(os.path.join(a.out_dir, "kernels.csv"), rows)
    write_csv(os.path.join(a.out_dir, "timeline.csv"), timeline)
    s = summary
    print(f"{a.label}: window {s.get('window_s')} s, busy {s.get('gpu_busy_pct')} %, kernels {s.get('kernels')}, "
          f"idle {s.get('idle_attribution_s')}", file=sys.stderr)
    for r in rows[:20]:
        print(f"  {r['pct_device']:6.2f}%  {r['total_ms']:10.1f} ms  n={r['count']:6d}  {r['mean_us']:9.1f} us"
              f"  [{r['category']}] {r['name'][:110]}", file=sys.stderr)


if __name__ == "__main__":
    main()
