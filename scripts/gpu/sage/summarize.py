#!/usr/bin/env python3
"""Markdown tables from bench.py results (JSONL, or a container log with "R " lines).

summarize.py LOG [--dense fv_fwd2,sdpa_cudnn]
"our dense" = the fastest of the --dense kernels per shape (on sm_12x our
`auto` picks cuDNN or fwd2 per shape, so min(fwd2, cuDNN) is what we run).
"""
import json
import sys
from collections import defaultdict

TOL_REL, TOL_COS = 5e-2, 0.998


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line.startswith("R "):
            line = line[2:]
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    path = sys.argv[1]
    dense = ["fv_fwd2", "sdpa_cudnn"]
    if "--dense" in sys.argv:
        dense = sys.argv[sys.argv.index("--dense") + 1].split(",")
    rows = load(path)
    t = {}
    acc = defaultdict(dict)
    shapes, kernels = [], []
    for r in rows:
        if "error" in r:
            print(f"<!-- error {r['shape']} {r['mode']} {r['kernel']}: {r['error'][:200]} -->")
        s, k = r["shape"], r["kernel"]
        if s not in shapes:
            shapes.append(s)
        if k not in kernels:
            kernels.append(k)
        if "ms" in r:
            t[(s, k)] = r
        if "vs_fp32" in r:
            acc[(s, k)][r["mode"]] = r
    gpu = rows[0]["gpu"] if rows else "?"
    print(f"GPU: {gpu}\n")
    # timing table
    print("| shape | heads x seq | kernel | ms | TFLOPS | vs our dense | vs FA2 | cos normal / peaked / outlier | rel-L2 normal / peaked / outlier (vs FP32) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in shapes:
        ours = [t[(s, k)]["ms"] for k in dense if (s, k) in t]
        base = min(ours) if ours else None
        fa2 = t.get((s, "sdpa_flash"), {}).get("ms")
        for k in kernels:
            r = t.get((s, k))
            a = acc.get((s, k), {})
            if r is None and not a:
                continue
            hs = f"{(r or next(iter(a.values())))['heads']} x {(r or next(iter(a.values())))['seq']}"
            ms = f"{r['ms']:.2f}" if r else "-"
            tf = f"{r['tflops']:.0f}" if r else "-"
            sp = f"{base / r['ms']:.2f}x" if r and base else "-"
            sf = f"{fa2 / r['ms']:.2f}x" if r and fa2 else "-"
            cos = " / ".join(f"{a[m]['vs_fp32']['cosine']:.5f}" if m in a else "-" for m in ("normal", "peaked", "outlier", "real") if m in a or m != "real")
            rel = " / ".join(f"{a[m]['vs_fp32']['rel_l2']:.4f}" if m in a else "-" for m in ("normal", "peaked", "outlier", "real") if m in a or m != "real")
            print(f"| {s} | {hs} | {k} | {ms} | {tf} | {sp} | {sf} | {cos} | {rel} |")
    # real captures
    real = [r for r in rows if r.get("mode") == "real" and "vs_fp32" in r]
    if real:
        print("\nReal FastWan 1.3B Q/K/V (12 x 32 760):\n")
        print("| capture | kernel | cosine vs FP32 | rel-L2 vs FP32 | max abs | rel-L2 vs fwd2 |")
        print("|---|---|---|---|---|---|")
        for r in real:
            v = r["vs_fp32"]
            w = r.get("vs_fwd2", {}).get("rel_l2", float("nan"))
            print(f"| {r['shape']} | {r['kernel']} | {v['cosine']:.5f} | {v['rel_l2']:.4f} | {v['max_abs']:.3g} | {w:.4f} |")
    # verdict
    print("\nVerdict per kernel (worst case over shapes and synthetic/real modes; tolerance rel-L2 <= 5e-2, cos >= 0.998 vs FP32; speed >= 1.3x our dense):\n")
    print("| kernel | min speedup | median speedup | max speedup | worst cosine | worst rel-L2 | passes accuracy | passes speed everywhere |")
    print("|---|---|---|---|---|---|---|---|")
    for k in kernels:
        sps = []
        for s in shapes:
            ours = [t[(s, d)]["ms"] for d in dense if (s, d) in t]
            if ours and (s, k) in t:
                sps.append(min(ours) / t[(s, k)]["ms"])
        errs = [r["vs_fp32"] for r in rows if r["kernel"] == k and "vs_fp32" in r]
        if not errs:
            continue
        wc = min(e["cosine"] for e in errs)
        wr = max(e["rel_l2"] for e in errs)
        sps.sort()
        med = sps[len(sps) // 2] if sps else float("nan")
        ok = wc >= TOL_COS and wr <= TOL_REL
        fast = bool(sps) and sps[0] >= 1.3
        print(f"| {k} | {sps[0] if sps else float('nan'):.2f}x | {med:.2f}x | {sps[-1] if sps else float('nan'):.2f}x | {wc:.5f} | {wr:.4f} | {'yes' if ok else 'no'} | {'yes' if fast else 'no'} |")


if __name__ == "__main__":
    main()
