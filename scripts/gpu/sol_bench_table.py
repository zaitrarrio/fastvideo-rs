#!/usr/bin/env python3
"""Results table: our sol-bench cells next to sol-engine's published numbers.

    sol_bench_table.py <run dir>...   (each holding <cell>/benchmark.json)

Prints a Markdown table and writes results.json next to the first run dir's
parent. Published numbers: NVlabs/Sana branch sol-engine (models/ at 670482d),
see docs/perf/sol-bench.md.
"""
import json
import sys
from pathlib import Path

# cell -> (model, config, their hardware, their seconds or None, note)
PUBLISHED = {
    "h3-768p-dense": ("MiniMax-H3", "768p 124 f 50 st, dense (MXFP8 linears: our sm_100+ default)", "RTX 5090", 1045.4, "rtx5090_dense.toml is BF16"),
    "h3-768p-dense-bf16": ("MiniMax-H3", "768p 124 f 50 st, dense, BF16", "RTX 5090", 1045.4, ""),
    "h3-768p-fullopt": ("MiniMax-H3", "768p 124 f 50 st, fullopt (Sol + TeaCache) + MXFP8 linears", "RTX 5090", 231.2, "rtx5090_fullopt.toml is BF16"),
    "h3-768p-fullopt-bf16": ("MiniMax-H3", "768p 124 f 50 st, fullopt (Sol + TeaCache), BF16", "RTX 5090", 231.2, ""),
    "ltx25-4k5s-sol-bf16": ("LTX-2.5 distilled", "4K 5 s, Sol stage 2, BF16", "RTX 5090", 273.63, ""),
    "ltx25-1080p20s-sol-bf16": ("LTX-2.5 distilled", "1080p 20 s, Sol stage 2, BF16", "RTX 5090", 261.65, ""),
    "ltx25-4k5s-sol-nvfp4": ("LTX-2.5 distilled", "4K 5 s, Sol stage 2, NVFP4 video FFN", "RTX 5090", 171.82, ""),
    "ltx25-1080p20s-sol-nvfp4": ("LTX-2.5 distilled", "1080p 20 s, Sol stage 2, NVFP4 video FFN", "RTX 5090", 164.40, ""),
    "wan5b-base": ("Wan2.2 TI2V-5B", "704x1280x121, 50 st, CFG 5, base", "1x GB200", 70.25, "theirs: 5-prompt median"),
    "wan5b-easycache": ("Wan2.2 TI2V-5B", "same, EasyCache 0.036", "1x GB200", 24.35, "theirs: fusion + EasyCache fullopt"),
    "wan5b-opt": ("Wan2.2 TI2V-5B", "same, EasyCache 0.036 + PISA", "1x GB200", 28.69, "theirs: golden kernel+EasyCache+PISA run"),
    "wan14-720p-base-s15": ("Wan2.1 T2V-14B", "1280x720x81, 50 st, CFG 5, base (15 st measured, x50/15)", "-", None, "absolute number withdrawn upstream"),
    "wan14-720p-fullstack": ("Wan2.1 T2V-14B", "1280x720x81, 50 st, EasyCache + Sol-Attn", "-", None, "absolute number withdrawn upstream"),
    # phase B
    "sana-baseline": ("SANA-Video 2B", "832x480x81, 50 st, cfg 6, baseline", "1x GB200", None, "theirs: ratio only (2.77x)"),
    "sana-full": ("SANA-Video 2B", "same, EasyCache 0.1 + QKV merge + bf16 linear attn", "1x GB200", None, "theirs: ratio only (2.77x)"),
    "wan13-sol-base": ("Wan2.1 T2V-1.3B", "832x480x81, 50 st, CFG 6, base (median of 5 prompts)", "-", None, "no published number"),
    "wan13-sol-fullstack": ("Wan2.1 T2V-1.3B", "same, EasyCache 0.036 + Sol-Attn", "-", None, "no published number"),
    "a14b-sol-base": ("Wan2.2 T2V-A14B", "1280x720x81, 40 st, CFG 4/3, base (expert swap)", "1x GB200", 449.67, ""),
    "a14b-sol-fullopt": ("Wan2.2 T2V-A14B", "same, EasyCache + PISA", "1x GB200", 207.01, "theirs also: kernel fusion"),
    "ltx23-hq-base": ("LTX-2.3 HQ", "1920x1088x241, res2s 15 + 3, dense stage 2", "1x GB200", None, "theirs: ratio only (2.40x)"),
    "ltx23-hq-fullopt": ("LTX-2.3 HQ", "same, SCSP + PISA s2 + midpoint prune + NVFP4 FFN", "1x GB200", None, "theirs: ratio only (2.40x)"),
    "lingbot-baseline": ("LingBot-Video MoE", "480p 121 f 40 st + 1080p refiner 8 st, 1 prompt", "4x GB200", 375.53, "theirs: 4 GPUs (CP4)"),
    "lingbot-baseline-rs3": ("LingBot-Video MoE", "480p 121 f 40 st + 1080p refiner 8 st, 1 prompt (refiner: 3 of 8 st measured, x8/3)", "4x GB200", 375.53, "theirs: 4 GPUs (CP4)"),
    "lingbot-fullopt": ("LingBot-Video MoE", "same, EasyCache + refiner PISA", "4x GB200", 144.36, "theirs: 4 GPUs (CP4)"),
    "cosmos3-baseline": ("Cosmos3-Super 64B", "1280x720x189, 35 st, CFG 6", "4x GB200", 130.41, "theirs: 4 GPUs (SP)"),
    "cosmos3-teacache": ("Cosmos3-Super 64B", "same, TeaCache 1.15/10/3 (BF16)", "4x GB200", None, "theirs: 2.26x incl. NVFP4"),
    "cosmos3-teacache-fp8": ("Cosmos3-Super 64B", "same, TeaCache + W8A8 FP8", "4x GB200", None, "theirs: 2.26x incl. NVFP4"),
    "cosmos3-baseline-fp8": ("Cosmos3-Super 64B", "same, no cache, W8A8 FP8", "4x GB200", 130.41, "theirs: BF16 on 4 GPUs (SP)"),
}

# Speedup pairs (baseline cell, optimized cell, their ratio or None).
PAIRS = [
    ("h3-768p-dense-bf16", "h3-768p-fullopt-bf16", 1045.4 / 231.2),
    ("wan5b-base", "wan5b-easycache", 70.25 / 24.35),
    ("wan14-720p-base-s15", "wan14-720p-fullstack", None),
    ("sana-baseline", "sana-full", 2.77),
    ("wan13-sol-base", "wan13-sol-fullstack", None),
    ("a14b-sol-base", "a14b-sol-fullopt", 449.67 / 207.01),
    ("ltx23-hq-base", "ltx23-hq-fullopt", 2.40),
    ("lingbot-baseline", "lingbot-fullopt", 375.53 / 144.36),
    ("lingbot-baseline-rs3", "lingbot-fullopt", 375.53 / 144.36),
    ("cosmos3-baseline", "cosmos3-teacache", 2.26),
    ("cosmos3-baseline", "cosmos3-teacache-fp8", 2.26),
    ("cosmos3-baseline-fp8", "cosmos3-teacache-fp8", None),
]


def sol_timings(cell):
    """The `timings` check of a `fv-gpucheck sol` stage report (LingBot / Cosmos3)."""
    for f in sorted((cell / "gpucheck-out").glob("*.json")) if (cell / "gpucheck-out").is_dir() else []:
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        for c in j.get("checks") or []:
            if c.get("name") == "timings":
                return c.get("values") or {}
    return None


def load(run_dirs):
    rows = {}
    for d in run_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for c in sorted(p for p in d.iterdir() if p.is_dir()):
            b = c / "benchmark.json"
            if not b.exists() and (c / "frames" / "benchmark.json").exists():
                b = c / "frames" / "benchmark.json"
            s = c / "summary.json"
            row = {"cell": c.name, "run": d.name}
            if s.exists():
                row["summary"] = json.loads(s.read_text())
            if b.exists():
                j = json.loads(b.read_text())
                st = j.get("stage_seconds", {})
                row.update(
                    total_s=j.get("total_s") or j.get("e2e_seconds"),
                    load_s=j.get("load_s"),
                    text_s=j.get("text_s", st.get("text")),
                    denoise_s=j.get("denoise_s", (st.get("stage_1") or 0) + (st.get("stage_2") or 0) or None),
                    stage_1_s=st.get("stage_1"),
                    stage_2_s=st.get("stage_2"),
                    decode_s=j.get("decode_s", st.get("video_decode")),
                    warm=j.get("warm_steady_state"),
                    text_cache=j.get("text_cache"),
                    peak_mib=j.get("max_device_memory_used_mib") or j.get("peak_memory_mb"),
                    gpu=j.get("gpu") or j.get("hardware"),
                    env=j.get("env"),
                )
                if c.name.endswith("-s15") and row["denoise_s"]:
                    row["measured_total_s"] = row["total_s"]
                    row["total_s"] = (row["total_s"] - row["denoise_s"]) + row["denoise_s"] * 50 / 15
                    row["denoise_s"] = row["denoise_s"] * 50 / 15
                    row["extrapolated"] = "denoise x50/15"
            t = None if b.exists() else sol_timings(c)
            if t:
                if "request_s_median" in t:  # LingBot
                    r0 = (t.get("runs") or [{}])[0]
                    row.update(total_s=t["request_s_median"], text_s=r0.get("text_encode_s"),
                               denoise_s=(r0.get("base_denoise_s") or 0) + (r0.get("refiner_denoise_s") or 0),
                               decode_s=(r0.get("base_decode_s") or 0) + (r0.get("refiner_decode_s") or 0),
                               load_s=sum(r0.get(k) or 0 for k in ("load_text_s", "load_base_s", "load_refiner_s")),
                               peak_mib=t.get("peak_mib"), gpu="RTX PRO 6000", warm=False,
                               detail={k: r0.get(k) for k in ("base_denoise_s", "refiner_denoise_s", "refiner_prepare_s",
                                                              "base_steps_reused", "refiner_steps_computed", "refiner_sparse_steps")})
                    # Baseline measured on fewer refiner steps (every dense CFG
                    # step costs the same): scale the refiner denoise to the
                    # official 8 (docs/perf/sol-bench.md, phase B3).
                    rsteps = r0.get("refiner_steps") or 0
                    if c.name.endswith("-rs3") and rsteps and r0.get("refiner_denoise_s"):
                        rd = r0["refiner_denoise_s"]
                        row["measured_total_s"] = row["total_s"]
                        row["total_s"] = row["total_s"] - rd + rd * 8 / rsteps
                        row["denoise_s"] = row["denoise_s"] - rd + rd * 8 / rsteps
                        row["extrapolated"] = f"refiner denoise x8/{rsteps}"
                else:  # Cosmos3
                    row.update(total_s=t.get("request_s"), text_s=t.get("text_tower_s"), denoise_s=t.get("denoise_s"),
                               decode_s=t.get("decode_s"), load_s=t.get("load_s"), peak_mib=t.get("peak_mib"),
                               gpu="RTX PRO 6000", warm=True,
                               detail={k: t.get(k) for k in ("steps", "steps_computed", "steps_reused", "fp8")})
            rows[c.name] = row
    return rows


def fmt(x):
    return "—" if x is None else f"{x:.2f}"


def main():
    rows = load(sys.argv[1:])
    dirs = [Path(a) for a in sys.argv[1:] if Path(a).is_dir()]
    out = ["| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |",
           "|---|---|---|---:|---|---:|---:|---|---|"]
    for cell, (model, cfg, thw, theirs, note) in PUBLISHED.items():
        r = rows.get(cell)
        if r is None:
            continue
        ours = r.get("total_s")
        notes = [n for n in (note, r.get("extrapolated") and "extrapolated: " + r["extrapolated"]) if n]
        if ours is None:
            skipped = r.get("summary", {}).get("skipped") or f"exit {r.get('summary', {}).get('exit')}"
            notes.append(f"not run: {skipped}")
        if r.get("warm") is False:
            notes.append("first request after load")
        if r.get("text_cache") == "hit":
            notes.append("text: cache hit")
        ratio = f"{ours / theirs:.2f}x" if ours and theirs else "—"
        detail = " / ".join(fmt(r.get(k)) for k in ("load_s", "text_s", "denoise_s", "decode_s"))
        hw = "RTX PRO 6000 (sm_120)" if ours is not None else "—"
        out.append(f"| {model} | {cfg} | {thw} | {fmt(theirs)} | {hw} | {fmt(ours)} | {ratio} | {detail} | {'; '.join(notes)} |")
    # LTX-2.5: sol-engine's RTX 5090 README also lists Sol stage-2 seconds,
    # the DiT-only part of the E2E (no text, VAE or writer).
    stage2 = {"ltx25-4k5s-sol-bf16": 130.32, "ltx25-1080p20s-sol-bf16": 122.27,
              "ltx25-4k5s-sol-nvfp4": 72.82, "ltx25-1080p20s-sol-nvfp4": 66.78}
    lines = []
    for cell, theirs in stage2.items():
        r = rows.get(cell, {})
        if r.get("stage_2_s"):
            lines.append(f"| {cell} | {theirs:.2f} | {r['stage_2_s']:.2f} | {r['stage_2_s'] / theirs:.2f}x | {fmt(r.get('stage_1_s'))} |")
    if lines:
        out += ["", "LTX-2.5 Sol stage 2 only (RTX 5090 published vs ours on RTX PRO 6000):", "",
                "| Cell | Theirs stage 2 (s) | Ours stage 2 (s) | Ours / theirs | Ours stage 1 (s) |", "|---|---:|---:|---:|---:|"] + lines
    lines = []
    for base, opt, theirs in PAIRS:
        b = rows.get(base, {}).get("total_s")
        o = rows.get(opt, {}).get("total_s")
        if b and o:
            lines.append(f"| {base} → {opt} | {b / o:.2f}x | {f'{theirs:.2f}x' if theirs else '—'} |")
    if lines:
        out += ["", "Optimized-arm speedup over our own baseline (same GPU) vs theirs:", "",
                "| Pair | Ours | Theirs |", "|---|---:|---:|"] + lines
    print("\n".join(out))
    dirs[0].parent.joinpath("results.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
