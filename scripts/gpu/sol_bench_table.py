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
}


def load(run_dirs):
    rows = {}
    for d in run_dirs:
        d = Path(d)
        hw = ""
        for c in sorted(p for p in d.iterdir() if p.is_dir()):
            b = c / "benchmark.json"
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
            rows[c.name] = row
    return rows


def fmt(x):
    return "—" if x is None else f"{x:.2f}"


def main():
    rows = load(sys.argv[1:])
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
        hw = "RTX PRO 6000 (sm_120)" if r.get("gpu") else "—"
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
    b = rows.get("wan14-720p-base-s15", {}).get("total_s")
    f = rows.get("wan14-720p-fullstack", {}).get("total_s")
    if b and f:
        out.append(f"\nWan2.1 T2V-14B 720p, our fullstack speedup over base: **{b / f:.2f}x** (sol-engine's withdrawn README headline: ~3.48x).")
    b = rows.get("wan5b-base", {}).get("total_s")
    for arm in ("wan5b-easycache", "wan5b-opt"):
        f = rows.get(arm, {}).get("total_s")
        if b and f:
            out.append(f"\nWan2.2 TI2V-5B {arm} speedup over our base: **{b / f:.2f}x** (theirs 2.885x fullopt, 2.45x golden).")
    print("\n".join(out))
    Path(sys.argv[1]).parent.joinpath("results.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
