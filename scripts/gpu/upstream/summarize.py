#!/usr/bin/env python3
"""Tabulate upstream cells pulled into artifacts/runpod/upstream/<tag>/<cell>/.

Reads each cell's cell.json (exit, wall seconds, nvidia-smi peak) and the
driver's result.json / benchmark.json, and prints one markdown row per cell.
The latest successful run of a cell name wins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def f(x, nd=1):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def sglang_log(c: Path, started) -> dict:
    """Stage times of the measured request, load time and TeaCache decisions from SGLang's log."""
    import datetime
    import re
    out: dict = {}
    try:
        text = (c / "stdout.log").read_text(errors="replace")
    except OSError:
        return out
    stages: dict = {}
    for m in re.finditer(r"\[([A-Za-z0-9]+Stage)\] finished in ([0-9.]+) seconds", text):
        stages[m.group(1).replace("RTX5090MiniMaxH3", "").replace("MiniMaxH3", "")] = float(m.group(2))
    out["stage_times"] = stages  # last occurrence = measured request
    m = re.search(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] Pipeline instantiated", text)
    if m and started:
        t = datetime.datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
        out["load"] = t.timestamp() - float(started)
    comp = len(re.findall(r'"action": "compute"[^\n]*"request_epoch": "measured', text))
    reuse = len(re.findall(r'"action": "reuse"[^\n]*"request_epoch": "measured', text))
    if comp or reuse:
        out["teacache"] = (comp, reuse)
    return out


def row(tag: str, c: Path) -> dict | None:
    cell = load(c / "cell.json")
    if not cell:
        return None
    r = load(c / "result.json") or {}
    b = load(c / "benchmark.json") or {}
    name = c.name
    d = {"cell": name, "tag": tag, "exit": cell.get("exit"), "wall": cell.get("seconds"),
         "peak_smi_gib": (cell.get("peak_smi_mib") or 0) / 1024 or None, "load": None, "warm": None,
         "stages": "", "notes": ""}
    if name.startswith("fv-"):
        d["load"] = r.get("load_s")
        d["warm"] = r.get("warm_total_s")
        runs = r.get("runs") or []
        if runs:
            st = runs[-1].get("stages_s") or {}
            keep = {k: v for k, v in st.items() if v and v > 0.05}
            d["stages"] = ", ".join(f"{k.replace('Stage', '')} {v:.1f}" for k, v in keep.items())
            d["notes"] = f"median of {len(runs)} after 1 warmup"
        if r.get("error"):
            d["notes"] = r["error"][:160]
    elif name.startswith("sol-h3r5090"):
        m = b.get("measured") or {}
        w = b.get("warmup") or {}
        d["warm"] = m.get("inference_time_s")
        d["stages"] = f"warmup {f(w.get('inference_time_s'))}"
        if m.get("peak_memory_mb"):
            d["notes"] = f"sglang peak {m['peak_memory_mb'] / 1024:.1f} GiB"
        d.update(sglang_log(c, cell.get("started")))
        if d.get("stage_times"):
            d["stages"] += "; " + ", ".join(f"{k} {v:.1f}" for k, v in d["stage_times"].items())
        if d.get("teacache"):
            d["notes"] += f"; measured DiT forwards computed/reused {d['teacache'][0]}/{d['teacache'][1]}"
    elif name.startswith("sol-h3-4step"):
        d["load"] = r.get("load_s")
        d["warm"] = r.get("warm_total_s")
        d["notes"] = f"peak alloc {f(r.get('peak_allocated_gib'))} GiB; TE offload={r.get('te_offload')}"
        if r.get("error"):
            d["notes"] = r["error"][:160]
    elif name.startswith("sol-ltx25"):
        m = b or (r.get("reference_metrics") or {})
        d["load"] = r.get("build_s")
        d["warm"] = m.get("e2e_seconds")
        d["stages"] = (f"stage1 {f(m.get('stage_1_seconds'))}, stage2 {f(m.get('stage_2_seconds'))}, "
                       f"vae {f(m.get('video_vae_seconds'))}")
        att = (m.get("attention") or {})
        d["notes"] = (f"peak alloc {f(m.get('peak_allocated_gib'))} GiB; sol calls {att.get('sol_calls')}"
                      f" dense video calls {att.get('dense_video_calls')}")
        if r.get("error"):
            d["notes"] = r["error"][:160]
    return d


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/runpod/upstream")
    best: dict[str, dict] = {}
    for tagdir in sorted(root.iterdir()):
        if not tagdir.is_dir():
            continue
        for c in sorted(tagdir.iterdir()):
            if c.is_dir():
                d = row(tagdir.name, c)
                if d and (d["exit"] == 0 or c.name not in best or best[c.name]["exit"] != 0):
                    best[c.name] = d
    print("| cell | run | exit | load s | warm total s | stages s | peak GiB (smi) | process wall s | notes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(best):
        d = best[k]
        print(f"| {d['cell']} | {d['tag']} | {d['exit']} | {f(d['load'])} | {f(d['warm'], 2)} | {d['stages']} | "
              f"{f(d['peak_smi_gib'])} | {d['wall']} | {d['notes']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
