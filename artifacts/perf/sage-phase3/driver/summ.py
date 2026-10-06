#!/usr/bin/env python3
"""Compact Phase 3 rows: per cell medians, per compare prompt metrics, gate verdicts.
    python3 summ.py <cell dir>... | gates | compares <glob>"""
import glob
import json
import sys


def cell(d):
    try:
        b = json.load(open(f"{d}/benchmark.json"))
    except Exception as e:  # noqa: BLE001
        r = {}
        try:
            r = json.load(open(f"{d}/run.json"))
        except Exception:  # noqa: BLE001
            pass
        print(f"CELL {d.split('/')[-1]} no benchmark.json ({e.__class__.__name__}) run={r}")
        return
    pp = b.get("aggregate", {}).get("per_prompt", [])
    st = {}
    for p in b.get("prompts", {}).values():
        for k, v in (p.get("stage_seconds") or {}).items():
            st.setdefault(k, []).append(v)
    med = {k: sorted(v)[len(v) // 2] for k, v in st.items()}
    run = {}
    try:
        run = json.load(open(f"{d}/run.json"))
    except Exception:  # noqa: BLE001
        pass
    att = b.get("attention", {})
    print(f"CELL {d.split('/')[-1]} denoise={b.get('denoise_s', 0):.2f} decode={b.get('decode_s', 0):.2f} "
          f"total={b.get('total_s', b.get('e2e_seconds', 0)):.2f} load={b.get('load_s', 0):.0f} "
          f"peakMiB={run.get('peak_mib')} wall={run.get('wall_s')} n={len(pp)} "
          f"att(dense={att.get('dense_video_calls')},sol={att.get('sol_calls')},vsa={att.get('vsa_calls')}) "
          f"stages={ {k: round(v, 2) for k, v in med.items() if v and v > 0.05} } "
          f"per_prompt={[(p['name'], round(p.get('denoise_s', 0), 2), round(p.get('total_s', 0), 2)) for p in pp]}")


def compares(pat):
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        c = {x["name"]: x["values"] for x in d.get("checks", [])}
        pm, lp = c.get("pixel_metrics", {}), c.get("lpips", {})
        nan = float("nan")
        pm = {k: (nan if v is None else v) for k, v in pm.items()}
        lp = {k: (nan if v is None else v) for k, v in lp.items()}
        print(f"CMP {f.split('compare-clips-')[-1][:-5]} lpips={lp.get('mean', float('nan')):.3f}/{lp.get('max', float('nan')):.3f} "
              f"psnr={pm.get('psnr_mean', float('nan')):.2f} sharp={pm.get('sharpness_ratio_mean', float('nan')):.3f} "
              f"jitter={pm.get('temporal_jitter_ratio_mean', float('nan')):.3f} frames={c.get('frame_counts', {}).get('pairs')} "
              f"shape={c.get('frame_shape', {}).get('shape_mismatch')}")


def gates():
    for f in sorted(glob.glob("/root/p3/gate/*.json")):
        d = json.load(open(f))
        fails = [x["name"].replace("quality/compare-clips-", "") for x in d.get("checks", []) if not x.get("ok")]
        sp = next((x["values"] for x in d.get("checks", []) if x["name"] == "performance/speedup"), {})
        print(f"GATE {f.split('gate-')[-1][:-5]} verdict={d.get('verdict')} denoise {sp.get('baseline')}->{sp.get('candidate')} "
              f"x{sp.get('speedup', 0):.3f} total x{sp.get('secondary_speedup', 0):.3f} fails={fails}")


if sys.argv[1] == "gates":
    gates()
elif sys.argv[1] == "compares":
    compares(sys.argv[2])
else:
    for d in sys.argv[1:]:
        cell(d)
