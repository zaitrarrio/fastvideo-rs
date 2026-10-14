#!/usr/bin/env python3
"""Merged-weight parity of a Turbo LoRA fold: ours against the MiniMax-H3-Turbo-Lora Space's (docs/ports/h3-turbo-loras.md).

    merged_parity.py REF_DIR OURS_DIR [--base BASE_DIR] [--json OUT]

REF_DIR holds the Space's `merged.<prefix>` (h3_turbo_lora_ref.py), OURS_DIR ours (`FV_H3_DUMP_MERGED` with
`FASTVIDEO_DUMP_DIR`), BASE_DIR the unmodified `base.<prefix>` (default REF_DIR). Per linear: rel-L2 of the merged
weights, rel-L2 of the deltas (merged - base, the LoRA's own contribution, which the merged rel-L2 dilutes by the
base's norm), the fraction of elements that differ (both sides are bf16, so a difference is at least one bf16 ulp) and
the largest difference in ulps; then the worst per target type (q/k/v, out, ff.net.0 with the SwiGLU swap, ff.net.2,
block AdaLN, norm_out, refiner). numpy only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load(prefix: Path):
    shape = [int(x) for x in Path(f"{prefix}.shape").read_text().split()]
    return np.fromfile(f"{prefix}.f32", dtype="<f4").reshape(shape)


def kind(p: str) -> str:
    if p.startswith("token_refiner."):
        return "refiner " + kind(p.split(".", 3)[3])
    for pat, k in ((r"attn\.to_[qkv]$", "attn.to_q/k/v (QKV thirds)"), (r"attn\.to_out\.0$", "attn.to_out.0 (out_proj)"),
                   (r"ff\.net\.0\.proj$", "ff.net.0.proj (fc1, SwiGLU swap)"), (r"ff\.net\.2$", "ff.net.2 (fc2)"),
                   (r"adaln_proj\.linear$", "block adaln_proj.linear"), (r"^norm_out\.linear$", "norm_out.linear (final AdaLN)")):
        if re.search(pat, p):
            return k
    return p


def rel(a, b) -> float:
    d = np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel())
    n = np.linalg.norm(b.astype(np.float64).ravel())
    return float(d / n) if n else float(d)


def ulps(a, b):
    """Distance in bf16 ulps (both are bf16 values widened to f32)."""
    ai = (a.view(np.uint32) >> 16).astype(np.int64)
    bi = (b.view(np.uint32) >> 16).astype(np.int64)
    # sign-magnitude to a monotonic integer line
    ai = np.where(ai & 0x8000, -(ai & 0x7FFF), ai)
    bi = np.where(bi & 0x8000, -(bi & 0x7FFF), bi)
    return np.abs(ai - bi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("ours")
    ap.add_argument("--base", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ref, ours, base = Path(a.ref), Path(a.ours), Path(a.base or a.ref)
    rows = []
    for f in sorted(ref.glob("merged.*.f32")):
        p = f.name[len("merged."):-len(".f32")]
        o = ours / f"merged.{p}"
        if not Path(f"{o}.f32").is_file():
            rows.append({"param": p, "kind": kind(p), "missing": "ours"})
            continue
        r, u = load(ref / f"merged.{p}"), load(o)
        row = {"param": p, "kind": kind(p), "shape": list(r.shape), "merged_rel_l2": rel(u, r)}
        bp = base / f"base.{p}"
        if Path(f"{bp}.f32").is_file():
            b = load(bp)
            row["delta_rel_l2"] = rel(u - b, r - b)
            row["delta_norm_vs_base"] = float(np.linalg.norm((r - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))
        d = ulps(u.astype(np.float32), r.astype(np.float32))
        row["frac_differ"] = float((d > 0).mean())
        row["max_ulps"] = int(d.max())
        rows.append(row)
    by_kind: dict[str, dict] = {}
    for r in rows:
        if "missing" in r:
            continue
        k = by_kind.setdefault(r["kind"], {"n": 0, "merged_rel_l2": 0.0, "delta_rel_l2": 0.0, "frac_differ": 0.0, "max_ulps": 0})
        k["n"] += 1
        for m in ("merged_rel_l2", "delta_rel_l2", "frac_differ"):
            k[m] = max(k[m], r.get(m, 0.0))
        k["max_ulps"] = max(k["max_ulps"], r["max_ulps"])
    for r in rows:
        print(json.dumps(r))
    print("worst per target type:")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:40s} n={v['n']} merged rel-L2 {v['merged_rel_l2']:.2e}  delta rel-L2 {v['delta_rel_l2']:.2e}  "
              f"differ {100 * v['frac_differ']:.3f}%  max {v['max_ulps']} ulp")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "by_kind": by_kind}, indent=1))
    return 1 if any("missing" in r for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
