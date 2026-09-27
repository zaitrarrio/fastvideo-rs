#!/usr/bin/env python3
"""FastVideo's VSA (fastvideo_kernel.video_sparse_attn), stage by stage, at
the grids fv-gpucheck's ``vsa_stages`` group times ours at.

Runs in the FastVideo venv (the pinned FastVideo checkout + the PyPI
fastvideo-kernel wheel; on sm_120 the sparse branch is the Triton kernel).

Workloads (tile (4, 4, 4), d=128, bf16, one batch):
  fasth3_768p  56 heads, token grid (37, 24, 42), sparsity 0.8 and 0.9
  fasth3_480p  56 heads, token grid (37, 15, 26), sparsity 0.8 and 0.9
  fastwan13    12 heads, token grid (21, 30, 52), sparsity 0.8 (basic_dmd.py)

Stages, each the median of 5 CUDA-event-timed calls after a warm-up, the
way ``video_sparse_attn`` (fastvideo_kernel/ops.py) runs them:
  tile      raster -> tile order with zero padding, q/k/v/gate (the model's
            attention backend does this around the op)
  coarse    fused_block_mean(q, k, v) + scores + softmax + P V + repeat
  topk      fused_topk_mask(scores, topk)
  fine      block_sparse_attn(q, k, v, mask, vbs) (mask -> index + Triton)
  combine   out_c * gate + out_s
  untile    tile order -> raster
  total     video_sparse_attn(...) end to end (tile/untile excluded)
Q/K/V/gate are N(0, 1); the top-k selection is data-dependent only in which
tiles it picks, not in how many.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback

WORKLOADS = [
    ("fasth3_768p", 56, (37, 24, 42), [0.8, 0.9]),
    ("fasth3_480p", 56, (37, 15, 26), [0.8, 0.9]),
    ("fastwan13", 12, (21, 30, 52), [0.8]),
]
D = 128
TILE = (4, 4, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workloads", default="")
    a = ap.parse_args()

    import torch
    import fastvideo_kernel as fk

    res: dict = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "fastvideo_kernel": getattr(fk, "__version__", "?"),
        "workloads": {},
    }
    try:
        from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean, fused_topk_mask
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn
        res["stages"] = "fused_block_mean / fused_topk_mask / block_sparse_attn"
    except Exception as exc:  # noqa: BLE001
        res["stage_import_error"] = f"{type(exc).__name__}: {exc}"
        fused_block_mean = fused_topk_mask = block_sparse_attn = None
    from fastvideo_kernel import video_sparse_attn
    from fastvideo_kernel.vsa_utils import build_vsa_metadata

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[2]

    g = torch.Generator(device="cuda").manual_seed(1234)
    only = {s for s in a.workloads.split(",") if s}
    for name, heads, grid, sparsities in WORKLOADS:
        if only and name not in only:
            continue
        meta = build_vsa_metadata(grid, TILE, "cuda")
        tidx, ridx = meta["tile_partition_indices"], meta["reverse_tile_partition_indices"]
        vbs, npi = meta["variable_block_sizes"], meta["non_pad_index"]
        nb = vbs.numel()
        seq = grid[0] * grid[1] * grid[2]
        padded = nb * 64
        out: dict = {"heads": heads, "grid": list(grid), "tokens": seq, "tiles": nb, "padded": padded}
        res["workloads"][name] = out
        x = [torch.randn(1, heads, seq, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(4)]

        def tile_one(t):
            p = torch.zeros(1, heads, padded, D, device="cuda", dtype=t.dtype)
            p[:, :, npi] = t[:, :, tidx]
            return p

        try:
            out["tile_ms"] = timed(lambda: [tile_one(t) for t in x])
            q, k, v, gate = (tile_one(t) for t in x)
            del x
            out["untile_ms"] = timed(lambda: q[:, :, npi][:, :, ridx].contiguous())
            for sp in sparsities:
                topk = max(1, min(nb, math.ceil((1.0 - sp) * nb)))
                key = f"sparsity{sp}"
                r: dict = {"topk": topk}
                out[key] = r
                try:
                    def total():
                        try:
                            return video_sparse_attn(q, k, v, vbs, vbs, topk, TILE, gate)
                        except TypeError:  # wheels before q_variable_block_sizes
                            return video_sparse_attn(q, k, v, variable_block_sizes=vbs, topk=topk,
                                                     block_size=TILE, compress_attn_weight=gate)
                    r["total_ms"] = timed(total)
                    if fused_block_mean is not None:
                        def coarse():
                            qc = fused_block_mean(q, vbs, 64)
                            kc = fused_block_mean(k, vbs, 64)
                            vc = fused_block_mean(v, vbs, 64)
                            scores = torch.matmul(qc, kc.transpose(-2, -1)) / (D ** 0.5)
                            attn = torch.softmax(scores, dim=-1)
                            oc = torch.matmul(attn, vc).view(1, heads, nb, 1, D)
                            oc = oc.repeat(1, 1, 1, 64, 1).view(1, heads, padded, D)
                            return scores, oc
                        r["coarse_ms"] = timed(coarse)
                        scores, out_c = coarse()
                        r["topk_ms"] = timed(lambda: fused_topk_mask(scores, topk))
                        mask = fused_topk_mask(scores, topk)
                        r["fine_ms"] = timed(lambda: block_sparse_attn(q, k, v, mask, vbs))
                        out_s = block_sparse_attn(q, k, v, mask, vbs)[0]
                        r["combine_ms"] = timed(lambda: out_c * gate + out_s)
                        # One head's fine-stage attended key tiles x 64 x 64 x 4 D FLOPs.
                        flops = 4.0 * heads * nb * 64 * topk * 64 * D
                        r["fine_tflops"] = flops / (r["fine_ms"] * 1e-3) / 1e12
                        del scores, out_c, mask, out_s
                except Exception as exc:  # noqa: BLE001
                    r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    r["trace"] = traceback.format_exc()[-800:]
                torch.cuda.empty_cache()
            del q, k, v, gate
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            out["trace"] = traceback.format_exc()[-800:]
        torch.cuda.empty_cache()
        print(json.dumps({name: out}), flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
