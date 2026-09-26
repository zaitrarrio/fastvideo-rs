#!/usr/bin/env python3
"""Attention microbenchmarks of the upstream references at our real shapes.

Runs in the sol-ltx25 venv (torch + cuDNN + sol-engine Sol-Attn):

* torch SDPA per backend (cuDNN fused attention, FlashAttention, memory
  efficient), bf16 BHSD in and out: the dense references;
* sol-engine ``sol_attn`` (the CuTe sm120 kernel on RTX PRO 6000, kv_splits=1
  as models/ltx25/RTX5090/attention.py:94-102 calls it), bf16 BTHD, total and
  ``prepare`` alone, plus the exact-block fraction of one head.

Shapes and data match fv-gpucheck's ``attn_bench`` group: H3 768p (56 heads,
37 710 tokens), LTX-2.5 stage 2 at 768x512 (6 144), 1080p 20 s (124 440) and
4K 5 s (130 560), 32 heads, d=128. Q/K are "structured" (per-64-block bases +
noise, one head repeated over all heads), V is N(0, 1). Each time is the
median of three CUDA-event-timed calls after a warm-up.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

SHAPES = [
    ("h3_768p", 56, 37_710, [1.0]),
    ("ltx_512p", 32, 6_144, [1.0, 1.25, 1.5]),
    ("ltx_1080p20s", 32, 124_440, [1.0, 1.25, 1.5]),
    ("ltx_4k5s", 32, 130_560, [1.0, 1.25, 1.5]),
]
D = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol-engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="")
    a = ap.parse_args()
    sys.path.insert(0, f"{a.sol_engine}/techniques/sparse_backends")
    sys.path.insert(0, a.sol_engine)

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    res: dict = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "shapes": {},
    }
    try:
        from sol_attn import get_sol_attn_backend, sol_attn
        from sol_attn.preprocess import prepare

        res["sol_backend"] = get_sol_attn_backend(0)
    except Exception as exc:  # noqa: BLE001
        res["sol_import_error"] = f"{type(exc).__name__}: {exc}"
        sol_attn = prepare = None

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[1]

    g = torch.Generator(device="cuda").manual_seed(1234)

    def structured(tokens: int) -> torch.Tensor:
        n = (tokens + 63) // 64
        base = torch.randn(n, D, device="cuda", generator=g) * 1.5
        common = torch.randn(n, D, device="cuda", generator=g) * 1.5
        blk = torch.arange(tokens, device="cuda") // 64
        src = torch.where(((blk % 4) < 2)[:, None], common[blk], base[blk])
        return src + torch.randn(tokens, D, device="cuda", generator=g) * 0.3

    only = {s for s in a.shapes.split(",") if s}
    for name, heads, tokens, taus in SHAPES:
        if only and name not in only:
            continue
        out: dict = {"heads": heads, "tokens": tokens}
        res["shapes"][name] = out
        # BTHD bf16, one head repeated (as ours): [1, T, H, 128]
        q = structured(tokens).to(torch.bfloat16)[None, :, None, :].expand(1, tokens, heads, D).contiguous()
        k = structured(tokens).to(torch.bfloat16)[None, :, None, :].expand(1, tokens, heads, D).contiguous()
        v = torch.randn(tokens, D, device="cuda", generator=g).to(torch.bfloat16)[None, :, None, :] \
            .expand(1, tokens, heads, D).contiguous()
        flops = 4.0 * heads * tokens * tokens * D
        qh, kh, vh = (x.transpose(1, 2).contiguous() for x in (q, k, v))  # BHSD
        for be_name, be in [("cudnn", SDPBackend.CUDNN_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION),
                            ("efficient", SDPBackend.EFFICIENT_ATTENTION)]:
            try:
                with sdpa_kernel([be]):
                    ms = timed(lambda: torch.nn.functional.scaled_dot_product_attention(qh, kh, vh))
                out[f"sdpa_{be_name}_ms"] = ms
                out[f"sdpa_{be_name}_tflops"] = flops / (ms * 1e-3) / 1e12
            except Exception as exc:  # noqa: BLE001
                out[f"sdpa_{be_name}_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        del qh, kh, vh
        torch.cuda.empty_cache()
        if sol_attn is None:
            continue
        scale = D ** -0.5
        for tau in taus:
            key = f"sol_tau{tau}"
            try:
                ms = timed(lambda: sol_attn(q, k, v, tau=tau, thresh_type="diag", kv_splits=1))
                pms = timed(lambda: prepare(q, k, v, tau=tau, scale=scale, thresh_type="diag"))
                # Exact fraction of head 0: column mean of S over live rows is
                # qbar . kc (linear), in log2-score units vs the threshold.
                kc, _vc, thr = prepare(q, k, v, tau=tau, scale=scale, thresh_type="diag")
                nt = (tokens + 63) // 64
                qf = q[0, :, 0, :].float()
                pad = nt * 64 - tokens
                lens = torch.full((nt,), 64.0, device="cuda")
                lens[-1] = 64 - pad
                qbar = torch.nn.functional.pad(qf, (0, 0, 0, pad)).view(nt, 64, D).sum(1) / lens[:, None]
                cm = (qbar @ kc[0, :, 0, :].float().T) * (scale * 1.4426950408889634)
                th = thr[0, :, 0].float()
                i = torch.arange(nt, device="cuda")
                exact = (cm > th[:, None]) | ((i[:, None] - i[None, :]).abs() <= 1)
                out[key] = {"ms": ms, "prepare_ms": pms,
                            "exact_fraction_head0_approx": exact.float().mean().item()}
            except Exception as exc:  # noqa: BLE001
                out[key] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}",
                            "trace": traceback.format_exc()[-800:]}
        del q, k, v
        torch.cuda.empty_cache()
        print(json.dumps({name: out}), flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
