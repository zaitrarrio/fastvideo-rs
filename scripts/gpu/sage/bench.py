#!/usr/bin/env python3
"""SageAttention2 / SageAttention3 vs our attention kernels at our real shapes.

Kernels (each is timed as the whole call a model would make, including any
quantization / smoothing / padding it does):

  fv_fwd2     our bf16 `flash_mma_fwd2_d128` (kernels.cu, nvcc cubin via libcuda)
  fv_fp8      our FP8-QK attempt `attn_fp8_fwd_d128` (+ its colsum/quant kernels)
  sdpa_flash  torch SDPA, FlashAttention-2 backend
  sdpa_cudnn  torch SDPA, cuDNN backend (our sm_12x default picks cuDNN or fwd2 per shape)
  sage2pp     sageattn() on sm_120: INT8 QK per-warp, FP8 PV, fp32+fp16 accumulation (SageAttention2++)
  sage2_thr   INT8 QK per-thread, FP8 PV, fp32+fp32 accumulation (SageAttention2, most accurate FP8-PV route)
  sage2_f16pv INT8 QK per-thread, FP16 PV, fp32 accumulation
  sage3       sageattn3_blackwell: NVFP4 microscaled QK and PV (sm_120a)

Accuracy: against an FP32 reference computed from the same bf16-rounded
inputs (TF32 off), and against fv_fwd2 (the bf16 kernel our tolerance was
stated against: rel-L2 <= 5e-2, cosine >= 0.998).

Usage: bench.py --shapes h3_768p,... --modes normal,peaked --out results.jsonl
       bench.py --real /root/w/cap/fastwan_s0_l15.pt --out results.jsonl
"""
import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# name: (heads, seq)   d = 128, Sq = Sk (self-attention incl. any joint text prefix)
SHAPES = {
    "h3_480p": (56, 15100),        # 256 text + 414 audio + 37 x 390 video (480x832, 5 s)
    "h3_768p": (56, 37966),        # 256 + 414 + 37 x 1008 (768x1344, 5 s)
    "ltx_480p_s1": (32, 1792),     # 896x512 canvas, stage 1 at 448x256, 16 latent frames
    "ltx_480p_s2": (32, 7168),     # stage 2 at 896x512
    "ltx_384p_1stage": (32, 4032),  # stage-1-only 672x384
    "ltx_720p_s1": (32, 3520),     # 1280x704 canvas, stage 1 at 640x352
    "ltx_720p_s2": (32, 14080),    # stage 2 at 1280x704
    "ltx_1080p_s2": (32, 32640),   # 1920x1088 stage 2 (reference point)
    "wan5b_480p": (24, 12090),     # Wan2.2 TI2V-5B 832x480, 121 frames (31 x 15 x 26)
    "fastwan_480p": (12, 32760),   # Wan2.1 1.3B 832x480, 81 frames (21 x 30 x 52)
}
D = 128
LOG2E = 1.4426950408889634


# ---------------------------------------------------------------- libcuda
class Cu:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        L = self.lib
        L.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        L.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        L.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]

    @staticmethod
    def ck(r, what):
        if r != 0:
            raise RuntimeError(f"{what}: CUresult {r}")

    def load(self, path):
        m = ctypes.c_void_p()
        self.ck(self.lib.cuModuleLoad(ctypes.byref(m), path.encode()), f"cuModuleLoad {path}")
        return m

    def func(self, mod, name, smem=0):
        f = ctypes.c_void_p()
        self.ck(self.lib.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), name)
        if smem:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
            self.ck(self.lib.cuFuncSetAttribute(f, 8, smem), f"smem {name}")
        return f

    def launch(self, f, grid, block, smem, args):
        arr = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        st = torch.cuda.current_stream().cuda_stream
        self.ck(self.lib.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                                        smem, ctypes.c_void_p(st), arr, None), "cuLaunchKernel")


def P(t):
    return ctypes.c_uint64(t.data_ptr())


class Ours:
    FWD2_SMEM = 4 * 64 * D * 2
    FP8_SMEM = 2 * (64 * 128 + 64 * 128 * 2)

    def __init__(self, cubin_dir):
        torch.zeros(1, device="cuda")  # make torch's primary context current for the driver API
        self.cu = Cu()
        m = self.cu.load(os.path.join(cubin_dir, "kernels.cubin"))
        self.fwd2 = self.cu.func(m, "flash_mma_fwd2_d128", self.FWD2_SMEM)
        self.fp8 = None
        p8 = os.path.join(cubin_dir, "attn_fp8.cubin")
        if os.path.exists(p8):
            m8 = self.cu.load(p8)
            self.colsum = self.cu.func(m8, "attn_fp8_colsum_bf16")
            self.quant = self.cu.func(m8, "attn_fp8_quant_bf16")
            self.fp8 = self.cu.func(m8, "attn_fp8_fwd_d128", self.FP8_SMEM)
        self.dummy = torch.empty(4, device="cuda", dtype=torch.float32)

    def dense(self, q, k, v):
        b, h, sq, d = q.shape
        sk = k.shape[2]
        out = torch.empty_like(q)
        sl2 = (1.0 / math.sqrt(d)) * LOG2E
        self.cu.launch(self.fwd2, ((sq + 127) // 128, b * h, 1), (256, 1, 1), self.FWD2_SMEM,
                       [P(q), P(k), P(v), P(self.dummy), P(out), ctypes.c_int(1),
                        ctypes.c_int(sq), ctypes.c_int(sk), ctypes.c_float(sl2)])
        return out

    def _quant(self, x, bh, rows, rows_pad, grp, smooth):
        s = torch.zeros(bh * 128, device="cuda", dtype=torch.float32)
        if smooth:
            self.cu.launch(self.colsum, ((rows + 255) // 256, bh, 1), (128, 1, 1), 0,
                           [P(x), P(s), ctypes.c_int(rows), ctypes.c_int(256)])
        out = torch.empty(bh * rows_pad * 128, device="cuda", dtype=torch.uint8)
        sc = torch.empty(bh * rows_pad // grp, device="cuda", dtype=torch.float32)
        self.cu.launch(self.quant, (rows_pad // 64, bh, 1), (128, 1, 1), 0,
                       [P(x), P(s), ctypes.c_float(1.0 / max(rows, 1)), ctypes.c_int(int(smooth)),
                        P(out), P(sc), ctypes.c_int(rows), ctypes.c_int(rows_pad), ctypes.c_int(grp)])
        return out, sc

    def dense_fp8(self, q, k, v):
        b, h, sq, d = q.shape
        sk = k.shape[2]
        bh = b * h
        sqp, skp = (sq + 127) // 128 * 128, (sk + 63) // 64 * 64
        q8, qs = self._quant(q, bh, sq, sqp, 16, False)
        k8, ks = self._quant(k, bh, sk, skp, 64, True)
        out = torch.empty_like(q)
        sl2 = (1.0 / math.sqrt(d)) * LOG2E
        self.cu.launch(self.fp8, (sqp // 128, bh, 1), (256, 1, 1), self.FP8_SMEM,
                       [P(q8), P(qs), P(k8), P(ks), P(v), P(self.dummy), P(out), ctypes.c_int(1),
                        ctypes.c_int(sq), ctypes.c_int(sk), ctypes.c_int(sqp), ctypes.c_int(skp),
                        ctypes.c_float(sl2)])
        return out


# ---------------------------------------------------------------- kernels
def build_kernels(cubin_dir, only=None):
    ks = {}
    try:
        ours = Ours(cubin_dir)
        ks["fv_fwd2"] = ours.dense
        if ours.fp8 is not None:
            ks["fv_fp8"] = ours.dense_fp8
    except Exception as e:  # noqa: BLE001
        print(f"[warn] our kernels unavailable: {e}", file=sys.stderr)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def sdpa_with(backend):
        def f(q, k, v):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v)
        return f
    ks["sdpa_flash"] = sdpa_with(SDPBackend.FLASH_ATTENTION)
    ks["sdpa_cudnn"] = sdpa_with(SDPBackend.CUDNN_ATTENTION)
    try:
        import sageattention as sa
        ks["sage2pp"] = lambda q, k, v: sa.sageattn(q, k, v, tensor_layout="HND")
        ks["sage2_thr"] = lambda q, k, v: sa.sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, tensor_layout="HND", qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp32")
        ks["sage2_f16pv"] = lambda q, k, v: sa.sageattn_qk_int8_pv_fp16_cuda(
            q, k, v, tensor_layout="HND", qk_quant_gran="per_thread", pv_accum_dtype="fp32")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] sageattention unavailable: {e}", file=sys.stderr)
    try:
        from sageattn3 import sageattn3_blackwell
        # its preprocess subtracts k's mean IN PLACE: give it a copy
        ks["sage3"] = lambda q, k, v: sageattn3_blackwell(q, k.clone(), v)
        ks["sage3_nomean"] = lambda q, k, v: sageattn3_blackwell(q, k.clone(), v, per_block_mean=False)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] sageattn3 unavailable: {e}", file=sys.stderr)
    if only:
        ks = {n: f for n, f in ks.items() if n in only}
    return ks


# ---------------------------------------------------------------- inputs
def make_inputs(h, s, mode, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shp = (1, h, s, D)
    q = torch.randn(shp, device="cuda", generator=g)
    k = torch.randn(shp, device="cuda", generator=g)
    v = torch.randn(shp, device="cuda", generator=g)
    if mode == "peaked":
        # the fv-gpucheck attn_fp8 "peaked" regime: q x 3, per-(head, channel) K offset (std 2)
        q *= 3.0
        k += 2.0 * torch.randn((1, h, 1, D), device="cuda", generator=g)
    elif mode == "outlier":
        # a few heavy channels shared by q and k (token-varying, so smoothing cannot remove them),
        # plus the K offset: score std ~3 concentrated in 4 of 128 channels
        ch = torch.randperm(D, device="cuda", generator=g)[:4]
        q[..., ch] *= 4.0
        k[..., ch] *= 4.0
        k += 2.0 * torch.randn((1, h, 1, D), device="cuda", generator=g)
    elif mode != "normal":
        raise ValueError(mode)
    return q.bfloat16().contiguous(), k.bfloat16().contiguous(), v.bfloat16().contiguous()


# ---------------------------------------------------------------- reference / metrics
@torch.no_grad()
def reference(q, k, v, chunk=4096):
    b, h, s, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.empty(q.shape, device="cuda", dtype=torch.float32)
    for hi in range(h):
        kf = k[0, hi].float()
        vf = v[0, hi].float()
        for i in range(0, s, chunk):
            sc = (q[0, hi, i:i + chunk].float() @ kf.T) * scale
            out[0, hi, i:i + chunk] = torch.softmax(sc, dim=-1) @ vf
            del sc
    return out


@torch.no_grad()
def metrics(got, want):
    d2 = w2 = g2 = gw = 0.0
    mx = 0.0
    for hi in range(got.shape[1]):
        g = got[0, hi].double()
        w = want[0, hi].double()
        diff = g - w
        d2 += float((diff * diff).sum())
        w2 += float((w * w).sum())
        g2 += float((g * g).sum())
        gw += float((g * w).sum())
        mx = max(mx, float(diff.abs().max()))
    finite = bool(torch.isfinite(got).all())
    if not finite:
        return {"rel_l2": float("inf"), "cosine": 0.0, "max_abs": float("inf")}
    return {"rel_l2": math.sqrt(d2 / max(w2, 1e-300)), "cosine": gw / max(math.sqrt(g2 * w2), 1e-300),
            "max_abs": mx}


def time_ms(f, q, k, v, iters, warm=3):
    for _ in range(warm):
        f(q, k, v)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        f(q, k, v)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), min(ts)


def emit(out, rec):
    rec["gpu"] = torch.cuda.get_device_name()
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = json.dumps(rec)
    print(line, flush=True)
    if out:
        with open(out, "a") as fh:
            fh.write(line + "\n")


def run_case(ks, name, q, k, v, mode, out, iters, do_time, do_acc):
    h, s = q.shape[1], q.shape[2]
    flops = 4.0 * s * k.shape[2] * D * h
    ref = reference(q, k, v) if do_acc else None
    base = None
    if do_acc and "fv_fwd2" in ks:
        base = ks["fv_fwd2"](q, k, v).float()
    for kn, f in ks.items():
        rec = {"shape": name, "heads": h, "seq": s, "mode": mode, "kernel": kn}
        try:
            if do_acc:
                o = f(q, k, v)
                torch.cuda.synchronize()
                o = o.float()
                rec["vs_fp32"] = metrics(o, ref)
                if base is not None:
                    rec["vs_fwd2"] = metrics(o, base)
                del o
            if do_time:
                med, mn = time_ms(f, q, k, v, iters)
                rec["ms"] = med
                rec["ms_min"] = mn
                rec["tflops"] = flops / med * 1e-9
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
        torch.cuda.empty_cache()
        emit(out, rec)
    del ref, base
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--modes", default="normal,peaked")
    ap.add_argument("--kernels", default="")
    ap.add_argument("--cubins", default="/root/w")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--real", default="", help="comma list of captured q/k/v .pt files")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-acc", action="store_true")
    a = ap.parse_args()
    only = set(a.kernels.split(",")) if a.kernels else None
    ks = build_kernels(a.cubins, only)
    print(f"# kernels: {list(ks)} on {torch.cuda.get_device_name()}", file=sys.stderr)
    if a.real:
        for path in a.real.split(","):
            t = torch.load(path)
            q, k, v = (t[n].cuda().bfloat16().contiguous() for n in "qkv")
            tag = os.path.basename(path).removesuffix(".pt")
            run_case(ks, tag, q, k, v, "real", a.out, a.iters, not a.no_time, not a.no_acc)
            del q, k, v, t
        return
    for si, sn in enumerate(a.shapes.split(",")):
        h, s = SHAPES[sn]
        for mi, mode in enumerate(a.modes.split(",")):
            q, k, v = make_inputs(h, s, mode, 1000 + si * 10 + mi)
            # timing does not depend on the data: time once per shape (first mode)
            run_case(ks, sn, q, k, v, mode, a.out, a.iters, (mi == 0) and not a.no_time, not a.no_acc)
            del q, k, v
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
