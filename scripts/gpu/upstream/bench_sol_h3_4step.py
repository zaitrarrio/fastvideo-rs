#!/usr/bin/env python3
"""sol-engine Sol-H3 (models/minimax_h3/Sol-H3) four-step T2V on one GPU.

Methodology is the package's own (README "T2V benchmarks"): load once
(MiniMaxH3Inference), one warmup request, then the median of N requests at one
seed; the timed span is `engine.generate` (text encoding + 4 DiT forwards +
video/audio VAE decode), excluding load, warmup and MP4 encoding. On one GPU
the engine only allows dense attention.

One deviation, recorded in the JSON: the released engine moves the whole
pipeline to the GPU (Qwen3-VL text encoder 66 GB + DiT 66 GB + VAEs), which
does not fit 96 GB. With --te-offload the text encoder is kept on the host and
streamed per forward (accelerate.cpu_offload); everything else stays resident.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


def gemm_probe(torch) -> dict:
    """Small GPU GEMMs (BF16 and FP32, default BLAS and cuBLASLt) plus library versions."""
    from importlib import metadata

    out: dict = {"torch": torch.__version__, "cuda": torch.version.cuda,
                 "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()
                              if (d.metadata["Name"] or "").lower().startswith(("nvidia-cublas", "nvidia-cuda-runtime",
                                                                                "nvidia-cudnn", "nvidia-nvjitlink"))}}
    shapes = [(64, 64, 64), (5120, 64, 5120)]
    for lib in ("default", "cublaslt"):
        ok = True
        if lib == "cublaslt":
            torch.backends.cuda.preferred_blas_library("cublaslt")
        for dt in (torch.bfloat16, torch.float32):
            for m, k, n in shapes:
                key = f"{lib}:{str(dt).split('.')[-1]}:{m}x{k}x{n}"
                try:
                    a = torch.randn(m, k, device="cuda", dtype=dt)
                    b = torch.randn(k, n, device="cuda", dtype=dt)
                    (a @ b).sum().item()
                    out[key] = "ok"
                except RuntimeError as e:
                    out[key] = str(e)[:160]
                    ok = False
        out[f"{lib}_ok"] = ok
    torch.backends.cuda.preferred_blas_library("default")
    out["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")
    try:
        with open("/proc/self/maps") as fh:
            out["loaded"] = sorted({ln.split()[-1] for ln in fh if "libcublas" in ln})
    except OSError:
        pass
    print("gemm_probe", out, flush=True)
    return out


_PROBE = "import torch; a = torch.ones(64, 64, device='cuda'); print((a @ a).sum().item())"


def library_path_fix() -> None:
    """Re-exec under a library path where a GEMM works, if the inherited one fails.

    torch 2.10.0+cu130 pins nvidia-cublas 13.1.0.3. The base image's CUDA 13.0 on
    LD_LIBRARY_PATH can supply a mismatched libcublasLt, and then every GEMM fails
    (CUBLAS_STATUS_INVALID_VALUE / NOT_INITIALIZED). Each attempt is recorded."""
    if os.environ.get("H3B_REEXEC"):
        return
    orig = os.environ.get("LD_LIBRARY_PATH", "")
    venv = Path(sys.executable).parent.parent
    wheel_libs = sorted(str(d) for d in venv.glob("lib/python3*/site-packages/nvidia/**/lib") if d.is_dir())
    tried = []
    for label, ldlp in (("inherited", orig), ("wheel-libs-first", ":".join([*wheel_libs, orig])), ("unset", None)):
        env = dict(os.environ)
        if ldlp is None:
            env.pop("LD_LIBRARY_PATH", None)
        else:
            env["LD_LIBRARY_PATH"] = ldlp
        r = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True)
        last = (r.stderr.strip().splitlines() or ["?"])[-1][:120]
        tried.append(f"{label}: {'ok' if r.returncode == 0 else last}")
        if r.returncode == 0:
            if label == "inherited":
                break
            env["H3B_REEXEC"] = " | ".join(tried)
            env["H3B_LDLP_ORIG"] = orig
            os.execve(sys.executable, [sys.executable, *sys.argv], env)
    os.environ["H3B_REEXEC"] = " | ".join(tried)


def main() -> int:
    library_path_fix()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol-h3", required=True, help="models/minimax_h3/Sol-H3 directory")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--te-offload", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.sol_h3)
    res: dict = {"impl": "sol-engine/Sol-H3", "attention_backend": "dense", "seed": a.seed,
                 "duration": a.duration, "te_offload": a.te_offload, "runs": [], "ok": False,
                 "library_path_probe": os.environ.get("H3B_REEXEC"),
                 "ld_library_path_inherited": os.environ.get("H3B_LDLP_ORIG", os.environ.get("LD_LIBRARY_PATH"))}
    try:
        import torch

        if a.te_offload:
            from accelerate import cpu_offload
            from diffusers import ModularPipeline

            orig_to = ModularPipeline.to

            def to(self, *args, **kw):
                te = getattr(self, "text_encoder", None)
                if te is None:
                    return orig_to(self, *args, **kw)
                te.to = lambda *x, **y: te  # keep it on the host during the bulk move
                try:
                    r = orig_to(self, *args, **kw)
                finally:
                    del te.to
                dev = args[0] if args else kw.get("device")
                cpu_offload(te, execution_device=torch.device(dev))
                return r

            ModularPipeline.to = to
        from h3_runtime import MiniMaxH3Inference

        # On this sm_120 box the adapter fuse (weight.addmm_(B, A) in BF16) fails
        # with CUBLAS_STATUS_INVALID_VALUE, and so did an FP32 GPU product
        # (cublasSgemm). Probe GEMMs before loading (library versions, BF16/FP32,
        # default vs cuBLASLt) so the failure is characterised; prefer cuBLASLt
        # when the default path fails and it does not.
        res["gemm_probe"] = gemm_probe(torch)
        probe = res["gemm_probe"]
        if not probe.get("default_ok") and probe.get("cublaslt_ok"):
            torch.backends.cuda.preferred_blas_library("cublaslt")
            res["blas_library"] = "cublaslt"
        # Adapter fuse fallback: the low-rank product on the host in FP32 (one
        # rounding when added to the BF16 weight), recorded per call.
        orig_addmm_ = torch.Tensor.addmm_
        res["addmm_fallbacks"] = 0

        def addmm_(self, m1, m2, *args, beta=1, alpha=1):
            try:
                return orig_addmm_(self, m1, m2, *args, beta=beta, alpha=alpha)
            except RuntimeError as e:
                if "CUBLAS" not in str(e) or args:
                    raise
                if res["addmm_fallbacks"] == 0:
                    res["addmm_first_failure"] = {
                        "error": str(e)[:200], "self": [list(self.shape), list(self.stride()), str(self.dtype)],
                        "m1": [list(m1.shape), list(m1.stride())], "m2": [list(m2.shape), list(m2.stride())]}
                res["addmm_fallbacks"] += 1
                prod = torch.matmul(m1.float().cpu(), m2.float().cpu()).mul_(alpha)
                if beta != 1:
                    self.mul_(beta)
                return self.add_(prod.to(device=self.device, dtype=self.dtype))

        torch.Tensor.addmm_ = addmm_

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        engine = MiniMaxH3Inference(a.model, a.adapter, attention_backend="dense", task="t2v")
        res["load_s"] = time.perf_counter() - t0
        torch.Tensor.addmm_ = orig_addmm_
        with engine:
            t = time.perf_counter()
            engine.warmup(duration=a.duration, prompt=a.prompt)
            res["warmup_s"] = time.perf_counter() - t
            torch.cuda.reset_peak_memory_stats()
            for i in range(a.repeats):
                r = engine.generate(a.prompt, duration=a.duration, seed=a.seed)
                res["runs"].append({"inference_s": r.elapsed_s})
                if i == a.repeats - 1:
                    t = time.perf_counter()
                    r.save(out / "out.mp4")
                    res["encode_s"] = time.perf_counter() - t
            res["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            res["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        res["warm_total_s"] = statistics.median(r["inference_s"] for r in res["runs"])
        res["dit_forwards"] = 4
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        traceback.print_exc()
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res.get(k) for k in ("ok", "load_s", "warm_total_s", "error")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
