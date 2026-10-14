#!/usr/bin/env python3
"""FastVideo's FastH3 NVFP4 recipe on one RTX PRO 6000, as its own page runs it
(docs/inference/fasth3_rtx_pro_6000.md and scripts/benchmarks/minimax_h3_pro6000/app.py
``run_variant`` at FastVideo d5253fd), for the fastvideo-rs comparison (docs/ports/fasth3-nvfp4.md).

    bench_fasth3_nvfp4.py --model DIR --sparsity 0.8 --runs RUNS.json --out DIR

DIR is a model folder whose ``transformer/`` was repacked by
``convert_minimax_h3_modelopt_nvfp4_dit.py --quantize-attention --quantize-gate`` (layer_profile
``h3_dit_vsa``) and whose ``text_encoder/`` by ``convert_minimax_h3_text_encoder_nvfp4.py``; every
component resident, no torch.compile of the DiT, the VAE compiled (``vae_enabled``), and the page's
environment: block-sparse FP4 attention, the H3 fusions, the cutlass FP4 GEMM, 28 VAE tiles per call.

RUNS.json is a list of requests, run in order in one loaded generator::

    [{"tag": "v2-480p-h3-demo-cold", "prompt": "...", "seed": 1024, "height": 480, "width": 832,
      "num_frames": 124, "steps": 9}, ...]

The first request is the cold one (FlashInfer JIT, Triton autotuning, the VAE compile at its shape) and,
with ``FV_ORACLE_DUMP_DIR`` set, the one the oracle hooks dump (oracle_dump.py dumps the first denoise
only). A request FastVideo refuses (8 grid points for the 8-rung V2) is recorded with its error and the
rest go on. Each run's wall-clock window (epoch seconds) is recorded so the pod can take its device
memory peak from an nvidia-smi trace: the DiT may run in a worker process, where this process's torch
counters see nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# The page's switches (app.py FAST_ENV) and its image's environment.
FAST_ENV = {
    "FASTVIDEO_H3_VSA_FP4": "1",
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
    "FASTVIDEO_NVFP4_MM_BACKEND": "cutlass",
    "FASTVIDEO_H3_VAE_TILE_BATCH": "28",
}
IMAGE_ENV = {
    "FLASHINFER_CUDA_ARCH_LIST": "12.0a",
    "FASTVIDEO_VSA_SM100A": "0",
    "FASTVIDEO_FA4": "0",
    "FASTVIDEO_STAGE_LOGGING": "1",
    "FASTVIDEO_VSA_TRITON": "1",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, required=True, help="VSA_sparsity: 0.8 (V2), 0.9 (4-step)")
    ap.add_argument("--layer-profile", default="h3_dit_vsa")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-vae-compile", action="store_true")
    a = ap.parse_args()
    for k, v in {**IMAGE_ENV, **FAST_ENV}.items():
        os.environ.setdefault(k, v)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = json.loads(Path(a.runs).read_text())
    res: dict = {"impl": "fastvideo", "model": a.model, "sparsity": a.sparsity, "layer_profile": a.layer_profile,
                 "env": {k: os.environ.get(k) for k in sorted({**IMAGE_ENV, **FAST_ENV})}, "runs": [], "ok": False}
    t_proc = time.time()
    try:
        import torch
        from fastvideo import VideoGenerator

        config = {
            "model_path": a.model,
            "engine": {
                "num_gpus": 1, "use_fsdp_inference": False,
                "quantization": {"transformer_quant": "NVFP4", "layer_profile": a.layer_profile},
                "parallelism": {"tp_size": 1, "sp_size": 1},
                "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False, "vae": False,
                            "pin_cpu_memory": True, "lazy_module_load": False},
                "compile": {"enabled": False, "vae_enabled": not a.no_vae_compile},
            },
            "pipeline": {"experimental": {
                "attention_backend": "VIDEO_SPARSE_ATTN_H3", "h3_sequential_load": False,
                "inference_torch_compile": False, "vae_parallel_decode": False, "video_decode_backend": "h3-vae",
                "VSA_sparsity": a.sparsity, "VSA_tile_size": 64,
            }},
        }
        res["config"] = config
        t0 = time.time()
        gen = VideoGenerator.from_config(config)
        res["load_s"] = time.time() - t0
        try:
            for i, r in enumerate(runs):
                path = out / f"{i:02d}_{r['tag']}.mp4"
                req = {"prompt": r["prompt"], "negative_prompt": "",
                       "sampling": {"seed": int(r["seed"]), "height": int(r["height"]), "width": int(r["width"]),
                                    "num_frames": int(r["num_frames"]), "fps": 24,
                                    "num_inference_steps": int(r["steps"]), "guidance_scale": 1.0,
                                    "batch_cfg": False},
                       "output": {"output_path": str(path), "save_video": True, "return_frames": False}}
                rec = {k: r[k] for k in ("tag", "seed", "height", "width", "num_frames", "steps")}
                rec["index"] = i
                torch.cuda.synchronize()
                rec["t_start"] = time.time()
                try:
                    result = gen.generate(req)
                    torch.cuda.synchronize()
                    rec["t_end"] = time.time()
                    rec["wall_s"] = rec["t_end"] - rec["t_start"]
                    rec["generation_time_s"] = getattr(result, "generation_time", None)
                    rec["peak_memory_mb"] = getattr(result, "peak_memory_mb", None)
                    stages = getattr(getattr(result, "logging_info", None), "stages", None) or {}
                    rec["stages_s"] = {k: v.get("execution_time") for k, v in stages.items() if isinstance(v, dict)}
                    rec["video"] = str(getattr(result, "video_path", None) or path)
                    rec["ok"] = True
                except Exception as e:  # noqa: BLE001  (a refused request; the generator stays loaded)
                    rec["t_end"] = time.time()
                    rec["ok"] = False
                    rec["error"] = f"{type(e).__name__}: {e}"[:2000]
                    traceback.print_exc()
                res["runs"].append(rec)
                print("RUN", json.dumps(rec, default=str), flush=True)
                (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        finally:
            gen.shutdown()
        res["ok"] = any(r.get("ok") for r in res["runs"])
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        traceback.print_exc()
    res["process_s"] = time.time() - t_proc
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res.get(k) for k in ("ok", "load_s", "process_s", "error")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
