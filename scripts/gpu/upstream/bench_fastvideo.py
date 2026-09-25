#!/usr/bin/env python3
"""Upstream FastVideo cell: FastH3 8-step, FastH3 4-step LoRA previews, MiniMax-H3 base.

Reuses FastVideo's own example builders (examples/inference/basic/basic_fasth3.py)
so the generator/engine configuration is exactly the published profile, and
follows its methodology: build the generator (timed here as load), one
excluded warmup request, then N measured requests at one seed. Per-stage
times come from FastVideo's own logging_info (FASTVIDEO_STAGE_LOGGING=1).

Deviations forced by one sm_120 GPU are passed explicitly by the caller and
recorded in the JSON: --num-gpus 1, --vsa-kernel triton (sm100a is
Blackwell-datacenter only), and --no-fa4 when FA4 is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastvideo-src", required=True)
    ap.add_argument("--recipe", required=True, choices=("8step", "lora", "base"))
    ap.add_argument("--out", required=True, help="cell directory")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup-seed", type=int, default=None)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the example's own CLI flags")
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    sys.path.insert(0, str(Path(a.fastvideo_src) / "examples/inference/basic"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"impl": "fastvideo", "recipe": a.recipe, "argv": rest, "runs": [], "ok": False}
    t_proc = time.perf_counter()
    try:
        import basic_fasth3  # noqa: F401  (FastVideo's example module)

        if a.recipe == "8step":
            import basic_fasth3_8step as mod

            args = mod.parse_args(rest + ["--output", str(out), "--repeats", str(a.repeats)])
        elif a.recipe == "lora":
            import basic_fasth3_lora_preview as mod

            args = mod.parse_args(rest + ["--output", str(out), "--repeats", str(a.repeats)])
        else:
            # MiniMax-H3 base (50-point grid): the base example's engine config
            # (text encoder + VAE offloaded, DiT resident, no compile by default).
            args = basic_fasth3.build_parser().parse_args(rest + ["--output", str(out), "--repeats", str(a.repeats)])
            args.vsa = False
        if a.warmup_seed is not None:
            args.warmup_seed = a.warmup_seed
        env = basic_fasth3.configure_environment(args)
        res["profile_env"] = env
        res["args"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}
        basic_fasth3.validate_profile_dependencies(args)
        from fastvideo import VideoGenerator

        if a.recipe == "base":
            from fastvideo.api import (CompileConfig, EngineConfig, GeneratorConfig, OffloadConfig,
                                       ParallelismConfig, PipelineSelection)
            cfg = GeneratorConfig(
                model_path=args.model_path,
                pipeline=PipelineSelection(experimental={"attention_backend": "FLASH_ATTN"}),
                engine=EngineConfig(
                    num_gpus=args.num_gpus, execution_backend="mp", use_fsdp_inference=False,
                    parallelism=ParallelismConfig(tp_size=1, sp_size=args.num_gpus),
                    offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True,
                                          pin_cpu_memory=False, lazy_module_load=args.lazy_module_load),
                    compile=CompileConfig(enabled=False, mode=None, vae_enabled=args.compile_vae),
                ),
            )
        else:
            cfg = basic_fasth3.build_generator_config(args)
        t0 = time.perf_counter()
        gen = VideoGenerator.from_config(cfg)
        res["load_s"] = time.perf_counter() - t0
        try:
            def one(seed: int, path: Path) -> dict:
                t = time.perf_counter()
                r = gen.generate(basic_fasth3.build_request(args, path, seed))
                wall = time.perf_counter() - t
                stages = getattr(getattr(r, "logging_info", None), "stages", None) or {}
                return {
                    "wall_s": wall,
                    "generation_time_s": getattr(r, "generation_time", None),
                    "peak_memory_mb": getattr(r, "peak_memory_mb", None),
                    "stages_s": {k: v.get("execution_time") for k, v in stages.items() if isinstance(v, dict)},
                    "video": str(getattr(r, "video_path", None) or path),
                }

            if args.warmup:
                res["warmup"] = one(args.warmup_seed, out / "warmup.mp4")
            for i in range(a.repeats):
                res["runs"].append(one(args.seed, out / f"run_{i + 1:02d}.mp4"))
        finally:
            gen.shutdown()
        walls = [r["wall_s"] for r in res["runs"]]
        res["warm_total_s"] = statistics.median(walls)
        den = [v for r in res["runs"] for k, v in r["stages_s"].items() if "denois" in k.lower() and v]
        res["denoise_s"] = statistics.median(den) if den else None
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        traceback.print_exc()
    res["process_s"] = time.perf_counter() - t_proc
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res.get(k) for k in ("ok", "load_s", "warm_total_s", "denoise_s", "error")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
