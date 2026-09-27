#!/usr/bin/env python3
"""Upstream FastVideo cell: FastWan2.1-T2V-1.3B DMD (3 steps), 480x832, 81 frames.

Follows FastVideo's own examples/inference/basic/basic_dmd.py: VIDEO_SPARSE_ATTN
with VSA_sparsity=0.8, the checkpoint's own SamplingParam (DMD timesteps
1000/757/522, guidance 1), save_video=True. Methodology as bench_fastvideo.py
(the H3 cells): build the generator (timed as load), one excluded warm-up
request, then every prompt of the set --repeats times; each prompt's number is
the median of its repeats, the cell's number the median over prompts.

Deviations, recorded in result.json: one GPU (num_gpus=1); the text encoder
stays on the GPU (text_encoder_cpu_offload=False, as our resident UMT5; the
example offloads it for < 32 GB cards); on sm_120 the VSA kernel is Triton
(FASTVIDEO_VSA_KERNEL / the fastvideo-kernel wheel has no sm_120 CUDA build).

Wan2.1-T2V-14B (pod.sh fv-wan21-14b): dense FLASH_ATTN, UniPC 50 steps, CFG 5,
480x832, 81 frames, flow_shift 3.0 -- FastVideo's WanT2V480PConfig recipe, the
same settings as our wan14 cells. FastVideo's registry maps this checkpoint to
WanT2V720PConfig / preset wan_t2v_14b (720x1280, flow_shift 5.0); the 480p
recipe is the one our matrix runs, so both sides are set explicitly.

SF-Wan 1.3B (pod.sh fv-sfwan13, oracle.sh sfwan13): wlsaidhi/SFWan2.1-T2V-1.3B-
Diffusers through FastVideo's WanCausalDMDPipeline at its own defaults
(SelfForcingWanT2V480PConfig: DMD steps 1000/750/500/250 warped by the
checkpoint's SelfForcingFlowMatchScheduler, 3-frame blocks through the KV
cache), dense FLASH_ATTN, 480x832, 81 frames. ``--no-warmup`` (the oracle)
skips the excluded warm-up request so the dump hooks see the first one.

The weights on our volume are FastVideo/FastWan2.1-T2V-1.3B-Diffusers under a
different directory name; FastVideo resolves the pipeline config by the
checkpoint's short name, so the cell links the directory under that name.
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

HF_NAME = "FastWan2.1-T2V-1.3B-Diffusers"

# FastVideo/FastWan2.1-T2V-1.3B-Diffusers scheduler/scheduler_config.json (Hub main).
SCHEDULER = {
    "_class_name": "UniPCMultistepScheduler", "_diffusers_version": "0.33.0.dev0",
    "beta_end": 0.02, "beta_schedule": "linear", "beta_start": 0.0001, "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995, "final_sigmas_type": "zero", "flow_shift": 3.0,
    "lower_order_final": True, "num_train_timesteps": 1000, "predict_x0": True,
    "prediction_type": "flow_prediction", "rescale_betas_zero_snr": False, "sample_max_value": 1.0,
    "solver_order": 2, "solver_p": None, "solver_type": "bh2", "steps_offset": 0,
    "thresholding": False, "timestep_spacing": "linspace", "trained_betas": None,
    "use_beta_sigmas": False, "use_exponential_sigmas": False, "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}


# wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers scheduler/scheduler_config.json (Hub main).
SF_SCHEDULER = {
    "_class_name": "SelfForcingFlowMatchScheduler", "_diffusers_version": "0.33.0.dev0",
    "num_inference_steps": 1000, "shift": 5.0, "sigma_min": 0.0, "extra_one_step": True,
    "training": True,
}


def stage_times(li) -> dict:
    stages = getattr(li, "stages", None) or {}
    out = {}
    for k, v in stages.items():
        if isinstance(v, dict):
            out[k] = v.get("execution_time")
    return out


def pick(stages: dict, *needles: str) -> float | None:
    vals = [v for k, v in stages.items() if v and any(n in k.lower().replace("_", "") for n in needles)]
    return sum(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Diffusers directory")
    ap.add_argument("--out", required=True, help="cell directory")
    ap.add_argument("--prompts", help="prompt-set JSON (scripts/gpu/prompts-eval.json)")
    ap.add_argument("--prompt", help="one prompt instead of a set (with --seed)")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--hf-name", default=HF_NAME,
                    help="the checkpoint's Hub short name (FastVideo resolves its config by it)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-seed", type=int, default=999)
    ap.add_argument("--no-warmup", action="store_true", help="no excluded warm-up request (the oracle)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=None, help="num_inference_steps (default: the checkpoint's)")
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument("--flow-shift", type=float, default=None,
                    help="pipeline flow_shift (default: the checkpoint's pipeline config)")
    ap.add_argument("--vsa-sparsity", type=float, default=0.8)
    ap.add_argument("--attention", default="VIDEO_SPARSE_ATTN")
    ap.add_argument("--text-encoder-cpu-offload", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"impl": "fastvideo", "model": a.hf_name, "args": vars(a), "runs": [], "ok": False}
    t_proc = time.perf_counter()
    try:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = a.attention
        os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
        # sm100a VSA kernels are Blackwell-datacenter only: Triton VSA on sm_120.
        os.environ.setdefault("FASTVIDEO_VSA_SM100A", "0")
        res["env"] = {k: v for k, v in os.environ.items() if k.startswith("FASTVIDEO_")}
        # A view of the volume's tree under the Hub name: every component
        # symlinked, plus scheduler/ (a 1 KB config our volume copy lacks and
        # FastVideo's loader requires; the DMD sampler does not use it).
        view = out / "model" / a.hf_name
        view.mkdir(parents=True, exist_ok=True)
        src = Path(a.model).resolve()
        for entry in src.iterdir():
            dst = view / entry.name
            if not dst.exists():
                dst.symlink_to(entry)
        if not (view / "scheduler").exists() and a.hf_name == HF_NAME:
            (view / "scheduler").mkdir()
            (view / "scheduler" / "scheduler_config.json").write_text(json.dumps(SCHEDULER, indent=2))
            res["scheduler_config"] = "written (Hub FastVideo/FastWan2.1-T2V-1.3B-Diffusers scheduler/)"
        if not (view / "scheduler").exists() and a.hf_name.startswith("SFWan2.1"):
            (view / "scheduler").mkdir()
            (view / "scheduler" / "scheduler_config.json").write_text(json.dumps(SF_SCHEDULER, indent=2))
            res["scheduler_config"] = "written (Hub wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers scheduler/)"
        model = str(view)
        if a.prompts:
            spec = json.loads(Path(a.prompts).read_text())
            prompts = spec["prompts"] if isinstance(spec, dict) else spec
        else:
            prompts = [{"name": "default", "prompt": a.prompt or "a cat walking on the grass", "seed": a.seed}]
        from fastvideo import VideoGenerator
        from fastvideo.api.sampling_param import SamplingParam

        t0 = time.perf_counter()
        gen = VideoGenerator.from_pretrained(
            model,
            num_gpus=1,
            use_fsdp_inference=False,
            text_encoder_cpu_offload=a.text_encoder_cpu_offload,
            pin_cpu_memory=True,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            **({"VSA_sparsity": a.vsa_sparsity} if a.attention == "VIDEO_SPARSE_ATTN" else {}),
            **({"flow_shift": a.flow_shift} if a.flow_shift is not None else {}),
        )
        res["load_s"] = time.perf_counter() - t0
        try:
            res["pipeline_flow_shift"] = gen.fastvideo_args.pipeline_config.flow_shift
        except Exception:  # noqa: BLE001
            pass
        try:
            def one(prompt: str, seed: int, path: Path) -> dict:
                sp = SamplingParam.from_pretrained(model)
                sp.num_frames = a.num_frames
                sp.height = a.height
                sp.width = a.width
                sp.seed = seed
                if a.steps is not None:
                    sp.num_inference_steps = a.steps
                if a.guidance_scale is not None:
                    sp.guidance_scale = a.guidance_scale
                t = time.perf_counter()
                r = gen.generate_video(prompt, sampling_param=sp, output_path=str(path), save_video=True)
                wall = time.perf_counter() - t
                r = r[0] if isinstance(r, list) else r
                st = stage_times(r.get("logging_info"))
                return {
                    "wall_s": wall,
                    "generation_time_s": r.get("generation_time"),
                    "peak_memory_mb": r.get("peak_memory_mb"),
                    "stages_s": st,
                    "denoise_s": pick(st, "denois"),
                    "decode_s": pick(st, "decodingstage"),
                    "postprocess_s": pick(st, "postdecode"),
                    "text_s": pick(st, "textencod", "promptencod"),
                    "save_s": pick(st, "save"),
                    "steps": getattr(sp, "num_inference_steps", None),
                    "guidance_scale": getattr(sp, "guidance_scale", None),
                }

            if not a.no_warmup:
                res["warmup"] = one(prompts[0]["prompt"], a.warmup_seed, out / "warmup")
            for p in prompts:
                runs = [one(p["prompt"], int(p.get("seed", 1024)), out / p["name"] / f"run_{i + 1:02d}") for i in range(a.repeats)]
                res["runs"].append({"name": p["name"], "seed": p.get("seed"), "runs": runs})
        finally:
            gen.shutdown()

        def med(key: str, rows: list[dict]) -> float | None:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return statistics.median(vals) if vals else None

        per_prompt = []
        for p in res["runs"]:
            per_prompt.append({
                "name": p["name"],
                "wall_s": med("wall_s", p["runs"]),
                "denoise_s": med("denoise_s", p["runs"]),
                "decode_s": med("decode_s", p["runs"]),
                "text_s": med("text_s", p["runs"]),
                "save_s": med("save_s", p["runs"]),
                "postprocess_s": med("postprocess_s", p["runs"]),
                "peak_memory_mb": max((r["peak_memory_mb"] or 0) for r in p["runs"]) or None,
            })
        res["per_prompt"] = per_prompt
        for k in ("wall_s", "denoise_s", "decode_s", "text_s", "save_s", "postprocess_s"):
            res[f"median_{k}"] = med(k, per_prompt)
        res["warm_total_s"] = res["median_wall_s"]
        res["peak_memory_mb"] = max((p["peak_memory_mb"] or 0) for p in per_prompt) or None
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        traceback.print_exc()
    res["process_s"] = time.perf_counter() - t_proc
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res.get(k) for k in ("ok", "load_s", "warm_total_s", "median_denoise_s", "median_decode_s", "error")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
