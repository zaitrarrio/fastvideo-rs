#!/usr/bin/env python3
"""Time upstream FastVideo on the box that just ran our clip stages.

Same weights, same resolution, frame count, DMD timesteps and seed as
`fv-gpucheck clip`, so the only variable is the implementation. Model load is
timed separately and excluded: every reported run has the pipeline resident,
which is how our clip stage measures too.

One backend per process — FastVideo picks its attention backend at import and
caches it — so the caller invokes this once per backend and merges the JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--backend", required=True, help="FASTVIDEO_ATTENTION_BACKEND value")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=129)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vsa-sparsity", type=float, default=None)
    ap.add_argument("--runs", type=int, default=2, help="timed runs after one warm-up")
    ap.add_argument("--prompt", default="A golden retriever puppy playing on a sunny beach, waves in the background")
    ap.add_argument(
        "--workload",
        default="t2v",
        choices=("t2v", "t2i"),
        help="t2v is Wan/FastWan clips; t2i is Flux2 stills (num_frames=1)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", required=True)
    args = ap.parse_args()
    if args.workload == "t2i" and args.num_frames == 129:
        args.num_frames = 1
    if args.workload == "t2i" and args.height == 448 and args.width == 832:
        args.height = 1024
        args.width = 1024

    # Must be set before FastVideo is imported.
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.backend
    if args.vsa_sparsity is not None:
        os.environ["FASTVIDEO_VSA_SPARSITY"] = str(args.vsa_sparsity)

    result: dict[str, object] = {
        "backend": args.backend,
        "workload": args.workload,
        "model_path": args.model_path,
        "spec": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "guidance": args.guidance,
            "fps": args.fps,
            "seed": args.seed,
            "vsa_sparsity": args.vsa_sparsity,
        },
        "runs": [],
    }

    try:
        import torch

        result["torch"] = torch.__version__
        result["gpu"] = torch.cuda.get_device_name(0)
        import fastvideo

        result["fastvideo"] = getattr(fastvideo, "__version__", "unknown")
        from fastvideo import VideoGenerator
    except Exception as e:  # noqa: BLE001 - the whole point is to record why
        result["error"] = f"import: {type(e).__name__}: {e}"
        write(args.out, result)
        return 1

    def sync() -> None:
        torch.cuda.synchronize()

    try:
        t0 = time.perf_counter()
        gen = VideoGenerator.from_pretrained(args.model_path, num_gpus=1)
        sync()
        result["load_seconds"] = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        result["error"] = f"load: {type(e).__name__}: {e}"
        write(args.out, result)
        return 1

    def one(tag: str) -> float:
        kwargs = dict(
            prompt=args.prompt,
            output_path=args.video_dir,
            save_video=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            fps=args.fps,
            seed=args.seed,
        )
        sync()
        t = time.perf_counter()
        gen.generate_video(**kwargs)
        sync()
        dt = time.perf_counter() - t
        print(f"[upstream] {args.backend} {tag}: {dt:.2f}s", flush=True)
        return dt

    try:
        result["warmup_seconds"] = one("warmup")
        times = [one(f"run{i + 1}") for i in range(max(1, args.runs))]
        result["runs"] = times
        result["median_seconds"] = statistics.median(times)
        result["min_seconds"] = min(times)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"generate: {type(e).__name__}: {e}"
        write(args.out, result)
        return 1

    write(args.out, result)
    return 0


def write(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
