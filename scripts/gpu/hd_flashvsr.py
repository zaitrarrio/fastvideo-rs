#!/usr/bin/env python3
"""FlashVSR v1.1 (Tiny decoder, official pipeline) on the hd benchmark's 768p clips.

    python hd_flashvsr.py <FlashVSR checkout> <RUNS> <scale,...> <clip>...

Run by hd-upscaler.sh (FV_HD_FLASHVSR=1) from <checkout>/examples/WanVSR, where
FlashVSR-v1.1/ holds symlinks to the volume copy (auxiliary/upscalers/flashvsr-v1.1).
The pipeline is built once (the official infer_flashvsr_v1.1_tiny.py
init_pipeline, block-sparse attention, sparse_ratio 2.0, kv_ratio 3.0,
local_range 11, colour fix on). For each clip and scale the input is the
768p PNG sequence (the script's own bicubic upscale and centre crop to
multiples of 128: x1.5 -> 1920x1152, x2 -> 2688x1536). Frames go to
<RUNS>/flashvsr-v1.1-x<scale>/frames/<clip>/, one JSON line per run to stdout
(model seconds with CUDA synchronised, input preparation separately, peak
allocated memory).
"""
import importlib.util
import json
import os
import sys
import time

import torch

repo, runs, scales, clips = sys.argv[1], sys.argv[2], [float(s) for s in sys.argv[3].split(",")], sys.argv[4:]
wan = os.path.join(repo, "examples", "WanVSR")
sys.path[:0] = [wan, repo]
os.chdir(wan)
spec = importlib.util.spec_from_file_location("fvsr_tiny", os.path.join(wan, "infer_flashvsr_v1.1_tiny.py"))
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)

t0 = time.time()
pipe = tiny.init_pipeline()
torch.cuda.synchronize()
print(json.dumps({"event": "load", "seconds": round(time.time() - t0, 2),
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}), flush=True)


def run(clip, scale, tag):
    src = os.path.join(runs, "turbo-768p", "frames", clip)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    lq, th, tw, f, _ = tiny.prepare_input_tensor(src, scale=scale, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    t2 = time.time()
    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
        LQ_video=lq, num_frames=f, height=th, width=tw, is_full_block=False, if_buffer=True,
        topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0, local_range=11, color_fix=True,
    )
    torch.cuda.synchronize()
    t3 = time.time()
    frames = tiny.tensor2video(video)
    out = os.path.join(runs, f"flashvsr-v1.1-x{scale:g}", "frames", clip)
    if tag == "":
        os.makedirs(out, exist_ok=True)
        for i, im in enumerate(frames[:124]):
            im.save(os.path.join(out, f"frame-{i:03d}.png"))
    n_in = len([p for p in os.listdir(src) if p.endswith(".png")])
    print(json.dumps({
        "event": "run", "clip": clip, "scale": scale, "tag": tag, "size": f"{tw}x{th}", "frames_in": n_in,
        "frames_model": f, "frames_out": len(frames), "prep_s": round(t2 - t1, 2), "model_s": round(t3 - t2, 2),
        "fps_out": round(len(frames) / (t3 - t2), 2), "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }), flush=True)


first = True
for clip in clips:
    for scale in scales:
        run(clip, scale, "")
        if first:
            run(clip, scale, "warm")
            first = False
