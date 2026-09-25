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
import statistics
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
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
                 "duration": a.duration, "te_offload": a.te_offload, "runs": [], "ok": False}
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

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        engine = MiniMaxH3Inference(a.model, a.adapter, attention_backend="dense", task="t2v")
        res["load_s"] = time.perf_counter() - t0
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
