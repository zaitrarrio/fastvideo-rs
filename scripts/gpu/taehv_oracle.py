#!/usr/bin/env python3
"""Reference TAEHV decode, for `fv-gpucheck taehv` to diff against.

Runs madebyollin's own `taehv.py` on a fixed latent and saves both sides of the
call, so our port is judged against the implementation it was read from rather
than against our reading of it.

The one thing this settles that no amount of code-reading can: TAEHV documents
its input as "~Gaussian", which should mean DiT-space latents — *before* the
per-channel latents_mean/latents_std un-normalisation `AutoencoderKLWan`
expects. Feeding the wrong one produces a plausible, wrongly-coloured video
rather than an error, which is exactly how a mirrored UMT5 bias hid for days.

Writes, with frames as the leading axis so both sides agree on layout without
anyone having to transpose:

    latent   [T, 16, H, W]     the decoder input
    video    [F, 3, 8H, 8W]    the reference output, in its native [0, 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

TAEHV_PY = "https://raw.githubusercontent.com/madebyollin/taehv/main/taehv.py"
TAEW_ST = "https://github.com/madebyollin/taehv/raw/main/safetensors/taew2_1.safetensors"


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/taehv", help="where taehv.py and the weights live")
    ap.add_argument("--latent-frames", type=int, default=3)
    ap.add_argument("--height", type=int, default=56, help="latent height (pixels/8)")
    ap.add_argument("--width", type=int, default=104)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    cache = Path(args.cache)
    py = fetch(TAEHV_PY, cache / "taehv.py")
    weights = fetch(TAEW_ST, cache / "taew2_1.safetensors")
    sys.path.insert(0, str(cache))
    from taehv import TAEHV  # noqa: E402  (only importable after the fetch)

    t0 = time.time()
    # arch_name must say taew2_1 or the architecture is guessed from the path.
    tae = TAEHV(checkpoint_path=str(weights), arch_name="taew2_1").to(args.device).float().eval()
    load_s = time.time() - t0

    # Generated on the CPU and saved, so both sides consume identical bytes and
    # nothing depends on a device RNG.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.randn(
        (1, args.latent_frames, tae.latent_channels, args.height, args.width),
        generator=g,
        dtype=torch.float32,
    )

    with torch.no_grad():
        t0 = time.time()
        video = tae.decode_video(latent.to(args.device), parallel=True, show_progress_bar=False)
        if args.device == "cuda":
            torch.cuda.synchronize()
        decode_s = time.time() - t0

    meta = {
        "arch_name": tae.arch_name,
        "latent_channels": tae.latent_channels,
        "patch_size": tae.patch_size,
        "t_upscale": tae.t_upscale,
        "frames_to_trim": tae.frames_to_trim,
        "load_s": load_s,
        "decode_s": decode_s,
        "latent_shape": list(latent.shape),
        "video_shape": list(video.shape),
        "video_range": [float(video.min()), float(video.max())],
        "torch": torch.__version__,
    }
    save_file(
        {
            "latent": latent[0].contiguous(),
            "video": video[0].float().cpu().contiguous(),
        },
        args.out,
    )
    json.dump(meta, open(args.meta, "w"), indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
