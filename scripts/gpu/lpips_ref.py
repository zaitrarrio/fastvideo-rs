#!/usr/bin/env python3
"""Reference LPIPS numbers for the Rust port, from the official package.

sol-engine scores frames with `lpips.LPIPS(net="alex")` on
`lpips.im2tensor(lpips.load_image(path))` (tools/vision/lpips_judge.py). This
does the same for every pair in crates/fastvideo-gpucheck/fixtures/lpips/
pairs.json, on the CPU (float32, the pinned numbers) and on CUDA when present,
and records the package versions and the SHA-256 of the two weight files it
loaded (torchvision AlexNet, LPIPS v0.1 alex linear heads) so the pin can be
tied to the files scripts/gpu/fetch-lpips.sh downloads.

PNG frames are read with PIL as RGB; `lpips.load_image` reads PNGs with
`cv2.imread(path)[:, :, ::-1]`, the same 8-bit RGB array.

    python3 scripts/gpu/lpips_ref.py <fixtures_dir> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(p: str) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> int:
    fixtures = Path(sys.argv[1])
    out = Path(sys.argv[2])
    import numpy as np
    import torch
    import torchvision
    import lpips
    from PIL import Image

    def load(name: str):
        arr = np.asarray(Image.open(fixtures / f"{name}.png").convert("RGB"))
        return lpips.im2tensor(arr)

    model = lpips.LPIPS(net="alex", verbose=False).eval()
    pairs = json.loads((fixtures / "pairs.json").read_text())["pairs"]
    rows = []
    with torch.no_grad():
        for a, b in pairs:
            ta, tb = load(a), load(b)
            row = {"a": a, "b": b, "lpips_cpu": float(model(ta, tb).item())}
            # Per-layer terms (the sum is the score), for a port that disagrees.
            layers = model(ta, tb, retPerLayer=True)[1]
            row["per_layer_cpu"] = [float(t.item()) for t in layers]
            rows.append(row)
        if torch.cuda.is_available():
            mc = lpips.LPIPS(net="alex", verbose=False).eval().cuda()
            for row in rows:
                row["lpips_cuda"] = float(mc(load(row["a"]).cuda(), load(row["b"]).cuda()).item())
    hub = torch.hub.get_dir()
    lin = os.path.join(os.path.dirname(lpips.__file__), "weights", "v0.1", "alex.pth")
    result = {
        "package": {"lpips": getattr(lpips, "__version__", None), "torch": torch.__version__, "torchvision": torchvision.__version__},
        "weights": {
            "alexnet-owt-7be5be79.pth": sha256(os.path.join(hub, "checkpoints", "alexnet-owt-7be5be79.pth")),
            "lpips_v0.1_alex.pth": sha256(lin),
        },
        "pairs": rows,
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
