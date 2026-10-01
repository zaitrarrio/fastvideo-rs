#!/usr/bin/env python3
"""Convert the LongLive-1.3B torch checkpoints to safetensors (same keys).

    python convert-longlive.py <longlive-1.3b tree> <out dir>

<tree> is the Hub snapshot of Efficient-Large-Model/LongLive-1.3B
(models/longlive_base.pt, models/lora.pt). Writes, into <out dir> (which must
not exist yet; written as <out>.partial-<stamp> and renamed at the end):

- longlive_base.safetensors: the `generator` state dict of longlive_base.pt
  (original Wan names under `model.`, as LongLive's interactive_inference.py
  loads it with use_ema: false);
- lora.safetensors: the `generator_lora` state dict of lora.pt (PEFT keys
  `base_model.model.blocks.N.<module>.lora_{A,B}.weight`, rank 256);
- keys-<name>.txt (key, shape, dtype) and sha256.txt.

Same tensors, same dtypes: every tensor is read back and compared with
torch.equal. The Rust side (crates/fastvideo-cudarc/src/wan/longlive.rs)
renames to Diffusers and merges the LoRA at load. Needs torch (CPU) and
safetensors; about 2x the larger checkpoint (5.7 GB) of host memory.
"""
import hashlib
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

JOBS = [("longlive_base", "models/longlive_base.pt", "generator"),
        ("lora", "models/lora.pt", "generator_lora")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(tree, out):
    tree, out = Path(tree), Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists: add-only, not touching it")
    tmp = out.parent / f".{out.name}.partial-{time.strftime('%Y%m%d%H%M%S')}"
    tmp.mkdir(parents=True)
    lines = []
    for name, rel, sub in JOBS:
        sd = torch.load(tree / rel, map_location="cpu", weights_only=True, mmap=True)
        if sub not in sd:
            raise SystemExit(f"{rel}: no `{sub}` (keys {list(sd)[:8]})")
        sd = sd[sub]
        tensors = {k: v.detach().contiguous() for k, v in sd.items() if torch.is_tensor(v)}
        skipped = [k for k, v in sd.items() if not torch.is_tensor(v)]
        path = tmp / f"{name}.safetensors"
        save_file(tensors, str(path), metadata={"source": rel, "sub_dict": sub})
        with safe_open(str(path), framework="pt") as f:
            if set(f.keys()) != set(tensors):
                raise SystemExit(f"{name}: key set differs after the round trip")
            for k in tensors:
                if not torch.equal(f.get_tensor(k), tensors[k]):
                    raise SystemExit(f"{name}: {k} differs after the round trip")
        with open(tmp / f"keys-{name}.txt", "w") as fk:
            for k in sorted(tensors):
                fk.write(f"{k}\t{list(tensors[k].shape)}\t{tensors[k].dtype}\n")
        lines.append(f"{sha256(path)}  {name}.safetensors")
        print(name, len(tensors), "tensors", sorted({str(v.dtype) for v in tensors.values()}),
              "skipped", skipped, path.stat().st_size, "bytes", flush=True)
        del sd, tensors
    (tmp / "sha256.txt").write_text("\n".join(lines) + "\n")
    (tmp / ".complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    tmp.rename(out)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
