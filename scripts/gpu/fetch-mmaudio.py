#!/usr/bin/env python3
"""Fetch MMAudio large-44k-v2 and everything its 44k V2A inference needs.

Runs on a CPU fetch pod with the weight volume at /workspace (see
docs/ports/mmaudio.md "Weights"). Upstream layout is kept under
/workspace/weights/mmaudio-44k-v2/; each checkpoint is also converted to a
safetensors file under safetensors/ (same keys, same dtype; a round-trip
equality check), which is what the Rust port reads. md5s are the ones pinned
in MMAudio's mmaudio/utils/download_utils.py. Writes .complete last.

Progress is appended to /srv/log.txt; /srv/DONE holds 0 (ok) or 1.
"""
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("MMAUDIO_ROOT", "/workspace/weights/mmaudio-44k-v2"))
SRV = Path(os.environ.get("FETCH_SRV", "/srv"))
LOG = SRV / "log.txt"

MD5 = {
    "weights/mmaudio_large_44k_v2.pth": "01ad4464f049b2d7efdaa4c1a59b8dfe",
    "ext_weights/v1-44.pth": "fab020275fa44c6589820ce025191600",
    "ext_weights/synchformer_state_dict.pth": "5b2f5594b0730f70e41e549b7c94390c",
}
# (hub repo, files, subdir under ROOT). MMAudio: eval_utils.large_44k_v2 +
# FeaturesUtils (CLIP hf-hub:apple/DFN5B-CLIP-ViT-H-14-384, Synchformer) +
# AutoEncoderModule 44k (BigVGAN v2 nvidia/bigvgan_v2_44khz_128band_512x).
JOBS = [
    ("hkchengrex/MMAudio",
     ["weights/mmaudio_large_44k_v2.pth", "ext_weights/v1-44.pth",
      "ext_weights/synchformer_state_dict.pth"], ""),
    ("nvidia/bigvgan_v2_44khz_128band_512x", ["config.json", "bigvgan_generator.pt"],
     "bigvgan_v2_44khz_128band_512x"),
    ("apple/DFN5B-CLIP-ViT-H-14-384",
     ["open_clip_config.json", "open_clip_pytorch_model.bin", "tokenizer.json",
      "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
      "config.json", "preprocessor_config.json"], "DFN5B-CLIP-ViT-H-14-384"),
]
CONVERT = {
    "mmaudio_large_44k_v2": ("weights/mmaudio_large_44k_v2.pth", None),
    "vae_44k": ("ext_weights/v1-44.pth", None),
    "synchformer": ("ext_weights/synchformer_state_dict.pth", None),
    "bigvgan_v2_44k": ("bigvgan_v2_44khz_128band_512x/bigvgan_generator.pt", "generator"),
    "clip_dfn5b_h14_384": ("DFN5B-CLIP-ViT-H-14-384/open_clip_pytorch_model.bin", None),
}


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with LOG.open("a") as f:
        f.write(s + "\n")


def main():
    from huggingface_hub import hf_hub_download
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / ".complete").unlink(missing_ok=True)
    for repo, files, sub in JOBS:
        for f in files:
            t = time.time()
            p = hf_hub_download(repo, f, local_dir=str(ROOT / sub) if sub else str(ROOT))
            log("got", repo, f, os.path.getsize(p), f"{time.time() - t:.1f}s")
    for rel, want in MD5.items():
        h = hashlib.md5()
        with open(ROOT / rel, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        got = h.hexdigest()
        log("md5", rel, got, "OK" if got == want else "MISMATCH want " + want)
        if got != want:
            raise SystemExit("md5 mismatch")

    import torch
    from safetensors.torch import load_file, save_file
    st = ROOT / "safetensors"
    st.mkdir(exist_ok=True)
    summary = {}
    for name, (rel, sub) in CONVERT.items():
        sd = torch.load(ROOT / rel, map_location="cpu", weights_only=True)
        if sub:
            sd = sd[sub]
        out = {k: v.detach().contiguous().clone() for k, v in sd.items() if torch.is_tensor(v)}
        skipped = [k for k, v in sd.items() if not torch.is_tensor(v)]
        path = st / f"{name}.safetensors"
        save_file(out, str(path))
        back = load_file(str(path))
        ok = set(back) == set(out) and all(torch.equal(back[k], out[k]) for k in out)
        nparam = sum(v.numel() for v in out.values())
        dtypes = sorted({str(v.dtype) for v in out.values()})
        log("convert", name, len(out), "tensors", nparam, "elems", dtypes, "skipped", skipped,
            "roundtrip", ok, os.path.getsize(path))
        if not ok:
            raise SystemExit("roundtrip mismatch " + name)
        summary[name] = {"source": rel, "tensors": len(out), "elems": nparam, "dtypes": dtypes,
                         "bytes": os.path.getsize(path)}
        with open(SRV / f"keys-{name}.txt", "w") as fk:
            for k in sorted(out):
                fk.write(f"{k}\t{list(out[k].shape)}\t{out[k].dtype}\n")
        del sd, out, back
    files = [(str(p.relative_to(ROOT)), p.stat().st_size) for p in sorted(ROOT.rglob("*"))
             if p.is_file() and ".cache" not in p.parts]
    (ROOT / "MANIFEST.json").write_text(json.dumps(
        {"files": files, "converted": summary, "md5": MD5,
         "sources": {r: f for r, f, _ in JOBS}}, indent=1))
    (ROOT / ".complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    log("DONE total bytes", sum(s for _, s in files))
    (SRV / "DONE").write_text("0\n")


if __name__ == "__main__":
    SRV.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except BaseException:
        log("FAILED", traceback.format_exc())
        (SRV / "DONE").write_text("1\n")
