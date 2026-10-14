#!/usr/bin/env python3
"""Cut an `r2-weights.py plan` down to what the FastH3 Trim NVFP4 validation pod reads, and pack it for the pod's env.

    h3-trim-nvfp4-plan.py PLAN.json MODE OUT.b64

MODE `volume`: h3-base comes from the EU volume, so only the Trim trees are fetched; `r2`: also h3-base's tokenizer,
audio VAE and bf16 text encoder (streamed once per prompt, then cached; not its 70 GB transformer). The FP8 encoder
tree is not used: it checks its bf16 source shards, which a pod without them cannot pass. The Trim tree's ComfyUI INT8 VAE file
is never read. OUT is gzip + base64 (the pod gets it in FV_PLAN_GZ_B64_<n>, at most 4 x 60 kB: nothing is staged anywhere).
"""
import base64, fnmatch, gzip, json, os, sys

plan_path, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
plan = json.load(open(plan_path))
keep = {
    "fasth3-trim-8step": ("transformer/*", "vae/config.json", "vae/diffusion_pytorch_model*", "scheduler/*",
                          "audio_scheduler/*", "*.json"),
    "fasth3-trim-8step-nvfp4": ("diffusion_models/*",),
    "h3-base": ("tokenizer/*", "audio_vae/*", "model_index.json", "text_encoder/*"),
}
trees = []
for t in plan["trees"]:
    if t["dest"] not in keep or (mode == "volume" and t["dest"].startswith("h3-base")):
        continue
    t["files"] = [f for f in t["files"] if any(fnmatch.fnmatch(f["path"], g) for g in keep[t["dest"]])]
    t["total_bytes"] = sum(f["size"] for f in t["files"])
    trees.append(t)
plan["trees"] = trees
raw = json.dumps(plan, separators=(",", ":")).encode()
b64 = base64.b64encode(gzip.compress(raw, 9))
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "wb") as f:
    f.write(b64)
if len(b64) > 4 * 60000:
    sys.exit(f"plan too large for the pod env: {len(b64)} bytes")
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in trees)
      + f"; env {len(b64)} bytes", file=sys.stderr)
