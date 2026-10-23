#!/usr/bin/env python3
"""Cut an `r2-weights.py plan` down to what the FastH3 NVFP4 apples-to-apples pod reads, and pack it for the pod's env.

    fasth3-nvfp4-plan.py PLAN.json OUT.b64 [--no-dense] [--only-v2]

The two ModelOpt NVFP4 trees' transformer/ (their NVFP4 vae/ is read by neither side: FastVideo d5253fd's H3 VAE and
ours have no ModelOpt path, both decode with the official VAE); h3-8step's bf16 text encoder (the reference's NVFP4
encoder is converted from it; ours streams it once per prompt), tokenizer, audio VAE, official VAE and, unless
--no-dense, its bf16 transformer for the served dense control. OUT is gzip + base64 (the pod gets it in
FV_PLAN_GZ_B64_<n>, at most 4 x 60 kB: nothing is staged for it).
"""
import base64, fnmatch, gzip, json, os, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
plan_path, out = args
keep = {
    "fasth3-v2-nvfp4": ("transformer/*",),
    "fasth3-4step-vsa-nvfp4": ("transformer/*",),
    "h3-8step": ("tokenizer/*", "audio_vae/*", "vae/*", "text_encoder/*")
    + (() if "--no-dense" in sys.argv else ("transformer/*",)),
}
if "--only-v2" in sys.argv:
    keep.pop("fasth3-4step-vsa-nvfp4")
plan = json.load(open(plan_path))
trees = []
for t in plan["trees"]:
    if t["dest"] not in keep:
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
