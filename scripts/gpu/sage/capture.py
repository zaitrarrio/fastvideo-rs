#!/usr/bin/env python3
"""Capture real self-attention Q/K/V (after QK-norm and RoPE) from FastWan2.1
1.3B at 832x480, 81 frames, for bench.py --real.

The text encoder (UMT5-XXL, 22 GB) is not downloaded: the cross-attention
context is all zeros (a null prompt). Self-attention statistics come from the
DiT's own projections, norms and RoPE on the video tokens, which is what the
quantizers see. Three DMD steps (t = 1000, 757, 522, FastWan's schedule);
layers 0, 15 and 29 are saved at steps 0 and 2 as [1, 12, 32760, 128] bf16.
"""
import os
import sys

import torch
import torch.nn.functional as F

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/root/w/fastwan"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/w/cap"
LAYERS = {0, 15, 29}
STEPS = {0, 2}
H = 12
os.makedirs(OUT, exist_ok=True)

from diffusers import WanTransformer3DModel  # noqa: E402
import diffusers.models.transformers.transformer_wan as tw  # noqa: E402

state = {"step": 0, "layer": 0, "saved": []}


def record(q, k, v):
    """q/k/v in any of [B,S,H,D] / [B,H,S,D]; save as [B,H,S,D] when self-attention."""
    if q.dim() != 4:
        return
    if q.shape[2] == H and q.shape[1] != H:      # [B,S,H,D]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if q.shape[1] != H or q.shape[2] != k.shape[2] or q.shape[2] < 4096:
        return                                   # cross-attention or not ours
    layer = state["layer"]
    state["layer"] += 1
    if layer in LAYERS and state["step"] in STEPS:
        path = os.path.join(OUT, f"fastwan_s{state['step']}_l{layer}.pt")
        torch.save({n: t.detach().to(torch.bfloat16).contiguous().cpu() for n, t in zip("qkv", (q, k, v))}, path)
        state["saved"].append(path)
        print("saved", path, tuple(q.shape), flush=True)


if hasattr(tw, "dispatch_attention_fn"):
    _orig = tw.dispatch_attention_fn

    def _hook(query, key, value, *a, **kw):
        record(query, key, value)
        return _orig(query, key, value, *a, **kw)
    tw.dispatch_attention_fn = _hook
else:
    _sdpa = F.scaled_dot_product_attention

    def _hook2(query, key, value, *a, **kw):
        record(query, key, value)
        return _sdpa(query, key, value, *a, **kw)
    F.scaled_dot_product_attention = _hook2

m = WanTransformer3DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).cuda().eval()
g = torch.Generator(device="cuda").manual_seed(7)
x = torch.randn((1, 16, 21, 60, 104), device="cuda", generator=g, dtype=torch.float32)
ctx = torch.zeros((1, 512, 4096), device="cuda", dtype=torch.bfloat16)
ts = [1000, 757, 522]
with torch.no_grad():
    for i, t in enumerate(ts):
        state["step"], state["layer"] = i, 0
        vel = m(hidden_states=x.bfloat16(), timestep=torch.tensor([t], device="cuda"),
                encoder_hidden_states=ctx, return_dict=False)[0].float()
        sig = t / 1000.0
        x0 = x - sig * vel
        if i + 1 < len(ts):
            sn = ts[i + 1] / 1000.0
            x = (1 - sn) * x0 + sn * torch.randn(x.shape, device="cuda", generator=g)
        print(f"step {i} t={t} layers_seen={state['layer']} x0_std={x0.std().item():.3f}", flush=True)
print("captured:", ",".join(state["saved"]))
