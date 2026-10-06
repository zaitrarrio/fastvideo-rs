#!/usr/bin/env python3
"""NumPy transcription of LingBot-Video's DiT forward, for a golden fixture.

Source: NVlabs/Sana `sol-engine`,
models/lingbot_video/baseline/lingbot_src/lingbot_video/transformer_lingbot_video.py
(`LingBotVideoTransformer3DModel.forward`, B = 1, no padding mask), read
line by line; nothing here comes from the Rust port. The router gate weights
are rounded to bf16 as `top_scores.to(tokens.dtype)` does in the bf16 model.

Writes a safetensors file (checked in as `.st`; `*.safetensors` is gitignored) with the tiny-MoE weights under their Hub names
plus `test.latents`, `test.text`, `test.timestep` and `test.expected`; the
Rust test `lingbot::transformer::tests::golden_tiny_moe_matches_numpy_reference`
loads it. Run: python3 -I scripts/ref/lingbot_reference.py OUT.safetensors
"""

import json
import struct
import sys

import numpy as np

CFG = dict(
    in_channels=4, out_channels=4, hidden=32, heads=2, depth=2, inter=64,
    text_dim=24, freq_dim=16, patch=(1, 2, 2), theta=256.0, axes_dims=(4, 6, 6),
    eps=1e-6, experts=8, top_k=2, moe_inter=16, n_shared=1, n_group=4,
    topk_group=2, route_scale=2.5, mlp_only_layers=(0,),
)


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    b = x.view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def rms(x, w, eps):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (w * (x / np.sqrt(var + eps))).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def linear(x, w, b=None):
    y = x @ w.T
    return y + b if b is not None else y


def make_weights(rng):
    c = CFG
    h, hd = c["hidden"], c["hidden"] // c["heads"]
    pd = c["in_channels"] * int(np.prod(c["patch"]))
    od = c["out_channels"] * int(np.prod(c["patch"]))
    w = {}

    def lin(name, o, i, bias=True, s=0.15):
        w[name + ".weight"] = (rng.standard_normal((o, i)) * s).astype(np.float32)
        if bias:
            w[name + ".bias"] = (rng.standard_normal(o) * s).astype(np.float32)

    def norm(name, d):
        w[name + ".weight"] = (1.0 + 0.1 * rng.standard_normal(d)).astype(np.float32)

    lin("patch_embedder", h, pd)
    lin("time_embedder.linear_1", h, c["freq_dim"])
    lin("time_embedder.linear_2", h, h)
    lin("time_modulation.1", 6 * h, h)
    norm("text_embedder.norm", c["text_dim"])
    lin("text_embedder.linear_1", h, c["text_dim"])
    lin("text_embedder.linear_2", h, h)
    lin("norm_out_modulation.1", 2 * h, h)
    lin("proj_out", od, h)
    for i in range(c["depth"]):
        p = f"blocks.{i}."
        w[p + "scale_shift_table"] = (0.1 * rng.standard_normal((1, 6 * h))).astype(np.float32)
        for n in ("norm1", "norm2", "norm_post_attn", "norm_post_ffn"):
            norm(p + n, h)
        for n in ("to_q", "to_k", "to_v"):
            lin(p + "attn." + n, h, h, bias=False)
        lin(p + "attn.to_out", h, h)
        norm(p + "attn.norm_q", hd)
        norm(p + "attn.norm_k", hd)
        if i in c["mlp_only_layers"]:
            lin(p + "ffn.gate_proj", c["inter"], h, bias=False)
            lin(p + "ffn.up_proj", c["inter"], h, bias=False)
            lin(p + "ffn.down_proj", h, c["inter"], bias=False)
        else:
            e, m = c["experts"], c["moe_inter"]
            w[p + "ffn.router.weight"] = (rng.standard_normal((e, h)) * 0.5).astype(np.float32)
            w[p + "ffn.router.e_score_correction_bias"] = (0.05 * rng.standard_normal(e)).astype(np.float32)
            w[p + "ffn.experts.w1"] = (rng.standard_normal((e, m, h)) * 0.15).astype(np.float32)
            w[p + "ffn.experts.w2"] = (rng.standard_normal((e, h, m)) * 0.15).astype(np.float32)
            w[p + "ffn.experts.w3"] = (rng.standard_normal((e, m, h)) * 0.15).astype(np.float32)
            s = m * c["n_shared"]
            lin(p + "ffn.shared_experts.gate_proj", s, h, bias=False)
            lin(p + "ffn.shared_experts.up_proj", s, h, bias=False)
            lin(p + "ffn.shared_experts.down_proj", h, s, bias=False)
    return w


def freqs_cis(dims, positions, theta):
    # precompute_freqs_cis + forward: torch.cat([freqs_cis[i][pos[:, i]] ...], -1)
    parts = []
    for i, d in enumerate(dims):
        inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        ang = np.outer(positions[:, i].astype(np.float64), inv).astype(np.float32)
        parts.append(np.cos(ang) + 1j * np.sin(ang))
    return np.concatenate(parts, axis=-1).astype(np.complex64)  # (S, hd/2)


def apply_rotary(x, fc):
    # x: (S, H, D); view_as_complex over the last dim pairs.
    xc = x.astype(np.float32).reshape(*x.shape[:-1], -1, 2)
    xc = xc[..., 0] + 1j * xc[..., 1]
    out = xc * fc[:, None, :]
    return np.stack([out.real, out.imag], axis=-1).reshape(x.shape).astype(np.float32)


def router(tokens, w, bias):
    c = CFG
    logits = tokens.astype(np.float32) @ w.astype(np.float32).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    sfc = scores + bias[None, :]
    n, e = sfc.shape
    per = e // c["n_group"]
    grouped = sfc.reshape(n, c["n_group"], per)
    gs = -np.sort(-grouped, axis=-1)[..., :2].sum(-1)
    top_idx = np.empty((n, c["top_k"]), dtype=np.int64)
    for t in range(n):
        keep = np.argsort(-gs[t], kind="stable")[: c["topk_group"]]
        mask = np.zeros(c["n_group"], dtype=bool)
        mask[keep] = True
        masked = np.where(np.repeat(mask, per), sfc[t], -np.inf)
        top_idx[t] = np.argsort(-masked, kind="stable")[: c["top_k"]]
    top = np.take_along_axis(scores, top_idx, axis=1)
    top = top / (top.sum(-1, keepdims=True) + 1e-20)
    top = top * c["route_scale"]
    return top_idx, bf16(top)


def mlp(x, g, u, d):
    return linear(silu(linear(x, g)) * linear(x, u), d)


def forward(w, latents, text_hidden, timestep):
    c = CFG
    h, heads = c["hidden"], c["heads"]
    hd = h // heads
    _, C, T, H, W = latents.shape
    pF, pH, pW = c["patch"]
    gt, gh, gw = T // pF, H // pH, W // pW
    n_video = gt * gh * gw
    L = text_hidden.shape[1]
    pt = latents.reshape(1, C, gt, pF, gh, pH, gw, pW).transpose(0, 2, 4, 6, 3, 5, 7, 1)
    pt = pt.reshape(1, n_video, pF * pH * pW * C)
    x = linear(pt, w["patch_embedder.weight"], w["patch_embedder.bias"])
    tn = rms(text_hidden, w["text_embedder.norm.weight"], 1e-6)
    text = linear(silu(linear(tn, w["text_embedder.linear_1.weight"], w["text_embedder.linear_1.bias"])),
                  w["text_embedder.linear_2.weight"], w["text_embedder.linear_2.bias"])
    joint = np.concatenate([x, text], axis=1)[0]  # (S, h)
    S = joint.shape[0]
    tt = np.arange(gt) + (L + 1)
    grid = np.stack(np.meshgrid(tt, np.arange(gh), np.arange(gw), indexing="ij"), -1).reshape(-1, 3)
    text_pos = np.stack([np.arange(L) + 1, np.zeros(L, int), np.zeros(L, int)], -1)
    fc = freqs_cis(c["axes_dims"], np.concatenate([grid, text_pos], 0), c["theta"])

    half = c["freq_dim"] // 2
    expo = -np.log(10000.0) * np.arange(half, dtype=np.float32) / half
    emb = np.float32(timestep) * np.exp(expo)
    tproj = np.concatenate([np.cos(emb), np.sin(emb)])  # flip_sin_to_cos
    t_emb = linear(silu(linear(tproj, w["time_embedder.linear_1.weight"], w["time_embedder.linear_1.bias"])),
                   w["time_embedder.linear_2.weight"], w["time_embedder.linear_2.bias"])
    temb6 = linear(silu(t_emb), w["time_modulation.1.weight"], w["time_modulation.1.bias"])

    for i in range(c["depth"]):
        p = f"blocks.{i}."
        mod = temb6[None, :] + w[p + "scale_shift_table"]
        sm, scm, gm, sf, scf, gf = np.split(mod, 6, axis=-1)
        gm, gf = np.tanh(gm), np.tanh(gf)
        scm, scf = 1.0 + scm, 1.0 + scf
        a_in = rms(joint, w[p + "norm1.weight"], c["eps"]) * scm + sm
        q = linear(a_in, w[p + "attn.to_q.weight"]).reshape(S, heads, hd)
        k = linear(a_in, w[p + "attn.to_k.weight"]).reshape(S, heads, hd)
        v = linear(a_in, w[p + "attn.to_v.weight"]).reshape(S, heads, hd)
        q = apply_rotary(rms(q, w[p + "attn.norm_q.weight"], c["eps"]), fc)
        k = apply_rotary(rms(k, w[p + "attn.norm_k.weight"], c["eps"]), fc)
        out = np.empty_like(q)
        for hh in range(heads):
            s = q[:, hh] @ k[:, hh].T / np.sqrt(hd)
            s = np.exp(s - s.max(-1, keepdims=True))
            s = s / s.sum(-1, keepdims=True)
            out[:, hh] = s @ v[:, hh]
        attn = linear(out.reshape(S, h), w[p + "attn.to_out.weight"], w[p + "attn.to_out.bias"])
        joint = joint + gm * rms(attn, w[p + "norm_post_attn.weight"], c["eps"])
        f_in = rms(joint, w[p + "norm2.weight"], c["eps"]) * scf + sf
        if i in c["mlp_only_layers"]:
            f = mlp(f_in, w[p + "ffn.gate_proj.weight"], w[p + "ffn.up_proj.weight"], w[p + "ffn.down_proj.weight"])
        else:
            idx, wt = router(f_in, w[p + "ffn.router.weight"], w[p + "ffn.router.e_score_correction_bias"])
            f = np.zeros_like(f_in)
            for t in range(S):
                for s_ in range(c["top_k"]):
                    e = idx[t, s_]
                    f[t] += wt[t, s_] * mlp(f_in[t], w[p + "ffn.experts.w1"][e], w[p + "ffn.experts.w3"][e],
                                            w[p + "ffn.experts.w2"][e])
            f = f + mlp(f_in, w[p + "ffn.shared_experts.gate_proj.weight"],
                        w[p + "ffn.shared_experts.up_proj.weight"], w[p + "ffn.shared_experts.down_proj.weight"])
        joint = joint + gf * rms(f, w[p + "norm_post_ffn.weight"], c["eps"])

    fm = linear(silu(t_emb), w["norm_out_modulation.1.weight"], w["norm_out_modulation.1.bias"])
    shift, scale = np.split(fm, 2)
    mu = joint.mean(-1, keepdims=True)
    var = ((joint - mu) ** 2).mean(-1, keepdims=True)
    fin = (joint - mu) / np.sqrt(var + c["eps"]) * (1.0 + scale) + shift
    proj = linear(fin, w["proj_out.weight"], w["proj_out.bias"])[:n_video]
    Co = c["out_channels"]
    y = proj.reshape(1, gt, gh, gw, pF, pH, pW, Co).transpose(0, 7, 1, 4, 2, 5, 3, 6)
    return y.reshape(1, Co, T, H, W).astype(np.float32)


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k], dtype=np.float32)
        b = a.tobytes()
        header[k] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def main():
    rng = np.random.default_rng(20261006)
    w = make_weights(rng)
    latents = rng.standard_normal((1, CFG["in_channels"], 2, 4, 6)).astype(np.float32)
    text_hidden = rng.standard_normal((1, 3, CFG["text_dim"])).astype(np.float32)
    timestep = 437.0
    out = forward(w, latents, text_hidden, timestep)
    t = dict(w)
    t["test.latents"] = latents
    t["test.text"] = text_hidden
    t["test.timestep"] = np.array([timestep], dtype=np.float32)
    t["test.expected"] = out
    write_safetensors(sys.argv[1], t)
    print("wrote", sys.argv[1], "expected |y| mean", float(np.abs(out).mean()))


if __name__ == "__main__":
    main()
