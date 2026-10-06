#!/usr/bin/env python3
"""NumPy transcription of diffusers' Cosmos3 T2V transformer forward.

Source: huggingface/diffusers main (2026-10-06),
src/diffusers/models/transformers/transformer_cosmos3.py
(`Cosmos3OmniTransformer.forward` with text + one all-noisy vision item, no
sound / action) and the position-id builders of
src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py. It runs the joint
packed sequence `[und | gen]` through both towers exactly as the reference
does (causal text attention, gen attending to text + vision keys), so it also
checks the Rust port's text-K/V cache decomposition.

Writes a safetensors file (checked in as `.st`; `*.safetensors` is
gitignored) with tiny weights under the Hub names plus `test.*` inputs and
the expected vision velocity. Run:
    python3 -I scripts/ref/cosmos3_reference.py OUT.st
"""

import json
import struct
import sys

import numpy as np

C = dict(hidden=32, layers=2, heads=4, kv_heads=2, head_dim=8, inter=48, latent_channel=3,
         patch=2, eps=1e-6, theta=10000.0, sections=(2, 1, 1), margin=15000, base_fps=24.0,
         timestep_scale=0.001, vocab=50)


def rms(x, w, eps):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def make_weights(rng):
    c = C
    h, q, kv, m, hd = c["hidden"], c["heads"] * c["head_dim"], c["kv_heads"] * c["head_dim"], c["inter"], c["head_dim"]
    pd = c["latent_channel"] * c["patch"] ** 2
    w = {}

    def lin(name, o, i, bias=False, s=0.2):
        w[name + ".weight"] = (rng.standard_normal((o, i)) * s).astype(np.float32)
        if bias:
            w[name + ".bias"] = (rng.standard_normal(o) * s).astype(np.float32)

    def norm(name, d):
        w[name + ".weight"] = (1.0 + 0.1 * rng.standard_normal(d)).astype(np.float32)

    w["embed_tokens.weight"] = rng.standard_normal((c["vocab"], h)).astype(np.float32)
    norm("norm_moe_gen", h)
    lin("proj_in", h, pd, bias=True)
    lin("proj_out", pd, h, bias=True)
    lin("time_embedder.linear_1", h, 256, bias=True, s=0.1)
    lin("time_embedder.linear_2", h, h, bias=True)
    for i in range(c["layers"]):
        p = f"layers.{i}."
        for n in ("input_layernorm", "input_layernorm_moe_gen", "post_attention_layernorm",
                  "post_attention_layernorm_moe_gen"):
            norm(p + n, h)
        lin(p + "self_attn.to_q", q, h)
        lin(p + "self_attn.to_k", kv, h)
        lin(p + "self_attn.to_v", kv, h)
        lin(p + "self_attn.to_out", h, q)
        lin(p + "self_attn.add_q_proj", q, h)
        lin(p + "self_attn.add_k_proj", kv, h)
        lin(p + "self_attn.add_v_proj", kv, h)
        lin(p + "self_attn.to_add_out", h, q)
        for n in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            norm(p + "self_attn." + n, hd)
        for mlp in ("mlp", "mlp_moe_gen"):
            lin(p + mlp + ".gate_proj", m, h)
            lin(p + mlp + ".up_proj", m, h)
            lin(p + mlp + ".down_proj", h, m)
    return w


def mrope_ids(und_len, grid, fps):
    # get_3d_mrope_ids_text_tokens(use_float_positions=True) then
    # get_3d_mrope_ids_vae_tokens(fps modulation, reset spatial ids).
    text = np.arange(und_len, dtype=np.float32)
    text_ids = np.stack([text] * 3)
    off = und_len + C["margin"]
    gt, gh, gw = grid
    tps = fps / 4
    base_tps = C["base_fps"] / 4
    fi = np.arange(gt, dtype=np.float32)
    scaled_t = (fi / np.float32(tps) * np.float32(base_tps) + np.float32(off)).astype(np.float32)
    t_index = np.repeat(scaled_t, gh * gw)
    h_index = np.tile(np.repeat(np.arange(gh), gw), gt).astype(np.float32)
    w_index = np.tile(np.arange(gw), gt * gh).astype(np.float32)
    vis = np.stack([t_index, h_index, w_index])
    return np.concatenate([text_ids, vis], axis=1)  # [3, N]


def rotary(position_ids):
    hd = C["head_dim"]
    inv = (1.0 / (C["theta"] ** (np.arange(0, hd, 2, dtype=np.float32) / hd))).astype(np.float32)
    freqs = (position_ids[:, :, None].astype(np.float32) * inv[None, None, :]).astype(np.float32)  # [3, N, hd/2]
    ft = freqs[0].copy()
    for dim, offset in ((1, 1), (2, 2)):
        length = C["sections"][dim] * 3
        ft[:, offset:length:3] = freqs[dim][:, offset:length:3]
    emb = np.concatenate([ft, ft], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def sdpa(q, k, v, causal):
    # q [S, Hq, D], k/v [Sk, Hkv, D]; enable_gqa: kv head = q head // groups.
    sq, hq, d = q.shape
    g = hq // k.shape[1]
    out = np.empty_like(q)
    for h in range(hq):
        s = q[:, h] @ k[:, h // g].T / np.sqrt(d)
        if causal:
            s = np.where(np.tril(np.ones((sq, k.shape[0]), dtype=bool)), s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        out[:, h] = s @ v[:, h // g]
    return out


def mlp(x, w, p):
    return (silu(x @ w[p + ".gate_proj.weight"].T) * (x @ w[p + ".up_proj.weight"].T)) @ w[p + ".down_proj.weight"].T


def forward(w, ids, latents, timestep, fps=24.0):
    c = C
    hq, hkv, hd, h = c["heads"], c["kv_heads"], c["head_dim"], c["hidden"]
    und_len = len(ids)
    _, ch, t, hh, ww = latents.shape
    p = c["patch"]
    hp, wp = -(-hh // p), -(-ww // p)
    lat = np.zeros((ch, t, hp * p, wp * p), dtype=np.float32)
    lat[:, :, :hh, :ww] = latents[0]
    tok = lat.reshape(ch, t, hp, p, wp, p)
    tok = np.einsum("cthpwq->thwpqc", tok).reshape(-1, p * p * ch)
    x = tok @ w["proj_in.weight"].T + w["proj_in.bias"]
    tt = np.float32(timestep * c["timestep_scale"])
    half = 128
    expo = -np.log(10000.0) * np.arange(half, dtype=np.float32) / half
    emb = tt * np.exp(expo)
    tp = np.concatenate([np.cos(emb), np.sin(emb)]).astype(np.float32)
    te = silu(tp @ w["time_embedder.linear_1.weight"].T + w["time_embedder.linear_1.bias"])
    te = te @ w["time_embedder.linear_2.weight"].T + w["time_embedder.linear_2.bias"]
    x = x + te[None, :]
    und = w["embed_tokens.weight"][np.array(ids)]
    gen = x
    cos, sin = rotary(mrope_ids(und_len, (t, hp, wp), fps))
    cu, su, cg, sg = cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:]
    for i in range(c["layers"]):
        pf = f"layers.{i}."
        a = pf + "self_attn."
        un = rms(und, w[pf + "input_layernorm.weight"], c["eps"])
        gn = rms(gen, w[pf + "input_layernorm_moe_gen.weight"], c["eps"])
        qu = (un @ w[a + "to_q.weight"].T).reshape(-1, hq, hd)
        ku = (un @ w[a + "to_k.weight"].T).reshape(-1, hkv, hd)
        vu = (un @ w[a + "to_v.weight"].T).reshape(-1, hkv, hd)
        qg = (gn @ w[a + "add_q_proj.weight"].T).reshape(-1, hq, hd)
        kg = (gn @ w[a + "add_k_proj.weight"].T).reshape(-1, hkv, hd)
        vg = (gn @ w[a + "add_v_proj.weight"].T).reshape(-1, hkv, hd)
        qu = rms(qu, w[a + "norm_q.weight"], c["eps"])
        ku = rms(ku, w[a + "norm_k.weight"], c["eps"])
        qg = rms(qg, w[a + "norm_added_q.weight"], c["eps"])
        kg = rms(kg, w[a + "norm_added_k.weight"], c["eps"])
        qu = qu * cu[:, None] + rotate_half(qu) * su[:, None]
        ku = ku * cu[:, None] + rotate_half(ku) * su[:, None]
        qg = qg * cg[:, None] + rotate_half(qg) * sg[:, None]
        kg = kg * cg[:, None] + rotate_half(kg) * sg[:, None]
        causal_out = sdpa(qu, ku, vu, causal=True).reshape(-1, hq * hd)
        full_out = sdpa(qg, np.concatenate([ku, kg]), np.concatenate([vu, vg]), causal=False).reshape(-1, hq * hd)
        und = und + causal_out @ w[a + "to_out.weight"].T
        gen = gen + full_out @ w[a + "to_add_out.weight"].T
        und = und + mlp(rms(und, w[pf + "post_attention_layernorm.weight"], c["eps"]), w, pf + "mlp")
        gen = gen + mlp(rms(gen, w[pf + "post_attention_layernorm_moe_gen.weight"], c["eps"]), w, pf + "mlp_moe_gen")
    gen = rms(gen, w["norm_moe_gen.weight"], c["eps"])
    pred = gen @ w["proj_out.weight"].T + w["proj_out.bias"]
    pr = pred.reshape(t, hp, wp, p, p, ch)
    lat = np.einsum("thwpqc->cthpwq", pr).reshape(ch, t, hp * p, wp * p)[:, :, :hh, :ww]
    return lat[None].astype(np.float32)


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
    rng = np.random.default_rng(20261007)
    w = make_weights(rng)
    ids = [3, 7, 11, 2]
    latents = rng.standard_normal((1, C["latent_channel"], 2, 3, 4)).astype(np.float32)
    timestep = 437
    out = forward(w, ids, latents, timestep)
    t = dict(w)
    t["test.ids"] = np.array(ids, dtype=np.float32)
    t["test.latents"] = latents
    t["test.timestep"] = np.array([timestep], dtype=np.float32)
    t["test.expected"] = out
    write_safetensors(sys.argv[1], t)
    print("wrote", sys.argv[1], "mean |v|", float(np.abs(out).mean()))


if __name__ == "__main__":
    main()
